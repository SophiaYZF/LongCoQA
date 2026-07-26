from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from longcoqa.utils.hashing import sha256_file, stable_json_hash
from longcoqa.utils.io import atomic_write_json, atomic_write_jsonl, read_jsonl

INSUFFICIENT_ANSWER = "提供的证据不足以回答该问题。"
SUFFICIENCY_LEVELS = {"sufficient", "partial", "insufficient"}
_REQUIRED_FIELDS = {
    "sample_id",
    "sufficiency",
    "selected_evidence_ids",
    "support_points",
    "final_answer",
    "risk_flags",
}
_EVIDENCE_HEADING = re.compile(r"\[EVIDENCE\s+([^\]]+)\]")

CLEAN_SYSTEM_PROMPT = """你是《哈利·波特》系列小说的证据约束问答助手。
只依据当前提供的证据回答；对话历史只用于理解指代，不作为事实来源。
只陈述证据直接支持的内容，信息不完整时明确限定，证据不足时拒答。
答案末尾只引用实际使用的证据编号。"""

EVIDENCE_AWARE_SYSTEM_PROMPT = """你是《哈利·波特》系列小说的证据约束问答助手。
只依据当前证据判断充分性、选择证据、列出原子支持点并回答。
必须使用规定的“充分性/选用证据/支持点/最终答案”格式；对话历史只用于指代消解。
证据不足时拒答，不得凭模型记忆补充。"""

TEACHER_SYSTEM_PROMPT = """你是证据约束问答数据的高级标注员。你的任务不是复述候选答案，而是只根据当前提供的证据重建可靠答案。

规则：
1. 对话历史仅用于指代消解，不能作为事实证据。
2. 原始答案是不可信候选，可能含幻觉、遗漏或与证据冲突。
3. 不得依赖模型记忆补全证据中没有的信息。
4. sufficient 表示核心问题可被完整回答；partial 表示只能回答一部分，并在答案中明确缺失；insufficient 表示无法可靠回答。
5. 每个 support point 必须是对应 evidence 直接支持的原子事实；只选择最终答案实际使用的 evidence。
6. insufficient 时 selected_evidence_ids 和 support_points 必须为空，final_answer 必须严格为“提供的证据不足以回答该问题。”
7. 只输出一个 JSON 对象，不要 Markdown 或额外解释。

JSON schema：
{
  "sample_id": "原样复制",
  "sufficiency": "sufficient|partial|insufficient",
  "selected_evidence_ids": ["不带 EVIDENCE 前缀的 ID"],
  "support_points": [{"evidence_id": "ID", "claim": "原子支持事实"}],
  "final_answer": "仅包含有证据支持的中文答案，不在此字段添加引用行",
  "risk_flags": ["可选风险，例如 evidence_conflict、ambiguous、multi_hop"]
}"""


def evidence_ids_from_instruction(instruction: str) -> list[str]:
    """Return evidence labels in display order without the EVIDENCE prefix."""
    return list(dict.fromkeys(match.strip() for match in _EVIDENCE_HEADING.findall(instruction)))


def _normalize_evidence_id(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("evidence ID must be a string")
    normalized = value.strip()
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1].strip()
    if normalized.upper().startswith("EVIDENCE "):
        normalized = normalized[len("EVIDENCE ") :].strip()
    if not normalized or any(character in normalized for character in "[]\n\r"):
        raise ValueError(f"invalid evidence ID: {value!r}")
    return normalized


def _decode_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as error:
        raise ValueError(f"teacher response is not valid JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ValueError("teacher response must be one JSON object")
    return payload


def parse_teacher_response(
    text: str,
    *,
    expected_sample_id: str,
    available_evidence_ids: set[str],
) -> dict[str, Any]:
    payload = _decode_json_object(text)
    missing = _REQUIRED_FIELDS.difference(payload)
    if missing:
        raise ValueError(f"teacher response missing fields: {sorted(missing)}")
    if str(payload["sample_id"]) != expected_sample_id:
        raise ValueError("teacher response sample_id mismatch")

    sufficiency = str(payload["sufficiency"]).strip().lower()
    if sufficiency not in SUFFICIENCY_LEVELS:
        raise ValueError(f"invalid sufficiency: {sufficiency!r}")

    raw_selected = payload["selected_evidence_ids"]
    if not isinstance(raw_selected, list):
        raise ValueError("selected_evidence_ids must be a list")
    selected = [_normalize_evidence_id(value) for value in raw_selected]
    if len(selected) != len(set(selected)):
        raise ValueError("selected_evidence_ids contains duplicates")
    unknown = set(selected).difference(available_evidence_ids)
    if unknown:
        raise ValueError(f"unknown evidence IDs: {sorted(unknown)}")

    raw_points = payload["support_points"]
    if not isinstance(raw_points, list):
        raise ValueError("support_points must be a list")
    points: list[dict[str, str]] = []
    for raw_point in raw_points:
        if not isinstance(raw_point, dict):
            raise ValueError("each support point must be an object")
        evidence_id = _normalize_evidence_id(raw_point.get("evidence_id"))
        claim = str(raw_point.get("claim", "")).strip()
        if not claim:
            raise ValueError("support-point claim must be non-empty")
        if evidence_id not in available_evidence_ids:
            raise ValueError(f"unknown evidence ID in support point: {evidence_id}")
        points.append({"evidence_id": evidence_id, "claim": claim})
    if sufficiency == "insufficient" and (selected or points):
        raise ValueError("insufficient label must not select evidence")
    if {point["evidence_id"] for point in points} != set(selected):
        raise ValueError("support-point evidence IDs must equal selected evidence IDs")

    final_answer = str(payload["final_answer"]).strip()
    risk_flags = payload["risk_flags"]
    if not isinstance(risk_flags, list) or not all(isinstance(flag, str) for flag in risk_flags):
        raise ValueError("risk_flags must be a list of strings")
    normalized_risks = list(dict.fromkeys(flag.strip() for flag in risk_flags if flag.strip()))

    if sufficiency == "insufficient":
        if final_answer != INSUFFICIENT_ANSWER:
            raise ValueError("insufficient label must use the exact refusal answer")
    else:
        if not selected or not points:
            raise ValueError(f"{sufficiency} label must select supporting evidence")
        if not final_answer or final_answer == INSUFFICIENT_ANSWER:
            raise ValueError(f"{sufficiency} label must contain a supported answer")

    return {
        "sample_id": expected_sample_id,
        "sufficiency": sufficiency,
        "selected_evidence_ids": selected,
        "support_points": points,
        "final_answer": final_answer,
        "risk_flags": normalized_risks,
    }


def build_teacher_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    metadata = row.get("metadata", {})
    sample_id = str(metadata.get("sample_id", ""))
    if not sample_id:
        raise ValueError("row metadata is missing sample_id")
    history = row.get("history", [])
    history_text = json.dumps(history, ensure_ascii=False, indent=2)
    user = f"""[SAMPLE_ID]
{sample_id}

[对话历史：仅用于指代消解，不是事实证据]
{history_text}

[当前证据与问题]
{row.get('instruction', '')}

[原始答案：不可信候选，只用于发现可能的遗漏或冲突]
{row.get('output', '')}

请按系统消息中的 JSON schema 重建标注。不得依赖模型记忆。"""
    return [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def render_clean_output(label: dict[str, Any]) -> str:
    answer = str(label["final_answer"]).strip()
    selected = [str(value) for value in label["selected_evidence_ids"]]
    if not selected:
        return answer
    citations = ", ".join(f"EVIDENCE {evidence_id}" for evidence_id in selected)
    return f"{answer}\n\n引用：[{citations}]"


def render_evidence_aware_output(label: dict[str, Any]) -> str:
    selected = [str(value) for value in label["selected_evidence_ids"]]
    selected_text = ", ".join(f"EVIDENCE {evidence_id}" for evidence_id in selected)
    points = label["support_points"]
    point_text = "\n".join(
        f"- [EVIDENCE {point['evidence_id']}] {point['claim']}" for point in points
    )
    if not point_text:
        point_text = "- 无"
    return (
        f"充分性: {label['sufficiency']}\n"
        f"选用证据: [{selected_text}]\n"
        f"支持点:\n{point_text}\n"
        f"最终答案: {render_clean_output(label)}"
    )


def load_reconstruction_rows(input_paths: dict[str, Path]) -> list[dict[str, Any]]:
    if set(input_paths) != {"train", "dev"}:
        raise ValueError("reconstruction inputs must contain exactly train and dev")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for split in ("train", "dev"):
        for source in read_jsonl(input_paths[split]):
            row = dict(source)
            metadata = dict(row.get("metadata", {}))
            sample_id = str(metadata.get("sample_id", ""))
            if not sample_id:
                raise ValueError(f"{split} row is missing sample_id")
            if sample_id in seen:
                raise ValueError(f"duplicate sample_id across reconstruction inputs: {sample_id}")
            if str(metadata.get("split")) != split:
                raise ValueError(
                    f"split mismatch for {sample_id}: expected {split}, got {metadata.get('split')}"
                )
            if split == "test" or str(metadata.get("split")) == "test":
                raise ValueError("test rows are forbidden in reconstruction")
            if str(metadata.get("variant")) != "raft":
                raise ValueError(f"expected RAFT input for {sample_id}")
            seen.add(sample_id)
            row["metadata"] = metadata
            rows.append(row)
    return rows


def _checkpoint_path(checkpoint_dir: Path, sample_id: str) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "__", sample_id)
    return checkpoint_dir / f"{safe_id}.json"


def _result_metadata(result: object) -> dict[str, Any]:
    if is_dataclass(result):
        raw = asdict(result)
    elif hasattr(result, "__dict__"):
        raw = dict(vars(result))
    else:
        raw = {}
    return {
        key: raw.get(key)
        for key in (
            "provider",
            "model",
            "usage",
            "finish_reason",
            "request_hash",
            "cached",
            "latency_seconds",
        )
        if key in raw
    }


def _load_valid_checkpoint(
    row: dict[str, Any], checkpoint_dir: Path
) -> dict[str, Any] | None:
    sample_id = str(row["metadata"]["sample_id"])
    messages = build_teacher_messages(row)
    prompt_hash = stable_json_hash(messages)
    path = _checkpoint_path(checkpoint_dir, sample_id)
    if not path.is_file():
        return None
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        if checkpoint.get("status") != "ok" or checkpoint.get("prompt_hash") != prompt_hash:
            return None
        return parse_teacher_response(
            str(checkpoint["raw_response"]),
            expected_sample_id=sample_id,
            available_evidence_ids=set(evidence_ids_from_instruction(str(row["instruction"]))),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def load_checkpoint_labels(
    rows: list[dict[str, Any]], checkpoint_dir: Path
) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for row in rows:
        label = _load_valid_checkpoint(row, checkpoint_dir)
        if label is not None:
            labels[str(row["metadata"]["sample_id"])] = label
    return labels


def run_reconstruction(
    rows: list[dict[str, Any]],
    *,
    client: Any,
    checkpoint_dir: Path,
    max_workers: int,
    max_attempts: int = 3,
    max_tokens: int = 1400,
) -> dict[str, Any]:
    if max_workers < 1 or max_attempts < 1:
        raise ValueError("max_workers and max_attempts must be positive")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    labels = load_checkpoint_labels(rows, checkpoint_dir)
    reused = len(labels)
    pending = [row for row in rows if str(row["metadata"]["sample_id"]) not in labels]
    failures: dict[str, str] = {}

    def generate_one(row: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str | None]:
        sample_id = str(row["metadata"]["sample_id"])
        messages = build_teacher_messages(row)
        prompt_hash = stable_json_hash(messages)
        available = set(evidence_ids_from_instruction(str(row["instruction"])))
        error_text = ""
        raw_response = ""
        result_metadata: dict[str, Any] = {}
        for attempt in range(1, max_attempts + 1):
            attempt_messages = list(messages)
            if error_text:
                attempt_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "上一次输出未通过格式验证。请重新输出完整 JSON；错误为："
                            + error_text[:500]
                        ),
                    }
                )
            try:
                result = client.generate(
                    attempt_messages,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    extra_body={"response_format": {"type": "json_object"}},
                )
                raw_response = str(result.text)
                result_metadata = _result_metadata(result)
                label = parse_teacher_response(
                    raw_response,
                    expected_sample_id=sample_id,
                    available_evidence_ids=available,
                )
                atomic_write_json(
                    _checkpoint_path(checkpoint_dir, sample_id),
                    {
                        "schema_version": "longcoqa-evidence-reconstruction-checkpoint-v1",
                        "sample_id": sample_id,
                        "status": "ok",
                        "attempts": attempt,
                        "prompt_hash": prompt_hash,
                        "available_evidence_ids": sorted(available),
                        "raw_response": raw_response,
                        "label": label,
                        "generation": result_metadata,
                    },
                )
                return sample_id, label, None
            except Exception as error:  # provider and strict parser failures share retry policy
                error_text = f"{type(error).__name__}: {error}"
        atomic_write_json(
            _checkpoint_path(checkpoint_dir, sample_id),
            {
                "schema_version": "longcoqa-evidence-reconstruction-checkpoint-v1",
                "sample_id": sample_id,
                "status": "failed",
                "attempts": max_attempts,
                "prompt_hash": prompt_hash,
                "available_evidence_ids": sorted(available),
                "raw_response": raw_response,
                "error": error_text,
                "generation": result_metadata,
            },
        )
        return sample_id, None, error_text

    if max_workers == 1:
        results = [generate_one(row) for row in pending]
    else:
        results = []
        executor = ThreadPoolExecutor(max_workers=max_workers)
        futures = [executor.submit(generate_one, row) for row in pending]
        try:
            for future in as_completed(futures):
                results.append(future.result())
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
    for sample_id, label, error in results:
        if label is not None:
            labels[sample_id] = label
        else:
            failures[sample_id] = str(error)
    return {
        "requested": len(rows),
        "accepted": len(labels),
        "generated": len(labels) - reused,
        "reused": reused,
        "failed": len(failures),
        "failures": failures,
        "labels": labels,
    }


def _question_from_instruction(instruction: str) -> str:
    marker = "[QUESTION]"
    if marker not in instruction:
        raise ValueError("instruction is missing [QUESTION]")
    question = instruction.rsplit(marker, 1)[1].strip()
    if not question:
        raise ValueError("instruction contains an empty question")
    return question


def _training_example(
    source: dict[str, Any],
    *,
    variant: str,
    system: str,
    output: str,
    history: list[list[str]],
    label: dict[str, Any],
) -> dict[str, Any]:
    metadata = dict(source["metadata"])
    metadata.update(
        {
            "source_variant": metadata.get("variant"),
            "variant": variant,
            "teacher_sufficiency": label["sufficiency"],
            "selected_evidence_ids": label["selected_evidence_ids"],
            "teacher_risk_flags": label["risk_flags"],
        }
    )
    return {
        "instruction": source["instruction"],
        "input": source.get("input", ""),
        "output": output,
        "system": system,
        "history": history,
        "metadata": metadata,
    }


def build_aligned_datasets(
    rows: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = ("raw_answer", "clean_answer", "evidence_aware")
    rendered: dict[str, dict[str, list[dict[str, Any]]]] = {
        variant: {"train": [], "dev": []} for variant in variants
    }
    accepted_rows = [
        row for row in rows if str(row["metadata"]["sample_id"]) in labels
    ]
    rejected = [
        str(row["metadata"]["sample_id"])
        for row in rows
        if str(row["metadata"]["sample_id"]) not in labels
    ]

    histories: dict[tuple[str, str, int], list[list[str]]] = {}
    ordered = sorted(
        accepted_rows,
        key=lambda row: (
            str(row["metadata"]["split"]),
            str(row["metadata"]["question_type"]),
            int(row["metadata"]["dialogue_id"]),
            int(row["metadata"]["turn"]),
        ),
    )
    examples_by_id: dict[str, dict[str, dict[str, Any]]] = {}
    for row in ordered:
        metadata = row["metadata"]
        sample_id = str(metadata["sample_id"])
        split = str(metadata["split"])
        if split not in {"train", "dev"}:
            raise ValueError(f"forbidden split in aligned dataset: {split}")
        key = (split, str(metadata["question_type"]), int(metadata["dialogue_id"]))
        history = [list(item) for item in histories.get(key, [])]
        label = labels[sample_id]
        clean_output = render_clean_output(label)
        examples_by_id[sample_id] = {
            "raw_answer": _training_example(
                row,
                variant="raw_answer",
                system=CLEAN_SYSTEM_PROMPT,
                output=str(row["output"]),
                history=history,
                label=label,
            ),
            "clean_answer": _training_example(
                row,
                variant="clean_answer",
                system=CLEAN_SYSTEM_PROMPT,
                output=clean_output,
                history=history,
                label=label,
            ),
            "evidence_aware": _training_example(
                row,
                variant="evidence_aware",
                system=EVIDENCE_AWARE_SYSTEM_PROMPT,
                output=render_evidence_aware_output(label),
                history=history,
                label=label,
            ),
        }
        histories.setdefault(key, []).append(
            [_question_from_instruction(str(row["instruction"])), clean_output]
        )

    for row in accepted_rows:
        sample_id = str(row["metadata"]["sample_id"])
        split = str(row["metadata"]["split"])
        for variant in variants:
            rendered[variant][split].append(examples_by_id[sample_id][variant])

    files: dict[str, dict[str, str]] = {variant: {} for variant in variants}
    checksums: dict[str, dict[str, str]] = {variant: {} for variant in variants}
    for variant in variants:
        for split in ("train", "dev"):
            path = output_dir / f"{variant}_{split}.jsonl"
            atomic_write_jsonl(path, rendered[variant][split])
            files[variant][split] = str(path)
            checksums[variant][split] = sha256_file(path)

    dataset_info = {
        f"longcoqa_evidence_{variant}_{split}": {
            "file_name": f"{variant}_{split}.jsonl",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
                "history": "history",
            },
        }
        for variant in variants
        for split in ("train", "dev")
    }
    atomic_write_json(output_dir / "dataset_info.json", dataset_info)
    counts = {
        split: len(rendered["raw_answer"][split]) for split in ("train", "dev")
    }
    id_hashes = {
        split: stable_json_hash(
            [
                str(row["metadata"]["sample_id"])
                for row in rendered["raw_answer"][split]
            ]
        )
        for split in ("train", "dev")
    }
    return {
        "schema_version": "longcoqa-evidence-sft-data-v1",
        "counts": counts,
        "accepted": sum(counts.values()),
        "rejected": len(rejected),
        "rejected_sample_ids": rejected,
        "sample_ids_hash": id_hashes,
        "files": files,
        "file_checksums": checksums,
    }


def select_human_audit_rows(
    records: list[dict[str, Any]], *, count: int, seed: int
) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("audit count must be positive")
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        sample_id = str(record.get("sample_id", ""))
        if not sample_id:
            raise ValueError("audit record is missing sample_id")
        if sample_id in by_id:
            raise ValueError(f"duplicate audit sample_id: {sample_id}")
        by_id[sample_id] = record

    def order_key(record: dict[str, Any]) -> tuple[int, str]:
        risky = bool(record.get("risk_flags"))
        digest = stable_json_hash({"seed": seed, "sample_id": record["sample_id"]})
        return (0 if risky else 1, digest)

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in by_id.values():
        key = (str(record.get("question_type")), str(record.get("sufficiency")))
        groups.setdefault(key, []).append(record)
    for values in groups.values():
        values.sort(key=order_key)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for key in sorted(groups):
        if len(selected) >= min(count, len(by_id)):
            break
        record = groups[key][0]
        selected.append(record)
        selected_ids.add(str(record["sample_id"]))

    remaining = [record for record in by_id.values() if str(record["sample_id"]) not in selected_ids]
    remaining.sort(key=order_key)
    selected.extend(remaining[: max(0, min(count, len(by_id)) - len(selected))])
    return selected


def build_quality_report(
    rows: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    *,
    data_report: dict[str, Any],
    input_paths: dict[str, Path],
) -> dict[str, Any]:
    accepted_rows = [
        row for row in rows if str(row["metadata"]["sample_id"]) in labels
    ]
    by_split = Counter(str(row["metadata"]["split"]) for row in accepted_rows)
    by_type = Counter(str(row["metadata"]["question_type"]) for row in accepted_rows)
    by_mode = Counter(str(row["metadata"].get("raft_mode", "unknown")) for row in accepted_rows)
    by_sufficiency = Counter(
        str(labels[str(row["metadata"]["sample_id"])]["sufficiency"])
        for row in accepted_rows
    )
    risks = Counter(
        str(flag)
        for label in labels.values()
        for flag in label.get("risk_flags", [])
    )
    selection_lengths = [
        len(labels[str(row["metadata"]["sample_id"])]["selected_evidence_ids"])
        for row in accepted_rows
    ]
    input_hashes = {
        split: sha256_file(path) for split, path in sorted(input_paths.items())
    }
    accepted = len(accepted_rows)
    return {
        "schema_version": "longcoqa-evidence-sft-quality-v1",
        "status": "AWAITING_HUMAN_AUDIT",
        "semantic_support_verified": False,
        "human_review": {
            "completed": False,
            "required_rows": min(250, accepted),
            "thresholds": {
                "supported_fact_ratio": 0.95,
                "evidence_selection_accuracy": 0.95,
                "sufficiency_agreement": 0.90,
                "non_factoid_must_independently_pass": True,
            },
        },
        "automatic_checks": {
            "test_rows_read": 0,
            "allowed_splits_only": all(
                str(row["metadata"]["split"]) in {"train", "dev"} for row in rows
            ),
            "aligned_variant_ids": data_report.get("accepted") == accepted,
            "static_schema_validated": True,
        },
        "counts": {
            "source": len(rows),
            "accepted": accepted,
            "rejected": len(rows) - accepted,
            "by_split": dict(sorted(by_split.items())),
            "by_question_type": dict(sorted(by_type.items())),
            "by_raft_mode": dict(sorted(by_mode.items())),
            "by_sufficiency": dict(sorted(by_sufficiency.items())),
            "risk_flags": dict(sorted(risks.items())),
        },
        "evidence_selection": {
            "mean_selected": (
                sum(selection_lengths) / len(selection_lengths) if selection_lengths else 0.0
            ),
            "max_selected": max(selection_lengths, default=0),
            "refusal_rate": (
                by_sufficiency.get("insufficient", 0) / accepted if accepted else 0.0
            ),
        },
        "input_checksums": input_hashes,
        "data": data_report,
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def build_human_audit_packet(
    rows: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    *,
    output_dir: Path,
    count: int,
    seed: int,
) -> dict[str, Any]:
    source_by_id = {
        str(row["metadata"]["sample_id"]): row
        for row in rows
        if str(row["metadata"]["sample_id"]) in labels
    }
    candidates = [
        {
            "sample_id": sample_id,
            "split": row["metadata"]["split"],
            "question_type": row["metadata"]["question_type"],
            "dialogue_id": row["metadata"]["dialogue_id"],
            "turn": row["metadata"]["turn"],
            "raft_mode": row["metadata"].get("raft_mode", ""),
            "sufficiency": labels[sample_id]["sufficiency"],
            "risk_flags": labels[sample_id]["risk_flags"],
        }
        for sample_id, row in source_by_id.items()
    ]
    selected = select_human_audit_rows(candidates, count=count, seed=seed)
    packet: list[dict[str, Any]] = []
    for selected_row in selected:
        sample_id = str(selected_row["sample_id"])
        source = source_by_id[sample_id]
        label = labels[sample_id]
        packet.append(
            {
                **selected_row,
                "instruction": source["instruction"],
                "raw_answer": source["output"],
                "selected_evidence_ids": label["selected_evidence_ids"],
                "support_points": label["support_points"],
                "clean_final_answer": label["final_answer"],
                "reviewer_sufficiency": "",
                "reviewer_evidence_correct": "",
                "reviewer_answer_supported": "",
                "reviewer_notes": "",
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "human_audit_250.jsonl"
    csv_path = output_dir / "human_audit_250.csv"
    atomic_write_jsonl(jsonl_path, packet)
    if packet:
        fd, temporary = tempfile.mkstemp(prefix=f".{csv_path.name}.", dir=output_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(packet[0]))
                writer.writeheader()
                for record in packet:
                    writer.writerow(
                        {
                            key: (
                                json.dumps(value, ensure_ascii=False)
                                if isinstance(value, (list, dict))
                                else value
                            )
                            for key, value in record.items()
                        }
                    )
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o644)
            os.replace(temporary, csv_path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
    else:
        _atomic_write_text(csv_path, "")

    instructions = """# Evidence-aware SFT 人工审核

当前状态：`AWAITING_HUMAN_AUDIT`。该审核包由模型标注生成，尚不能进入训练。

请逐行填写：

- `reviewer_sufficiency`: sufficient / partial / insufficient；
- `reviewer_evidence_correct`: yes / no，所选证据是否真的支持列出的支持点；
- `reviewer_answer_supported`: yes / no，最终答案的每个事实是否受证据支持；
- `reviewer_notes`: 错误类型、建议修改或其他备注。

训练门槛：受支持事实比例与证据选择正确率均不低于 95%，充分性一致率不低于 90%，且 non-factoid 必须单独达到同样门槛。完成后保留 sample_id，不要改变行数或顺序。
"""
    _atomic_write_text(output_dir / "HUMAN_AUDIT.md", instructions)
    return {
        "rows": len(packet),
        "requested": count,
        "seed": seed,
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
        "sample_ids_hash": stable_json_hash([row["sample_id"] for row in packet]),
    }
