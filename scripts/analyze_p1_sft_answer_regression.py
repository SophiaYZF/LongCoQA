# ruff: noqa: E501 -- report narrative strings are intentionally kept intact.

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "reports/generated/analysis/p1_sft_answer_regression"

REPRESENTATIVE_EXAMPLES = {
    "non_factoid:124:10": (
        "证据不足变为自信编造",
        "Base 正确指出当前证据没有覆盖误会澄清后的长期变化；SFT 编造其此前宽容、后来更敌视。",
    ),
    "non_factoid:134:5": (
        "完全脱离问题与证据",
        "SFT 编造哈利成为狼人、莉莉参与等不存在情节，是明显生成退化。",
    ),
    "non_factoid:136:3": (
        "忽略直接证据",
        "证据明确写卡多根自告奋勇并频繁换口令；SFT 改写成惊讶、不满和期待见哈利。",
    ),
    "non_factoid:137:2": (
        "病态重复",
        "SFT 输出长串重复的“校长”，属于解码或适配器稳定性故障。",
    ),
    "non_factoid:125:8": (
        "表面关键词替代答案",
        "问题问守护神形态，SFT 把证据中的咒语“呼神护卫”当成形态；Character-F1 反而大幅升高。",
    ),
    "factoid:22:9": (
        "关系抽取错误",
        "证据直接说明纳西莎是德拉科母亲，SFT 回答夫妻关系。",
    ),
    "factoid:22:1": (
        "过度压缩丢失核心身份",
        "SFT 只答“曾曾祖父”，遗漏问题真正需要的霍格沃茨校长身份。",
    ),
    "factoid:31:5": (
        "多轮指代后答错槽位",
        "问题询问死亡日期，SFT 却输出人物姓名，表现为回答了错误语义槽位。",
    ),
    "factoid:28:5": (
        "Judge 可能偏严",
        "SFT 回答“克劳奇”方向正确但不够精确；从 5 降到 2 可能夸大了实际质量差距。",
    ),
}

TRAINING_MISMATCHES = [
    {
        "sample_id": "non_factoid:130:13",
        "issue": "Target 说湖边保护小天狼星，但 supplied gold evidence 只覆盖私人课上学习咒语；真正支持 target 的湖边段落只在 RAFT hard negative 中出现。",
        "risk": "Grounded SFT 学到可以脱离 gold evidence 回答；RAFT 还可能学到使用 distractor、却引用 gold ID。",
    },
    {
        "sample_id": "non_factoid:102:13",
        "issue": "Evidence 只覆盖保护符咒失效与转移住所；target 增加魔法技能、盟友、邓布利多指导和寻找魂器，并把这些内容引用给不支持它们的 evidence。",
        "risk": "训练目标系统性奖励 unsupported synthesis 和错误 citation attribution。",
    },
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_results(model: str) -> tuple[dict[str, dict[str, Any]], Path]:
    path = PROJECT_ROOT / f"outputs/formal/p1/merged/{model}/grounded/full/results.jsonl"
    rows = read_jsonl(path)
    return {str(row["sample_id"]): row for row in rows}, path


def load_judgments(model: str) -> tuple[dict[str, dict[str, Any]], Path]:
    path = PROJECT_ROOT / f"outputs/judgments/qwen37/p1/{model}/grounded/full/judgments.jsonl"
    rows = {
        str(row["sample_id"]): row["judgment"]
        for row in read_jsonl(path)
        if row["judgment_type"] == "answer_quality"
    }
    return rows, path


def question_from_result(row: dict[str, Any]) -> str:
    content = row["messages"][-1]["content"]
    return content.rsplit("[QUESTION]", 1)[-1].strip()


def pathological_repetition(text: str) -> bool:
    repeated_phrase = re.search(r"(.{2,12})(?:[、，,\s]*\1){5,}", text)
    low_diversity = len(text) > 300 and len(set(text)) / len(text) < 0.12
    return bool(repeated_phrase) or low_diversity


def excerpt(text: str, length: int = 220) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= length else compact[: length - 1] + "…"


def build_paired_frame() -> tuple[pd.DataFrame, dict[str, str]]:
    base, base_path = load_results("base")
    grounded, grounded_path = load_results("grounded")
    base_judge, base_judge_path = load_judgments("base")
    grounded_judge, grounded_judge_path = load_judgments("grounded")

    sample_ids = sorted(set(base) & set(grounded) & set(base_judge) & set(grounded_judge))
    rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        before = base[sample_id]
        after = grounded[sample_id]
        before_judge = base_judge[sample_id]
        after_judge = grounded_judge[sample_id]
        before_metrics = before.get("metrics") or {}
        after_metrics = after.get("metrics") or {}
        before_answer = str(before["answer_text"]).strip()
        after_answer = str(after["answer_text"]).strip()
        before_score = int(before_judge["correctness"])
        after_score = int(after_judge["correctness"])
        rows.append(
            {
                "sample_id": sample_id,
                "question_type": before["question_type"],
                "question": question_from_result(before),
                "reference": before["reference"],
                "base_answer": before_answer,
                "sft_answer": after_answer,
                "base_judge_correctness": before_score,
                "sft_judge_correctness": after_score,
                "judge_delta_1_to_5": after_score - before_score,
                "judge_delta_100": 25 * (after_score - before_score),
                "base_character_f1": before_metrics.get("character_f1"),
                "sft_character_f1": after_metrics.get("character_f1"),
                "character_f1_delta": (
                    after_metrics.get("character_f1", 0) - before_metrics.get("character_f1", 0)
                ),
                "base_answer_chars": len(before_answer),
                "sft_answer_chars": len(after_answer),
                "base_refusal": "证据不足" in before_answer,
                "sft_refusal": "证据不足" in after_answer,
                "base_factual_conflict": bool(before_judge.get("has_factual_conflict")),
                "sft_factual_conflict": bool(after_judge.get("has_factual_conflict")),
                "base_unsupported_claim": bool(before_judge.get("has_unsupported_claim")),
                "sft_unsupported_claim": bool(after_judge.get("has_unsupported_claim")),
                "base_missing_key_points": bool(before_judge.get("missing_key_points")),
                "sft_missing_key_points": bool(after_judge.get("missing_key_points")),
                "sft_pathological_repetition": pathological_repetition(after_answer),
                "base_judge_reason": before_judge.get("error_summary") or "",
                "sft_judge_reason": after_judge.get("error_summary") or "",
            }
        )

    sources = {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in (base_path, grounded_path, base_judge_path, grounded_judge_path)
    }
    return pd.DataFrame(rows), sources


def rate(value: pd.Series) -> float:
    return float(value.astype(bool).mean())


def summarize(paired: pd.DataFrame) -> dict[str, Any]:
    regressed = paired[paired["judge_delta_1_to_5"] < 0]
    worst = paired[paired["judge_delta_1_to_5"] == -4]
    return {
        "paired_samples": len(paired),
        "regressed_samples": len(regressed),
        "unchanged_samples": int((paired["judge_delta_1_to_5"] == 0).sum()),
        "improved_samples": int((paired["judge_delta_1_to_5"] > 0).sum()),
        "mean_judge_delta_1_to_5": float(paired["judge_delta_1_to_5"].mean()),
        "mean_judge_delta_100": float(paired["judge_delta_100"].mean()),
        "mean_character_f1_delta": float(paired["character_f1_delta"].mean()),
        "f1_improved_among_regressions": rate(regressed["character_f1_delta"] > 0),
        "worst_four_level_regressions": len(worst),
        "sft_error_rates_among_regressions": {
            "factual_conflict": rate(regressed["sft_factual_conflict"]),
            "unsupported_claim": rate(regressed["sft_unsupported_claim"]),
            "missing_key_points": rate(regressed["sft_missing_key_points"]),
            "pathological_repetition": rate(regressed["sft_pathological_repetition"]),
        },
    }


def build_error_profile(paired: pd.DataFrame) -> pd.DataFrame:
    regressed = paired[paired["judge_delta_1_to_5"] < 0]
    rows = []
    for label, base_col, sft_col in (
        ("事实冲突", "base_factual_conflict", "sft_factual_conflict"),
        ("无证据支持的主张", "base_unsupported_claim", "sft_unsupported_claim"),
        ("缺少关键点", "base_missing_key_points", "sft_missing_key_points"),
        ("明确拒答", "base_refusal", "sft_refusal"),
    ):
        rows.extend(
            [
                {
                    "error_type": label,
                    "model": "SFT 前：Base",
                    "rate_pct": 100 * rate(regressed[base_col]),
                    "samples": len(regressed),
                },
                {
                    "error_type": label,
                    "model": "SFT 后：Grounded SFT",
                    "rate_pct": 100 * rate(regressed[sft_col]),
                    "samples": len(regressed),
                },
            ]
        )
    return pd.DataFrame(rows)


def build_question_type_table(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for question_type, group in paired.groupby("question_type", sort=True):
        rows.append(
            {
                "question_type": question_type,
                "samples": len(group),
                "regressed": int((group["judge_delta_1_to_5"] < 0).sum()),
                "unchanged": int((group["judge_delta_1_to_5"] == 0).sum()),
                "improved": int((group["judge_delta_1_to_5"] > 0).sum()),
                "mean_judge_delta_1_to_5": float(group["judge_delta_1_to_5"].mean()),
                "mean_judge_delta_100": float(group["judge_delta_100"].mean()),
                "median_base_answer_chars": float(group["base_answer_chars"].median()),
                "median_sft_answer_chars": float(group["sft_answer_chars"].median()),
            }
        )
    return pd.DataFrame(rows)


def build_refusal_table(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base_refusals = paired[paired["base_refusal"]]
    for question_type, group in base_refusals.groupby("question_type", sort=True):
        rows.append(
            {
                "question_type": question_type,
                "samples_where_base_refused": len(group),
                "base_mean_judge_1_to_5": float(group["base_judge_correctness"].mean()),
                "sft_mean_judge_1_to_5": float(group["sft_judge_correctness"].mean()),
                "sft_lower": int((group["judge_delta_1_to_5"] < 0).sum()),
                "same": int((group["judge_delta_1_to_5"] == 0).sum()),
                "sft_higher": int((group["judge_delta_1_to_5"] > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def build_training_profile() -> pd.DataFrame:
    rows = []
    for variant in ("conversation", "grounded", "raft"):
        path = PROJECT_ROOT / f"artifacts/training/{variant}_train.jsonl"
        data = read_jsonl(path)
        for question_type in ("factoid", "non_factoid"):
            subset = [row for row in data if row["metadata"]["question_type"] == question_type]
            lengths = [len(row["output"].split("\n\n引用：", 1)[0].strip()) for row in subset]
            rows.append(
                {
                    "variant": variant,
                    "question_type": question_type,
                    "samples": len(subset),
                    "median_target_chars": float(pd.Series(lengths).median()),
                    "targets_at_most_10_chars_pct": 100
                    * sum(value <= 10 for value in lengths)
                    / len(lengths),
                    "explicit_refusal_targets": sum("证据不足" in row["output"] for row in subset),
                }
            )
    return pd.DataFrame(rows)


def build_examples(paired: pd.DataFrame) -> pd.DataFrame:
    indexed = paired.set_index("sample_id")
    rows = []
    for sample_id, (category, diagnosis) in REPRESENTATIVE_EXAMPLES.items():
        row = indexed.loc[sample_id]
        rows.append(
            {
                "sample_id": sample_id,
                "question_type": row["question_type"],
                "category": category,
                "judge_change": f"{int(row['base_judge_correctness'])}→{int(row['sft_judge_correctness'])}",
                "character_f1_change": f"{row['base_character_f1']:.3f}→{row['sft_character_f1']:.3f}",
                "question": excerpt(str(row["question"]), 140),
                "base_answer": excerpt(str(row["base_answer"])),
                "sft_answer": excerpt(str(row["sft_answer"])),
                "diagnosis": diagnosis,
            }
        )
    return pd.DataFrame(rows)


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def build_artifact(
    summary: dict[str, Any],
    error_profile: pd.DataFrame,
    question_types: pd.DataFrame,
    refusals: pd.DataFrame,
    training_profile: pd.DataFrame,
    examples: pd.DataFrame,
    mismatches: pd.DataFrame,
) -> dict[str, Any]:
    source_defs = [
        {
            "id": "regression_summary",
            "label": "Base 与 Grounded SFT 配对答案审计",
            "path": "reports/generated/analysis/p1_sft_answer_regression/summary.json",
            "description": "539 条 Grounded-eval 配对答案、Character-F1、qwen3.7-max correctness 与错误标记。",
        },
        {
            "id": "error_profile_source",
            "label": "退化样本的错误标记率",
            "path": "reports/generated/analysis/p1_sft_answer_regression/error_profile.csv",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "description": "Compare Base and Grounded SFT Judge error-flag rates on the 307 samples whose correctness score declined.",
                "executed_at": "2026-07-23T00:00:00Z",
                "sql": "SELECT * FROM read_csv_auto('reports/generated/analysis/p1_sft_answer_regression/error_profile.csv')",
            },
        },
        {
            "id": "question_type_source",
            "label": "按题型拆分的 SFT 前后退化",
            "path": "reports/generated/analysis/p1_sft_answer_regression/question_type_regression.csv",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "description": "Aggregate paired correctness changes and answer lengths by question type.",
                "executed_at": "2026-07-23T00:00:00Z",
                "sql": "SELECT * FROM read_csv_auto('reports/generated/analysis/p1_sft_answer_regression/question_type_regression.csv')",
            },
        },
        {
            "id": "refusal_source",
            "label": "Base 拒答后的 SFT 分数转移",
            "path": "reports/generated/analysis/p1_sft_answer_regression/base_refusal_transitions.csv",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "description": "Compare paired scores where Base explicitly said the supplied evidence was insufficient.",
                "executed_at": "2026-07-23T00:00:00Z",
                "sql": "SELECT * FROM read_csv_auto('reports/generated/analysis/p1_sft_answer_regression/base_refusal_transitions.csv')",
            },
        },
        {
            "id": "training_profile_source",
            "label": "SFT target 长度与拒答监督",
            "path": "reports/generated/analysis/p1_sft_answer_regression/training_target_profile.csv",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "description": "Profile target length and explicit refusal supervision in the three released training variants.",
                "executed_at": "2026-07-23T00:00:00Z",
                "sql": "SELECT * FROM read_csv_auto('reports/generated/analysis/p1_sft_answer_regression/training_target_profile.csv')",
            },
        },
        {
            "id": "example_source",
            "label": "人工阅读的代表性 Base/SFT 答案对",
            "path": "reports/generated/analysis/p1_sft_answer_regression/representative_examples.csv",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "description": "Select nine deterministic examples spanning verified failure modes and one Judge-severity caveat.",
                "executed_at": "2026-07-23T00:00:00Z",
                "sql": "SELECT * FROM read_csv_auto('reports/generated/analysis/p1_sft_answer_regression/representative_examples.csv')",
            },
        },
        {
            "id": "training_mismatch_source",
            "label": "人工核对的训练 target-evidence 不一致案例",
            "path": "reports/generated/analysis/p1_sft_answer_regression/training_mismatch_examples.csv",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "description": "Select two manually verified Grounded/RAFT training examples whose targets are not supported by the cited gold evidence.",
                "executed_at": "2026-07-23T00:00:00Z",
                "sql": "SELECT * FROM read_csv_auto('reports/generated/analysis/p1_sft_answer_regression/training_mismatch_examples.csv')",
            },
        },
    ]

    blocks = [
        {
            "id": "title",
            "type": "markdown",
            "body": "# LongCoQA P1：Grounded SFT 答案退化诊断\n\n2026-07-23 数据快照",
        },
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": "regression_summary",
            "body": (
                "## 技术摘要\n\n"
                "**分数下降主要对应真实的回答策略退化，不只是 Judge-100 缩放。** 在 539 条相同 Grounded-eval 问题上，307 条（57.0%）SFT 后 correctness 下降，只有 90 条提升；平均下降 1.08/5。退化样本中 77.5% 被标记为事实冲突、83.1% 含无证据支持的主张、97.4% 缺关键点。\n\n"
                "**最强机制是“更短且永不拒答”的策略迁移。** Grounded SFT 生成答案中位长度从 Base 的 125 字降到 34 字，并从 Base 的 215 次 evidence-insufficient 拒答变为 0 次拒答。该变化对 factoid 有时有益，但在 non-factoid 上大量转化为自信编造。\n\n"
                "**训练数据质量是主要解释，但训练配方和单一 Judge 仍是混淆项。** 人工阅读确认 Grounded targets 中存在答案超出 evidence、citation 错配，以及多轮短答案丢失语义槽位的实例；另有约 2% 的病态重复，提示适配器稳定性也有问题。"
            ),
        },
        {
            "id": "error_finding",
            "type": "markdown",
            "sourceId": "error_profile_source",
            "body": "## 退化不是单纯的措辞偏好：事实冲突和 unsupported claims 同时激增\n\n在 Judge correctness 下降的 307 条配对中，SFT 答案的三个语义风险标记都远高于 Base。63.5% 的这些样本 Character-F1 反而上升，说明字符串指标正在奖励与 reference 的局部词语重合，而没有保护事实正确性。",
        },
        {"id": "error_chart_block", "type": "chart", "chartId": "error_profile_chart"},
        {
            "id": "type_finding",
            "type": "markdown",
            "sourceId": "question_type_source",
            "body": "## 问题集中在 non-factoid；factoid 更多表现为过度压缩和槽位错误\n\nFactoid 的平均 correctness 只下降 0.24/5；non-factoid 下降 1.68/5，且 314 条中有 227 条退化。训练 target 的长度分布几乎原样迁移到生成：factoid 训练 target 中位数只有 6 字，SFT 测试输出中位数 7 字，容易把“谁、为何、日期、关系”等不同槽位压成邻近实体。",
        },
        {"id": "type_table_block", "type": "table", "tableId": "question_type_table"},
        {
            "id": "refusal_finding",
            "type": "markdown",
            "sourceId": "refusal_source",
            "body": "## “永不拒答”在 factoid 上有时修复 Base 的过度保守，却在 non-factoid 上制造幻觉\n\nBase 拒答的 79 条 factoid 中，SFT 有 44 条提升、13 条下降；但 Base 拒答的 136 条 non-factoid 中，SFT 有 94 条下降、仅 16 条提升。这说明拒答策略不能统一处理：短事实题可适度利用历史或常识补全，开放题必须要求 evidence 覆盖问题的全部方面。",
        },
        {"id": "refusal_table_block", "type": "table", "tableId": "refusal_table"},
        {
            "id": "training_finding",
            "type": "markdown",
            "sourceId": "training_profile_source",
            "body": "## Grounded supervision 没有教模型在证据不足时停下来\n\n1,935 条 Grounded train targets 中明确拒答为 0；只有 RAFT 的 208 条 missing-gold 样本提供拒答监督。与此同时，Grounded factoid targets 的 87.5% 不超过 10 个字符。这个标签分布直接解释了 SFT 后的两个行为：不拒答，以及极短回答。",
        },
        {"id": "training_profile_table_block", "type": "table", "tableId": "training_profile_table"},
        {
            "id": "mismatch_finding",
            "type": "markdown",
            "sourceId": "training_mismatch_source",
            "body": "## 人工核对发现 target 与 cited evidence 的实质不一致\n\n下面两例不是措辞差异，而是训练 target 包含 supplied gold evidence 无法推出的情节。尤其 non_factoid:130:13 的正确支撑段落出现在 RAFT hard negative 中，target 却引用 gold IDs；这会直接训练模型从 distractor 抽取答案并给出错误引用。",
        },
        {"id": "mismatch_table_block", "type": "table", "tableId": "mismatch_table"},
        {
            "id": "examples_finding",
            "type": "markdown",
            "sourceId": "example_source",
            "body": "## 逐题阅读显示五种主要失败模式\n\n代表性答案对覆盖：证据不足后自信编造、忽略直接证据、关键词替代答案、关系/槽位错误、过度压缩，以及病态重复。也保留了一条 Judge 可能过严的反例，避免把所有下降机械解释为同等严重。",
        },
        {"id": "examples_table_block", "type": "table", "tableId": "examples_table"},
        {
            "id": "limitations",
            "type": "markdown",
            "body": "## 结论边界\n\n这是诊断性而非严格因果分析。qwen3.7-max 是单一 Judge，部分 5→4 或 5→2 的下降可能把‘略不完整’放大；人工阅读也只覆盖确定性选出的代表案例。另一方面，112 条 5→1 的最严重下降、事实冲突标记和直接文本对照足以排除‘全部只是 Judge 尺度问题’。当前结果不能区分数据标签问题与 rank-64、all-module、3 epochs、较高 LR 的训练配方各自贡献。",
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": "## 建议的最小验证\n\n1. 对 200 条 train 样本做 evidence-support 双人标注，factoid/non-factoid 各半；把 target 拆成 supported、partially supported、unsupported。\n2. 先修复 target-evidence/citation 不一致，再训练一个 1 epoch、低 LR 的小规模 adapter，与原配方做 2×2 对照。\n3. 将 abstention target 加入 Grounded SFT，并按题型设规则：factoid 允许精确短答，non-factoid 只有 evidence 覆盖全部子问题时才作答。\n4. 用人工盲评或第二个不同家族 Judge 复核 112 条 5→1 和 32 条无 conflict/unsupported 标记的下降。",
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": "## 尚待回答的问题\n\n清洗后 SFT 是否仍会出现关系槽位错误？病态重复来自 adapter 训练还是解码参数？non-factoid 的 reference 是否需要改写为逐条 evidence-supported claims，而不是一段综合叙述？这些问题会决定数据集更适合作为训练集、评测集，还是只适合作为检索任务来源。",
        },
    ]

    charts = [
        {
            "id": "error_profile_chart",
            "title": "Judge 退化样本中的错误标记率",
            "subtitle": "307 条 Grounded SFT correctness 低于 Base 的配对；同一样本可同时命中多个标记。",
            "type": "bar",
            "dataset": "error_profile",
            "sourceId": "error_profile_source",
            "valueFormat": "number",
            "encodings": {
                "x": {"field": "error_type", "type": "nominal", "label": "行为或错误标记"},
                "y": {"field": "rate_pct", "type": "quantitative", "label": "样本占比 (%)"},
                "color": {"field": "model", "type": "nominal", "label": "模型阶段"},
                "tooltip": [
                    {"field": "rate_pct", "type": "quantitative", "label": "占比 (%)"},
                    {"field": "samples", "type": "quantitative", "label": "配对样本数"},
                ],
            },
        }
    ]

    tables = [
        {
            "id": "question_type_table",
            "title": "按题型拆分的 correctness 与答案长度变化",
            "subtitle": "Grounded eval，同一 539 条样本中的 Base 与 Grounded SFT 配对。",
            "dataset": "question_types",
            "sourceId": "question_type_source",
            "defaultSort": {"field": "samples", "direction": "desc"},
            "columns": [
                {"field": "question_type", "label": "题型", "type": "text"},
                {"field": "samples", "label": "N", "type": "number"},
                {"field": "regressed", "label": "下降", "type": "number"},
                {"field": "unchanged", "label": "不变", "type": "number"},
                {"field": "improved", "label": "提升", "type": "number"},
                {
                    "field": "mean_judge_delta_1_to_5",
                    "label": "平均 correctness Δ",
                    "type": "number",
                    "semantic": "movement",
                },
                {"field": "median_base_answer_chars", "label": "Base 中位字数", "type": "number"},
                {"field": "median_sft_answer_chars", "label": "SFT 中位字数", "type": "number"},
            ],
        },
        {
            "id": "refusal_table",
            "title": "Base 明确拒答后的 SFT 分数转移",
            "subtitle": "仅统计 Base answer 含“证据不足”的样本。",
            "dataset": "refusals",
            "sourceId": "refusal_source",
            "defaultSort": {"field": "samples_where_base_refused", "direction": "desc"},
            "columns": [
                {"field": "question_type", "label": "题型", "type": "text"},
                {"field": "samples_where_base_refused", "label": "N", "type": "number"},
                {"field": "base_mean_judge_1_to_5", "label": "Base 均分", "type": "number"},
                {"field": "sft_mean_judge_1_to_5", "label": "SFT 均分", "type": "number"},
                {"field": "sft_lower", "label": "SFT 更低", "type": "number"},
                {"field": "same", "label": "相同", "type": "number"},
                {"field": "sft_higher", "label": "SFT 更高", "type": "number"},
            ],
        },
        {
            "id": "training_profile_table",
            "title": "训练 targets 的长度与明确拒答监督",
            "subtitle": "P1 train；长度剥离 citation footer 后按中文字符计。",
            "dataset": "training_profile",
            "sourceId": "training_profile_source",
            "defaultSort": {"field": "variant", "direction": "asc"},
            "columns": [
                {"field": "variant", "label": "训练变体", "type": "text"},
                {"field": "question_type", "label": "题型", "type": "text"},
                {"field": "samples", "label": "N", "type": "number"},
                {"field": "median_target_chars", "label": "Target 中位字数", "type": "number"},
                {"field": "targets_at_most_10_chars_pct", "label": "≤10 字占比 (%)", "type": "number"},
                {"field": "explicit_refusal_targets", "label": "明确拒答 targets", "type": "number"},
            ],
        },
        {
            "id": "mismatch_table",
            "title": "已人工确认的 target-evidence 不一致",
            "subtitle": "用于证明训练构造风险存在，不代表总体发生率。",
            "dataset": "mismatches",
            "sourceId": "training_mismatch_source",
            "defaultSort": {"field": "sample_id", "direction": "asc"},
            "columns": [
                {"field": "sample_id", "label": "样本", "type": "text"},
                {"field": "issue", "label": "观察到的问题", "type": "text"},
                {"field": "risk", "label": "训练风险", "type": "text"},
            ],
        },
        {
            "id": "examples_table",
            "title": "代表性 Base/SFT 答案对",
            "subtitle": "确定性选取 9 条，覆盖主要失败模式及一条 Judge 严格度反例；完整文本保存在 supporting CSV。",
            "dataset": "examples",
            "sourceId": "example_source",
            "defaultSort": {"field": "sample_id", "direction": "asc"},
            "columns": [
                {"field": "sample_id", "label": "样本", "type": "text"},
                {"field": "category", "label": "模式", "type": "text"},
                {"field": "judge_change", "label": "Judge", "type": "text"},
                {"field": "character_f1_change", "label": "Character-F1", "type": "text"},
                {"field": "question", "label": "问题", "type": "text"},
                {"field": "base_answer", "label": "Base", "type": "text"},
                {"field": "sft_answer", "label": "Grounded SFT", "type": "text"},
                {"field": "diagnosis", "label": "诊断", "type": "text"},
            ],
        },
    ]

    datasets = {
        "error_profile": records(error_profile),
        "question_types": records(question_types),
        "refusals": records(refusals),
        "training_profile": records(training_profile),
        "examples": records(examples),
        "mismatches": records(mismatches),
    }
    manifest_sources = [
        {key: value for key, value in source.items() if key != "description"} for source in source_defs
    ]
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "LongCoQA P1：Grounded SFT 答案退化诊断",
            "description": "逐题比较 Base 与 Grounded SFT 的生成答案，诊断 Judge correctness 下降的主要机制。",
            "generatedAt": "2026-07-23T00:00:00Z",
            "filters": [],
            "blocks": blocks,
            "cards": [],
            "charts": charts,
            "tables": tables,
            "sources": manifest_sources,
        },
        "snapshot": {
            "version": 1,
            "status": "ready",
            "generatedAt": "2026-07-23T00:00:00Z",
            "datasets": datasets,
            "accessIssues": [],
        },
        "sources": source_defs,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paired, source_hashes = build_paired_frame()
    if len(paired) != 539 or paired["sample_id"].nunique() != 539:
        raise ValueError("Expected 539 unique paired Grounded-eval samples")

    summary = summarize(paired)
    error_profile = build_error_profile(paired)
    question_types = build_question_type_table(paired)
    refusals = build_refusal_table(paired)
    training_profile = build_training_profile()
    examples = build_examples(paired)
    mismatches = pd.DataFrame(TRAINING_MISMATCHES)

    paired.to_csv(OUTPUT_DIR / "paired_answers.csv", index=False)
    error_profile.to_csv(OUTPUT_DIR / "error_profile.csv", index=False)
    question_types.to_csv(OUTPUT_DIR / "question_type_regression.csv", index=False)
    refusals.to_csv(OUTPUT_DIR / "base_refusal_transitions.csv", index=False)
    training_profile.to_csv(OUTPUT_DIR / "training_target_profile.csv", index=False)
    examples.to_csv(OUTPUT_DIR / "representative_examples.csv", index=False)
    mismatches.to_csv(OUTPUT_DIR / "training_mismatch_examples.csv", index=False)

    summary["source_hashes"] = source_hashes
    summary["training_file_hashes"] = {
        f"artifacts/training/{variant}_train.jsonl": sha256_file(
            PROJECT_ROOT / f"artifacts/training/{variant}_train.jsonl"
        )
        for variant in ("conversation", "grounded", "raft")
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    artifact = build_artifact(
        summary,
        error_profile,
        question_types,
        refusals,
        training_profile,
        examples,
        mismatches,
    )
    (OUTPUT_DIR / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
