from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer

from longcoqa.config import load_provider_settings
from longcoqa.data import LongCoQADataset
from longcoqa.data.alignment import write_evidence_alignment
from longcoqa.data.corpus import write_corpus
from longcoqa.data.manifest import write_manifests
from longcoqa.evaluation.api_smoke import (
    run_api_smoke as run_guarded_api_smoke,
)
from longcoqa.evaluation.api_smoke import (
    select_stratified_smoke12,
    validate_api_smoke_request,
)
from longcoqa.evaluation.judge_calibration import (
    calibrate_judges,
    export_human_annotation,
    import_human_annotations,
    load_judge_profiles,
    make_judge_evaluator,
)
from longcoqa.evaluation.judge_runner import run_llm_judge
from longcoqa.evaluation.p0_claim_repair_migration import (
    execute_claim_recall_v2_migration,
    plan_claim_recall_v2_migration,
)
from longcoqa.evaluation.p0_claim_repair_v3_migration import (
    execute_claim_recall_v3_migration,
    plan_claim_recall_v3_migration,
)
from longcoqa.evaluation.p0_nonfactoid_formal import (
    CLAIM_RECALL_MAX_REPAIR_ATTEMPTS,
    CLAIM_RECALL_REPAIR_MAX_TOKENS,
    EVALUATOR_MODEL,
    GENERATION_CONFIG_VERSION,
    JUDGE_100_PARSER_VERSION,
    JUDGE_100_PROMPT_VERSION,
    build_claim_recall_evaluator_identity,
    get_generation_options,
    load_p0_nonfactoid_config,
    make_claim_recall_evaluator,
    make_judge_100_evaluator,
    normalize_nonfactoid_row,
    run_p0_nonfactoid_task,
    update_main_table_2,
    validate_p0_nonfactoid_inputs,
    write_p0_nonfactoid_matrix_summary,
)
from longcoqa.evaluation.p0_nonfactoid_formal import (
    TASKS as P0_NONFACTOID_TASKS,
)
from longcoqa.evaluation.p0_rag_formal import (
    load_evidence_parent_map,
    load_p0_rag_config,
    make_answer_quality_evaluator,
    make_formal_rag_quality_evaluator,
    normalize_formal_row,
    run_formal_task,
    validate_p0_rag_inputs,
    write_matrix_summary,
)
from longcoqa.evaluation.p1_cpt_reporting import evaluate_p1_cpt
from longcoqa.evaluation.p1_judge import run_p1_judge_matrix
from longcoqa.evaluation.paper_metric_profiles import load_metric_profiles
from longcoqa.evaluation.paper_metrics import build_learned_adapter
from longcoqa.evaluation.paper_reporting import (
    _atomic_write_text,
    aggregate_paper_scores,
    export_paper_table,
    score_batch_rows,
    score_rows,
)
from longcoqa.evaluation.rag_quality import make_rag_quality_evaluator
from longcoqa.evaluation.reporting import compare_inference_results, evaluate_inference_results
from longcoqa.inference.api_runner import run_api_inference, run_api_suite
from longcoqa.inference.gpt_oss_runner import run_gpt_oss_suite
from longcoqa.inference.local_runner import run_local_inference, run_local_suite
from longcoqa.inference.merge_shards import merge_local_shards
from longcoqa.inference.p1_runner import merge_p1_shards, run_p1_suite
from longcoqa.paths import PROJECT_ROOT, resolve_project_path
from longcoqa.providers import ProviderClient
from longcoqa.providers.cache import ResponseCache
from longcoqa.rag.context import build_rag_contexts
from longcoqa.rag.runner import merge_rag_shards, run_local_rag
from longcoqa.rag.validation import (
    validate_generation_artifact,
    validate_ranking_artifact,
)
from longcoqa.retrieval.bm25 import BM25Index
from longcoqa.retrieval.contextual import (
    generate_contextual_corpus,
    run_contextual_retrieval,
)
from longcoqa.retrieval.dense import run_dense_and_hybrid
from longcoqa.retrieval.evaluate import write_retrieval_evaluation
from longcoqa.retrieval.graph_progressive import run_graph_progressive
from longcoqa.retrieval.progressive import (
    generate_progressive_plans,
    retrieve_progressive_plans,
)
from longcoqa.retrieval.reranker import run_qwen_reranker
from longcoqa.retrieval.rewrite import (
    retrieve_rewritten_queries,
    run_local_query_rewrite,
)
from longcoqa.training.data import build_cpt_dataset, build_supervised_datasets
from longcoqa.utils.hashing import stable_json_hash
from longcoqa.utils.io import atomic_write_json, atomic_write_jsonl, read_jsonl, read_yaml

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


def _print_json(value: Any) -> None:
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _load_p0_rag_validation(config: Path):
    section, cells = load_p0_rag_config(resolve_project_path(config))
    expected_samples = int(section.get("expected_samples", 3019))
    alignment_path = resolve_project_path(str(section["evidence_alignment_path"]))
    parents_path = resolve_project_path(str(section["parents_path"]))
    evidence_parent_map = load_evidence_parent_map(alignment_path, parents_path)
    report = validate_p0_rag_inputs(
        cells,
        expected_samples=expected_samples,
        evidence_parent_map=evidence_parent_map,
    )
    return section, cells, expected_samples, evidence_parent_map, report


def _load_p0_nonfactoid_validation(config: Path):
    section, cells = load_p0_nonfactoid_config(resolve_project_path(config))
    expected_source_samples = int(section["expected_source_samples"])
    expected_nonfactoid_samples = int(section["expected_nonfactoid_samples"])
    report = validate_p0_nonfactoid_inputs(
        cells,
        expected_source_samples=expected_source_samples,
        expected_nonfactoid_samples=expected_nonfactoid_samples,
    )
    return section, cells, expected_nonfactoid_samples, report


def _positive_worker_override(value: str, *, source: str) -> int:
    try:
        workers = int(value)
    except ValueError as error:
        raise typer.BadParameter(
            f"{source} must be a positive integer", param_hint="--max-workers"
        ) from error
    if workers <= 0:
        raise typer.BadParameter(
            f"{source} must be a positive integer", param_hint="--max-workers"
        )
    return workers


def _positive_rpm_override(value: str, *, source: str) -> int:
    try:
        requests_per_minute = int(value)
    except ValueError as error:
        raise typer.BadParameter(
            f"{source} must be a positive integer", param_hint=source
        ) from error
    if requests_per_minute <= 0:
        raise typer.BadParameter(
            f"{source} must be a positive integer", param_hint=source
        )
    return requests_per_minute


@app.command("validate-p0-rag-formal")
def validate_p0_rag_formal_command(
    config: Path = typer.Option(
        PROJECT_ROOT / "configs/experiments/p0_rag_formal.yaml",
        help="Formal P0/RAG matrix manifest.",
    ),
) -> None:
    """Read-only validation of all formal inputs and planned call counts."""
    *_unused, report = _load_p0_rag_validation(config)
    _print_json(report)


@app.command("evaluate-p0-rag-formal")
def evaluate_p0_rag_formal_command(
    config: Path = typer.Option(
        PROJECT_ROOT / "configs/experiments/p0_rag_formal.yaml",
        help="Formal P0/RAG matrix manifest.",
    ),
    execute: bool = typer.Option(False, "--execute", help="Required paid API execution gate."),
    cell: str | None = typer.Option(None, "--cell", help="One explicit model/protocol cell."),
    task: str | None = typer.Option(None, "--task", help="answer_quality or rag_quality."),
    all_cells: bool = typer.Option(
        False, "--all-cells", help="Explicitly authorize every declared task/cell."
    ),
    max_tokens: int = typer.Option(1024, min=64),
) -> None:
    """Run a guarded formal task or, with a separate flag, the full matrix."""
    if not execute:
        raise typer.BadParameter("Formal API evaluation requires --execute", param_hint="--execute")
    if all_cells and (cell is not None or task is not None):
        raise typer.BadParameter("--all-cells cannot be combined with --cell or --task")
    if not all_cells and (cell is None or task is None):
        raise typer.BadParameter(
            "Without --all-cells, both --cell and --task are required",
            param_hint="--cell/--task",
        )
    section, cells, expected_samples, evidence_parent_map, validation = (
        _load_p0_rag_validation(config)
    )
    by_id = {candidate.cell_id: candidate for candidate in cells}
    if all_cells:
        selected = [(candidate, name) for candidate in cells for name in candidate.tasks]
    else:
        if cell not in by_id:
            raise typer.BadParameter(f"Unknown formal cell: {cell}", param_hint="--cell")
        selected_cell = by_id[cell]
        if task not in selected_cell.tasks:
            raise typer.BadParameter(
                f"Task {task!r} is not declared for {cell}", param_hint="--task"
            )
        selected = [(selected_cell, str(task))]

    settings = load_provider_settings(str(section["provider"]))
    output_root = resolve_project_path(str(section["output_root"]))
    cache_path = resolve_project_path(str(section["cache_path"]))
    summaries: list[dict[str, Any]] = []
    with ResponseCache(cache_path) as cache, ProviderClient(settings, cache=cache) as client:
        for selected_cell, selected_task in selected:
            raw_rows = read_jsonl(selected_cell.input_path)
            rows = [
                normalize_formal_row(
                    row, selected_cell, evidence_parent_map=evidence_parent_map
                )
                for row in raw_rows
            ]
            if selected_task == "answer_quality":
                evaluator = make_answer_quality_evaluator(
                    client, max_tokens=min(max_tokens, 1024)
                )
            else:
                evaluator = make_formal_rag_quality_evaluator(
                    client,
                    citation_mode=selected_cell.citation_mode,
                    max_tokens=max_tokens,
                )
            identity = {
                "schema_version": "longcoqa-p0-rag-formal-evaluator-v1",
                "task": selected_task,
                "citation_mode": selected_cell.citation_mode,
                "max_tokens": max_tokens,
                "provider": settings.safe_summary(),
            }
            summaries.append(
                run_formal_task(
                    rows,
                    cell_id=selected_cell.cell_id,
                    task=selected_task,
                    output_dir=(
                        output_root / selected_task / selected_cell.cell_id / "full"
                    ),
                    evaluator=evaluator,
                    evaluator_identity=identity,
                    expected_samples=expected_samples,
                    max_workers=settings.concurrency,
                )
            )
    matrix = write_matrix_summary(
        cells, output_root=output_root, expected_samples=expected_samples
    )
    response = {"validation": validation, "runs": summaries, "matrix": matrix}
    _print_json(response)
    if any(summary["failed_samples"] for summary in summaries):
        raise typer.Exit(code=1)


@app.command("validate-p0-nonfactoid-main")
def validate_p0_nonfactoid_main_command(
    config: Path = typer.Option(
        PROJECT_ROOT / "configs/experiments/p0_nonfactoid_main.yaml",
        help="Locked P0 Non-factoid matrix manifest.",
    ),
) -> None:
    """Validate every locked source input without loading provider settings."""
    *_unused, report = _load_p0_nonfactoid_validation(config)
    _print_json(report)


@app.command("canary-p0-nonfactoid-claim-v3")
def canary_p0_nonfactoid_claim_v3_command(
    config: Path = typer.Option(
        PROJECT_ROOT / "configs/experiments/p0_nonfactoid_main.yaml",
        help="Locked P0 Non-factoid matrix manifest.",
    ),
    cell: str = typer.Option(..., "--cell", help="One explicit locked matrix cell."),
    sample_id: str = typer.Option(..., "--sample-id", help="One Non-factoid sample ID."),
) -> None:
    """Evaluate exactly one Claim Recall row without formal-output persistence."""
    if os.environ.get("LONGCOQA_P0_NONFACTOID_CLAIM_V3_CANARY") != "1":
        raise typer.BadParameter(
            "Canary execution requires LONGCOQA_P0_NONFACTOID_CLAIM_V3_CANARY=1",
            param_hint="LONGCOQA_P0_NONFACTOID_CLAIM_V3_CANARY",
        )

    section, cells = load_p0_nonfactoid_config(resolve_project_path(config))
    selected_cells = [candidate for candidate in cells if candidate.cell_id == cell]
    if not selected_cells:
        raise typer.BadParameter(f"Unknown P0 Non-factoid cell: {cell}", param_hint="--cell")
    if len(selected_cells) != 1:
        raise typer.BadParameter(f"Ambiguous P0 Non-factoid cell: {cell}", param_hint="--cell")
    selected_cell = selected_cells[0]

    sample_rows = [
        row
        for row in read_jsonl(selected_cell.input_path)
        if str(row.get("sample_id", "")).strip() == sample_id
    ]
    if not sample_rows:
        raise typer.BadParameter(
            f"Unknown P0 Non-factoid sample: {sample_id}", param_hint="--sample-id"
        )
    if len(sample_rows) != 1:
        raise typer.BadParameter(
            f"Ambiguous P0 Non-factoid sample: {sample_id}", param_hint="--sample-id"
        )
    row = normalize_nonfactoid_row(sample_rows[0], selected_cell)

    settings = load_provider_settings(str(section["provider"]))
    cache_path = resolve_project_path(str(section["cache_path"]))
    identity = build_claim_recall_evaluator_identity(settings.safe_summary())
    with ResponseCache(cache_path) as cache, ProviderClient(settings, cache=cache) as client:
        evaluator = make_claim_recall_evaluator(
            client,
            max_tokens=1024,
            repair_max_tokens=CLAIM_RECALL_REPAIR_MAX_TOKENS,
            max_repair_attempts=CLAIM_RECALL_MAX_REPAIR_ATTEMPTS,
        )
        evaluation = evaluator(row)

    _print_json(
        {
            "status": "ok",
            "cell": selected_cell.cell_id,
            "sample_id": row["sample_id"],
            "evaluator_hash": stable_json_hash(identity),
            "evaluation": evaluation,
            "cache_path": str(cache_path),
        }
    )


@app.command("migrate-p0-nonfactoid-claim-v2")
def migrate_p0_nonfactoid_claim_v2_command(
    config: Path = typer.Option(
        PROJECT_ROOT / "configs/experiments/p0_nonfactoid_main.yaml",
        help="Locked P0 Non-factoid matrix manifest.",
    ),
    execute: bool = typer.Option(
        False, "--execute", help="Execute the validated provider-free migration."
    ),
    backup_root: Path = typer.Option(
        ..., "--backup-root", help="Explicit root for a recoverable snapshot."
    ),
) -> None:
    """Plan or execute the provider-free Claim Recall v1-to-v2 migration."""
    gate = "LONGCOQA_P0_NONFACTOID_MIGRATE_V2"
    if execute and os.environ.get(gate) != "1":
        raise typer.BadParameter(
            f"Migration execution requires {gate}=1", param_hint=gate
        )
    plan = plan_claim_recall_v2_migration(resolve_project_path(config))
    if not execute:
        _print_json(plan.as_report())
        return
    _print_json(
        execute_claim_recall_v2_migration(
            plan, resolve_project_path(backup_root)
        )
    )


@app.command("migrate-p0-nonfactoid-claim-v3")
def migrate_p0_nonfactoid_claim_v3_command(
    config: Path = typer.Option(
        PROJECT_ROOT / "configs/experiments/p0_nonfactoid_main.yaml",
        help="Locked P0 Non-factoid matrix manifest.",
    ),
    execute: bool = typer.Option(
        False, "--execute", help="Execute the validated provider-free migration."
    ),
    backup_root: Path = typer.Option(
        ..., "--backup-root", help="Explicit root for a recoverable snapshot."
    ),
) -> None:
    """Plan or execute the provider-free Claim Recall v2-to-v3 migration."""
    gate = "LONGCOQA_P0_NONFACTOID_MIGRATE_V3"
    if execute and os.environ.get(gate) != "1":
        raise typer.BadParameter(
            f"Migration execution requires {gate}=1", param_hint=gate
        )
    plan = plan_claim_recall_v3_migration(resolve_project_path(config))
    if not execute:
        _print_json(plan.as_report())
        return
    _print_json(
        execute_claim_recall_v3_migration(
            plan, resolve_project_path(backup_root)
        )
    )


@app.command("update-p0-nonfactoid-main-table")
def update_p0_nonfactoid_main_table_command(
    document: Path = typer.Option(
        PROJECT_ROOT / "final_result.md",
        help="Main Table 2 Markdown document to update atomically.",
    ),
    matrix_root: Path = typer.Option(
        PROJECT_ROOT / "outputs/judgments/qwen37/p0_nonfactoid_main",
        help="Completed P0 Non-factoid judgment matrix.",
    ),
    config: Path = typer.Option(
        PROJECT_ROOT / "configs/experiments/p0_nonfactoid_main.yaml",
        help="Locked P0 Non-factoid matrix manifest.",
    ),
) -> None:
    """Backfill validated P0 Non-factoid metrics without constructing providers."""
    document_path = resolve_project_path(document)
    resolved_matrix_root = resolve_project_path(matrix_root)
    _section, cells = load_p0_nonfactoid_config(resolve_project_path(config))
    with document_path.open(encoding="utf-8", newline="") as handle:
        original = handle.read()
    updated = update_main_table_2(original, resolved_matrix_root, cells)
    _atomic_write_text(document_path, updated)
    _print_json(
        {
            "document": str(document_path),
            "updated_metric_cells": 28,
            "updated_rows": 14,
        }
    )


@app.command("evaluate-p0-nonfactoid-main")
def evaluate_p0_nonfactoid_main_command(
    config: Path = typer.Option(
        PROJECT_ROOT / "configs/experiments/p0_nonfactoid_main.yaml",
        help="Locked P0 Non-factoid matrix manifest.",
    ),
    execute: bool = typer.Option(False, "--execute", help="Required paid API execution gate."),
    cell: str | None = typer.Option(None, "--cell", help="One explicit locked matrix cell."),
    task: str | None = typer.Option(None, "--task", help="claim_recall or judge_100."),
    all_cells: bool = typer.Option(
        False, "--all-cells", help="Explicitly authorize every locked task-cell."
    ),
    max_workers: int | None = typer.Option(
        None, "--max-workers", help="Positive bounded sample concurrency."
    ),
) -> None:
    """Run guarded, resumable P0 Non-factoid evaluation task-cells."""
    if not execute:
        raise typer.BadParameter(
            "P0 Non-factoid API evaluation requires --execute", param_hint="--execute"
        )
    if os.environ.get("LONGCOQA_P0_NONFACTOID_EXECUTE") != "1":
        raise typer.BadParameter(
            "Paid execution requires LONGCOQA_P0_NONFACTOID_EXECUTE=1",
            param_hint="LONGCOQA_P0_NONFACTOID_EXECUTE",
        )
    if all_cells and (cell is not None or task is not None):
        raise typer.BadParameter("--all-cells cannot be combined with --cell or --task")
    if not all_cells and (cell is None or task is None):
        raise typer.BadParameter(
            "Without --all-cells, both --cell and --task are required",
            param_hint="--cell/--task",
        )
    if task is not None and task not in P0_NONFACTOID_TASKS:
        raise typer.BadParameter(
            f"Unknown P0 Non-factoid task: {task}", param_hint="--task"
        )
    if max_workers is not None and max_workers <= 0:
        raise typer.BadParameter(
            "--max-workers must be a positive integer", param_hint="--max-workers"
        )
    env_workers_raw = os.environ.get("LONGCOQA_P0_NONFACTOID_MAX_WORKERS", "").strip()
    env_workers = (
        _positive_worker_override(
            env_workers_raw, source="LONGCOQA_P0_NONFACTOID_MAX_WORKERS"
        )
        if env_workers_raw
        else None
    )
    rpm_source = "LONGCOQA_P0_NONFACTOID_REQUESTS_PER_MINUTE"
    env_rpm_raw = os.environ.get(rpm_source, "").strip()
    env_rpm = (
        _positive_rpm_override(env_rpm_raw, source=rpm_source) if env_rpm_raw else None
    )

    section, cells, expected_samples, validation = _load_p0_nonfactoid_validation(config)
    by_id = {candidate.cell_id: candidate for candidate in cells}
    if all_cells:
        selected = [
            (candidate, selected_task)
            for candidate in cells
            for selected_task in P0_NONFACTOID_TASKS
        ]
    else:
        if cell not in by_id:
            raise typer.BadParameter(
                f"Unknown P0 Non-factoid cell: {cell}", param_hint="--cell"
            )
        selected = [(by_id[str(cell)], str(task))]

    provider_name = str(section.get("provider", "")).strip()
    if provider_name != "qwen37":
        raise typer.BadParameter(
            "P0 Non-factoid evaluation requires manifest provider qwen37",
            param_hint="--config",
        )
    settings = load_provider_settings(provider_name)
    if settings.model != EVALUATOR_MODEL:
        raise typer.BadParameter(
            f"qwen37 model must be exactly {EVALUATOR_MODEL}", param_hint="--config"
        )
    if not settings.base_url or not settings.api_keys:
        raise typer.BadParameter(
            "qwen37 credentials and base URL must be configured", param_hint="--config"
        )
    workers = max_workers or env_workers or settings.concurrency
    if isinstance(workers, bool) or workers <= 0:
        raise typer.BadParameter(
            "Provider concurrency must be a positive integer", param_hint="--max-workers"
        )
    client_settings = (
        replace(settings, requests_per_minute=env_rpm) if env_rpm is not None else settings
    )

    output_root = resolve_project_path(str(section["output_root"]))
    cache_path = resolve_project_path(str(section["cache_path"]))
    summaries: list[dict[str, Any]] = []
    matrix: dict[str, Any] = {}
    with ResponseCache(cache_path) as cache, ProviderClient(
        client_settings, cache=cache
    ) as client:
        for selected_cell, selected_task in selected:
            rows = [
                normalize_nonfactoid_row(row, selected_cell)
                for row in read_jsonl(selected_cell.input_path)
                if row.get("question_type") == "non_factoid"
            ]
            if len(rows) != expected_samples:
                raise ValueError(
                    f"{selected_cell.cell_id} normalization expected {expected_samples} rows, "
                    f"found {len(rows)}"
                )
            if selected_task == "claim_recall":
                max_tokens = 1024
                evaluator = make_claim_recall_evaluator(
                    client,
                    max_tokens=max_tokens,
                    repair_max_tokens=CLAIM_RECALL_REPAIR_MAX_TOKENS,
                    max_repair_attempts=CLAIM_RECALL_MAX_REPAIR_ATTEMPTS,
                )
                identity = build_claim_recall_evaluator_identity(
                    settings.safe_summary()
                )
            else:
                max_tokens = 512
                prompt_version = JUDGE_100_PROMPT_VERSION
                parser_version = JUDGE_100_PARSER_VERSION
                evaluator = make_judge_100_evaluator(client, max_tokens=max_tokens)
                identity = {
                    "identity_schema_version": "longcoqa-p0-nonfactoid-evaluator-identity-v1",
                    "task": selected_task,
                    "model": EVALUATOR_MODEL,
                    "prompt_version": prompt_version,
                    "parser_version": parser_version,
                    "generation_config_version": GENERATION_CONFIG_VERSION,
                    "generation_options": get_generation_options(),
                    "max_tokens": max_tokens,
                    "provider": settings.safe_summary(),
                }
            summary = run_p0_nonfactoid_task(
                rows=rows,
                cell=selected_cell,
                task=selected_task,
                output_root=output_root,
                evaluator=evaluator,
                evaluator_identity=identity,
                expected_samples=expected_samples,
                max_workers=workers,
            )
            summaries.append(summary)
            matrix = write_p0_nonfactoid_matrix_summary(
                cells, output_root=output_root, expected_samples=expected_samples
            )

    _print_json({"validation": validation, "runs": summaries, "matrix": matrix})
    if any(summary["failed_samples"] for summary in summaries):
        raise typer.Exit(code=1)


@app.command("audit-data")
def audit_data(
    output: Path = typer.Option(
        PROJECT_ROOT / "reports/generated/data_audit.json",
        help="Machine-readable audit report.",
    ),
) -> None:
    """Validate the released CSVs and raw novels without modifying them."""
    dataset = LongCoQADataset()
    report = dataset.validate()
    report["generated_at"] = datetime.now(UTC).isoformat()
    atomic_write_json(output, report)
    _print_json(
        {
            "status": report["status"],
            "factoid": report["factoid"],
            "non_factoid": report["non_factoid"],
            "evidence": report["evidence"],
            "output": str(output),
        }
    )


@app.command("build-manifests")
def build_manifests(
    config: Path = typer.Option(
        PROJECT_ROOT / "configs/experiments/phase_a.yaml",
        help="Phase configuration.",
    ),
) -> None:
    """Create immutable P0 and leakage-minimized P1 manifests."""
    raw = read_yaml(config)
    manifest_config = raw["manifests"]
    search = manifest_config["split_search"]
    p0_path, p1_path = write_manifests(
        LongCoQADataset(),
        resolve_project_path(manifest_config["output_dir"]),
        seed=int(raw["seed"]),
        ratios={key: float(value) for key, value in manifest_config["p1_ratios"].items()},
        restarts=int(search["restarts"]),
        local_steps=int(search["local_steps"]),
        evidence_cut_weight=float(search["evidence_cut_weight"]),
        duplicate_question_penalty=float(search["duplicate_question_penalty"]),
    )
    p1 = json.loads(p1_path.read_text(encoding="utf-8"))
    _print_json(
        {
            "p0": str(p0_path),
            "p1": str(p1_path),
            "p1_content_hash": p1["content_hash"],
            "p1_objective": p1["objective"],
            "p1_diagnostics": p1["diagnostics"],
        }
    )


@app.command("provider-preflight")
def provider_preflight(
    provider: str = typer.Option(..., help="Provider name from configs/providers.yaml."),
    generate: bool = typer.Option(False, "--generate/--no-generate"),
    list_models: bool = typer.Option(True, "--list-models/--skip-list-models"),
    timeout_seconds: float | None = typer.Option(
        None,
        min=1.0,
        help="Optional short timeout override for health checks.",
    ),
    max_retries: int | None = typer.Option(
        None,
        min=0,
        help="Optional retry override; use zero for low-cost preflights.",
    ),
    cache_path: Path = typer.Option(PROJECT_ROOT / "runtime/api_cache.sqlite3"),
    output: Path | None = typer.Option(None, help="Safe preflight report path."),
) -> None:
    """Check provider configuration and optionally issue one tiny generation."""
    settings = load_provider_settings(provider)
    settings = replace(
        settings,
        timeout_seconds=(
            float(timeout_seconds)
            if timeout_seconds is not None
            else settings.timeout_seconds
        ),
        max_retries=max_retries if max_retries is not None else settings.max_retries,
    )
    report: dict[str, Any] = {
        "provider": settings.safe_summary(),
        "checked_at": datetime.now(UTC).isoformat(),
    }
    with ResponseCache(cache_path) as cache, ProviderClient(settings, cache=cache) as client:
        if list_models:
            try:
                models = client.list_models()
                report["list_models"] = {
                    "status": "ok",
                    "count": len(models),
                    "configured_model_visible": any(
                        model.get("id") == settings.model for model in models
                    ),
                    "models": models[:100],
                }
            except Exception as error:
                # The generation endpoint may work even when /models is disabled.
                report["list_models"] = {
                    "status": "unavailable",
                    "error_type": type(error).__name__,
                    "error": str(error)[:500],
                }
        else:
            report["list_models"] = {"status": "skipped"}

        if generate:
            try:
                result = client.generate(
                    [{"role": "user", "content": "只回复：LONGCOQA_PREFLIGHT_OK"}],
                    max_tokens=32,
                    temperature=0.0,
                )
                result_dict = asdict(result)
                result_dict["text"] = result.text[:200]
                report["generation"] = result_dict
            except Exception as error:
                report["generation"] = {
                    "status": "unavailable",
                    "error_type": type(error).__name__,
                    "error": str(error)[:500],
                }

    output = output or PROJECT_ROOT / f"runtime/preflight/{provider}.json"
    atomic_write_json(output, report)
    report["output"] = str(output)
    _print_json(report)


@app.command("build-corpus")
def build_corpus(
    config: Path = typer.Option(
        PROJECT_ROOT / "configs/experiments/phase_b.yaml",
        help="Phase B configuration.",
    ),
) -> None:
    """Build paragraph, parent, child, and legacy-alignment corpora."""
    raw = read_yaml(config)["corpus"]
    output_dir = resolve_project_path(raw["output_dir"])
    metadata = write_corpus(
        resolve_project_path(raw["novel_dir"]),
        output_dir,
        encoding=str(raw["encoding"]),
        legacy_short_paragraph_chars=int(raw["legacy_short_paragraph_chars"]),
        parent_target_chars=int(raw["parent_target_chars"]),
        parent_max_chars=int(raw["parent_max_chars"]),
        child_target_chars=int(raw["child_target_chars"]),
        child_max_chars=int(raw["child_max_chars"]),
    )
    _print_json({"output": str(output_dir), **metadata})


@app.command("align-evidence")
def align_released_evidence(
    config: Path = typer.Option(
        PROJECT_ROOT / "configs/experiments/phase_b.yaml",
        help="Phase B configuration.",
    ),
) -> None:
    """Align paraphrased released evidence to reconstructed raw units."""
    raw = read_yaml(config)["alignment"]
    corpus_dir = resolve_project_path(raw["corpus_dir"])
    output_dir = resolve_project_path(raw["output_dir"])
    report = write_evidence_alignment(
        resolve_project_path(raw["evidence_csv"]),
        corpus_dir / "legacy_units.jsonl",
        output_dir,
        candidate_window=int(raw["candidate_window"]),
        high_similarity=float(raw["high_similarity"]),
        medium_similarity=float(raw["medium_similarity"]),
        structural_min_similarity=float(raw["structural_min_similarity"]),
        high_margin=float(raw["high_margin"]),
    )
    _print_json({"output": str(output_dir), **report})


@app.command("build-bm25")
def build_bm25(
    config: Path = typer.Option(
        PROJECT_ROOT / "configs/experiments/phase_b.yaml",
        help="Phase B configuration.",
    ),
) -> None:
    """Build the deterministic character-bigram BM25 baseline index."""
    raw = read_yaml(config)["retrieval"]
    children = read_jsonl(resolve_project_path(raw["children"]))
    index = BM25Index.build(
        children,
        k1=float(raw["k1"]),
        b=float(raw["b"]),
    )
    index_path = resolve_project_path(raw["index"])
    metadata = index.save(index_path)
    _print_json({"output": str(index_path), **metadata})


@app.command("evaluate-bm25")
def evaluate_bm25(
    config: Path = typer.Option(
        PROJECT_ROOT / "configs/experiments/phase_b.yaml",
        help="Phase B configuration.",
    ),
) -> None:
    """Evaluate BM25 retrieval separately from answer generation."""
    raw = read_yaml(config)["retrieval"]
    output_dir = resolve_project_path(raw["output_dir"])
    reports = write_retrieval_evaluation(
        LongCoQADataset(),
        resolve_project_path(raw["index"]),
        resolve_project_path(raw["alignment"]),
        resolve_project_path(raw["children"]),
        output_dir,
        query_modes=tuple(str(value) for value in raw["query_modes"]),
    )
    _print_json({"output": str(output_dir), "reports": reports})


@app.command("run-dense-retrieval")
def run_dense_retrieval(
    config: Path = typer.Option(
        PROJECT_ROOT / "configs/experiments/phase_b.yaml",
        help="Phase B configuration.",
    ),
) -> None:
    """Encode the corpus and evaluate dense plus BM25/dense RRF retrieval."""
    raw = read_yaml(config)["dense"]
    reports = run_dense_and_hybrid(
        model_path=resolve_project_path(raw["model_path"]),
        children_path=resolve_project_path(raw["children"]),
        alignment_path=resolve_project_path(raw["alignment"]),
        bm25_index_path=resolve_project_path(raw["bm25_index"]),
        index_path=resolve_project_path(raw["index"]),
        output_dir=resolve_project_path(raw["output_dir"]),
        dimensions=int(raw["dimensions"]),
        gpu_memory_utilization=float(raw["gpu_memory_utilization"]),
        max_model_len=int(raw["max_model_len"]),
        max_num_seqs=int(raw["max_num_seqs"]),
        query_modes=tuple(str(value) for value in raw["query_modes"]),
        instruction=str(raw["query_instruction"]),
    )
    _print_json(reports)


@app.command("run-reranker")
def run_reranker(
    config: Path = typer.Option(
        PROJECT_ROOT / "configs/experiments/phase_d.yaml",
        help="Phase D RAG configuration.",
    ),
    limit: int | None = typer.Option(None, min=1),
    output_dir: Path | None = typer.Option(None),
    candidates: Path | None = typer.Option(
        None, help="Override the configured first-stage candidate file."
    ),
) -> None:
    """Rerank hybrid candidates with Qwen3-Reranker-4B."""
    raw = read_yaml(config)
    reranker = raw["reranker"]
    summary = run_qwen_reranker(
        model_path=resolve_project_path(reranker["model_path"]),
        candidates_path=resolve_project_path(candidates or reranker["candidates"]),
        children_path=resolve_project_path(raw["children"]),
        output_dir=resolve_project_path(output_dir or reranker["output_dir"]),
        model_revision=str(reranker["revision"]),
        instruction=str(reranker["instruction"]),
        candidate_k=int(reranker["candidate_k"]),
        output_k=int(reranker["output_k"]),
        limit=limit,
        max_model_len=int(reranker["max_model_len"]),
        max_num_seqs=int(reranker["max_num_seqs"]),
        gpu_memory_utilization=float(reranker["gpu_memory_utilization"]),
        batch_size=int(reranker["batch_size"]),
    )
    _print_json(summary)


@app.command("rewrite-queries")
def rewrite_queries(
    config: Path = typer.Option(
        PROJECT_ROOT / "configs/experiments/phase_d.yaml",
        help="Phase D RAG configuration.",
    ),
    limit: int | None = typer.Option(None, min=1),
) -> None:
    """Rewrite contextual questions into standalone retrieval queries."""
    raw = read_yaml(config)["rewrite"]
    summary = run_local_query_rewrite(
        model_path=resolve_project_path(raw["model_path"]),
        output_path=resolve_project_path(raw["output"]),
        model_revision=str(raw["model_revision"]),
        gpu_memory_utilization=float(raw["gpu_memory_utilization"]),
        max_model_len=int(raw["max_model_len"]),
        max_num_seqs=int(raw["max_num_seqs"]),
        limit=limit,
    )
    _print_json(summary)


@app.command("retrieve-rewritten")
def retrieve_rewritten(
    config: Path = typer.Option(
        PROJECT_ROOT / "configs/experiments/phase_d.yaml",
        help="Phase D RAG configuration.",
    ),
) -> None:
    """Retrieve with standalone rewrites against the cached dense index."""
    raw = read_yaml(config)
    rewrite = raw["rewrite"]
    retrieval = raw["rewritten_retrieval"]
    summary = retrieve_rewritten_queries(
        embedding_model_path=resolve_project_path(retrieval["embedding_model_path"]),
        dense_index_path=resolve_project_path(retrieval["dense_index"]),
        rewrites_path=resolve_project_path(rewrite["output"]),
        alignment_path=resolve_project_path(raw["alignment"]),
        children_path=resolve_project_path(raw["children"]),
        output_dir=resolve_project_path(retrieval["output_dir"]),
        dimensions=int(retrieval["dimensions"]),
        top_k=int(retrieval["top_k"]),
        gpu_memory_utilization=float(retrieval["gpu_memory_utilization"]),
        max_model_len=int(retrieval["max_model_len"]),
        max_num_seqs=int(retrieval["max_num_seqs"]),
    )
    _print_json(summary)


@app.command("generate-contextual-corpus")
def generate_contextual_corpus_command(
    config: Path = typer.Option(PROJECT_ROOT / "configs/experiments/phase_d.yaml"),
    limit: int | None = typer.Option(None, min=1),
) -> None:
    """Generate Anthropic-style chunk context with a local Qwen model."""
    raw = read_yaml(config)
    contextual = raw["contextual_retrieval"]
    summary = generate_contextual_corpus(
        model_path=resolve_project_path(contextual["context_model_path"]),
        children_path=resolve_project_path(raw["children"]),
        parents_path=resolve_project_path(raw["parents"]),
        output_path=resolve_project_path(contextual["corpus"]),
        model_revision=str(contextual["context_model_revision"]),
        limit=limit,
        batch_size=int(contextual["generation_batch_size"]),
        gpu_memory_utilization=float(contextual["gpu_memory_utilization"]),
        max_model_len=int(contextual["context_max_model_len"]),
        max_num_seqs=int(contextual["max_num_seqs"]),
    )
    _print_json(summary)


@app.command("run-contextual-retrieval")
def run_contextual_retrieval_command(
    config: Path = typer.Option(PROJECT_ROOT / "configs/experiments/phase_d.yaml"),
) -> None:
    """Evaluate contextual BM25, dense, and RRF with current and rewritten queries."""
    raw = read_yaml(config)
    contextual = raw["contextual_retrieval"]
    summary = run_contextual_retrieval(
        embedding_model_path=resolve_project_path(contextual["embedding_model_path"]),
        contextual_children_path=resolve_project_path(contextual["corpus"]),
        original_children_path=resolve_project_path(raw["children"]),
        alignment_path=resolve_project_path(raw["alignment"]),
        rewrites_path=resolve_project_path(raw["rewrite"]["output"]),
        output_dir=resolve_project_path(contextual["output_dir"]),
        index_path=resolve_project_path(contextual["dense_index"]),
        bm25_index_path=resolve_project_path(contextual["bm25_index"]),
        dimensions=int(contextual["dimensions"]),
        top_k=int(contextual["top_k"]),
        gpu_memory_utilization=float(contextual["gpu_memory_utilization"]),
        max_model_len=int(contextual["embedding_max_model_len"]),
        max_num_seqs=int(contextual["max_num_seqs"]),
    )
    _print_json(summary)


@app.command("run-graph-progressive")
def run_graph_progressive_command(
    config: Path = typer.Option(PROJECT_ROOT / "configs/experiments/phase_d.yaml"),
) -> None:
    """Run the Chinese controlled ChainRAG-style graph progressive retriever."""
    raw = read_yaml(config)
    graph = raw["graph_progressive"]
    summary = run_graph_progressive(
        dense_index_path=resolve_project_path(raw["dense_index"]),
        base_rankings_path=resolve_project_path(graph["base_rankings"]),
        children_path=resolve_project_path(raw["children"]),
        alignment_path=resolve_project_path(raw["alignment"]),
        output_dir=resolve_project_path(graph["output_dir"]),
        semantic_k=int(graph["semantic_k"]),
        position_window=int(graph["position_window"]),
        seed_k=int(graph["seed_k"]),
        steps=int(graph["steps"]),
        top_k=int(graph["top_k"]),
    )
    _print_json(summary)


@app.command("generate-progressive-plans")
def generate_progressive_plans_command(
    config: Path = typer.Option(PROJECT_ROOT / "configs/experiments/phase_d.yaml"),
    limit: int | None = typer.Option(None, min=1),
) -> None:
    """Plan one-to-three evidence-completing subqueries after first-stage retrieval."""
    raw = read_yaml(config)
    progressive = raw["progressive_retrieval"]
    summary = generate_progressive_plans(
        model_path=resolve_project_path(progressive["planner_model_path"]),
        base_rankings_path=resolve_project_path(progressive["base_rankings"]),
        children_path=resolve_project_path(raw["children"]),
        output_path=resolve_project_path(progressive["plans"]),
        limit=limit,
        seed_k=int(progressive["seed_k"]),
        batch_size=int(progressive["batch_size"]),
        gpu_memory_utilization=float(progressive["gpu_memory_utilization"]),
        max_model_len=int(progressive["planner_max_model_len"]),
        max_num_seqs=int(progressive["max_num_seqs"]),
    )
    _print_json(summary)


@app.command("retrieve-progressive")
def retrieve_progressive_command(
    config: Path = typer.Option(PROJECT_ROOT / "configs/experiments/phase_d.yaml"),
) -> None:
    """Retrieve each planned subquery and fuse all progressive rounds."""
    raw = read_yaml(config)
    progressive = raw["progressive_retrieval"]
    summary = retrieve_progressive_plans(
        embedding_model_path=resolve_project_path(progressive["embedding_model_path"]),
        dense_index_path=resolve_project_path(raw["dense_index"]),
        plans_path=resolve_project_path(progressive["plans"]),
        base_rankings_path=resolve_project_path(progressive["base_rankings"]),
        children_path=resolve_project_path(raw["children"]),
        alignment_path=resolve_project_path(raw["alignment"]),
        output_dir=resolve_project_path(progressive["output_dir"]),
        dimensions=int(progressive["dimensions"]),
        top_k=int(progressive["top_k"]),
        gpu_memory_utilization=float(progressive["gpu_memory_utilization"]),
        max_model_len=int(progressive["embedding_max_model_len"]),
        max_num_seqs=int(progressive["max_num_seqs"]),
    )
    _print_json(summary)


@app.command("build-rag-contexts")
def build_rag_context_artifact(
    retrieval: Path = typer.Option(
        PROJECT_ROOT / "reports/generated/rag/rewritten_dense/all_results.jsonl"
    ),
    config: Path = typer.Option(
        PROJECT_ROOT / "configs/experiments/phase_d.yaml",
        help="Phase D RAG configuration.",
    ),
    output: Path | None = typer.Option(
        None, help="Override the configured context JSONL output path."
    ),
    ranking_level: str = typer.Option(
        "child", help="Ranking ID level: child or parent."
    ),
    top_k: int | None = typer.Option(
        None, min=1, help="Override the configured number of unique parents."
    ),
    expected_samples: int = typer.Option(
        3019, min=1, help="Required number of unique retrieval rows."
    ),
) -> None:
    """Expand retrieved children to parents and build grounded prompts."""
    raw = read_yaml(config)
    rag = raw["rag"]
    summary = build_rag_contexts(
        retrieval_path=resolve_project_path(retrieval),
        children_path=resolve_project_path(raw["children"]),
        parents_path=resolve_project_path(raw["parents"]),
        output_path=resolve_project_path(output or rag["contexts"]),
        top_k=int(top_k if top_k is not None else rag["top_k"]),
        chronological=bool(rag["chronological_assembly"]),
        ranking_level=ranking_level,
        expected_samples=expected_samples,
    )
    _print_json(summary)


@app.command("validate-rag-ranking")
def validate_rag_ranking_command(
    input_path: Path = typer.Option(..., "--input"),
    ranking_level: str = typer.Option(..., help="Ranking ID level: child or parent."),
    expected_samples: int = typer.Option(3019, min=1),
) -> None:
    """Validate that a retrieval ranking is safe to use as a formal artifact."""
    _print_json(
        validate_ranking_artifact(
            resolve_project_path(input_path),
            ranking_level=ranking_level,
            expected_samples=expected_samples,
        )
    )


@app.command("validate-rag-generation")
def validate_rag_generation_command(
    input_path: Path = typer.Option(..., "--input"),
    expected_samples: int = typer.Option(3019, min=1),
) -> None:
    """Validate that generated answers form a complete successful artifact."""
    _print_json(
        validate_generation_artifact(
            resolve_project_path(input_path), expected_samples=expected_samples
        )
    )


@app.command("run-rag-local")
def run_rag_local(
    model_config: Path = typer.Option(
        PROJECT_ROOT / "configs/models/qwen3_5_9b.yaml"
    ),
    config: Path = typer.Option(
        PROJECT_ROOT / "configs/experiments/phase_d.yaml"
    ),
    output_dir: Path = typer.Option(
        PROJECT_ROOT / "outputs/rag/qwen3_5_9b_rewritten_parent_top10"
    ),
    contexts: Path | None = typer.Option(
        None, help="Override the configured RAG context JSONL."
    ),
    limit: int | None = typer.Option(None, min=1),
    shard_index: int = typer.Option(0, min=0),
    shard_count: int = typer.Option(1, min=1),
) -> None:
    """Generate answers from retrieved parent contexts with local vLLM."""
    raw = read_yaml(config)
    model = read_yaml(model_config)
    model["path"] = str(resolve_project_path(model["path"]))
    summary = run_local_rag(
        model=model,
        contexts_path=resolve_project_path(contexts or raw["rag"]["contexts"]),
        output_dir=resolve_project_path(output_dir),
        limit=limit,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    _print_json(summary)


@app.command("merge-rag-shards")
def merge_rag_local_shards(
    shard_root: Path = typer.Option(...),
    output_dir: Path = typer.Option(...),
    shard_count: int = typer.Option(8, min=1),
    expected_samples: int = typer.Option(3019, min=1),
) -> None:
    """Validate and merge RAG generation shards."""
    summary = merge_rag_shards(
        shard_root=resolve_project_path(shard_root),
        output_dir=resolve_project_path(output_dir),
        shard_count=shard_count,
        expected_samples=expected_samples,
    )
    _print_json(summary)


def _load_p1_config(config: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = read_yaml(config)
    p1 = raw["p1_inference"]
    model = read_yaml(resolve_project_path(p1["model_config"]))
    model["path"] = str(resolve_project_path(model["path"]))
    for key in ("gpu_memory_utilization", "max_num_seqs", "max_num_batched_tokens"):
        if key in p1:
            model[key] = p1[key]
    if "enable_chunked_prefill" in p1:
        model["enable_chunked_prefill"] = p1["enable_chunked_prefill"]
    return raw, model


@app.command("run-p1-local")
def run_p1_local(
    config: Path = typer.Option(
        PROJECT_ROOT / "configs/experiments/phase_f.yaml",
        help="P1 held-out inference configuration.",
    ),
    output_root: Path = typer.Option(..., help="Shard output root."),
    limit: int | None = typer.Option(None, min=1),
    shard_index: int = typer.Option(0, min=0),
    shard_count: int = typer.Option(1, min=1),
) -> None:
    """Run the P1 Base/LoRA by evaluation-input matrix with one shared vLLM engine."""
    raw, model = _load_p1_config(config)
    p1 = raw["p1_inference"]
    model_variants = {
        str(name): (
            resolve_project_path(value["adapter_path"])
            if value.get("adapter_path")
            else None
        )
        for name, value in p1["model_variants"].items()
    }
    eval_sets = {
        str(name): resolve_project_path(value["path"])
        for name, value in p1["eval_sets"].items()
    }
    generation = p1["generation"]
    summary = run_p1_suite(
        model=model,
        model_variants=model_variants,
        eval_sets=eval_sets,
        output_root=resolve_project_path(output_root),
        seed=int(raw["seed"]),
        limit=limit,
        shard_index=shard_index,
        shard_count=shard_count,
        temperature=float(generation["temperature"]),
        max_tokens_factoid=int(generation["max_tokens_factoid"]),
        max_tokens_non_factoid=int(generation["max_tokens_non_factoid"]),
        max_lora_rank=int(p1["max_lora_rank"]),
        expected_split=str(p1.get("expected_split", "test")),
    )
    _print_json(summary)


@app.command("merge-p1-shards")
def merge_p1_local_shards(
    shard_root: Path = typer.Option(...),
    output_root: Path = typer.Option(...),
    config: Path = typer.Option(
        PROJECT_ROOT / "configs/experiments/phase_f.yaml",
    ),
    shard_count: int = typer.Option(8, min=1),
    limit: int | None = typer.Option(None, min=1),
) -> None:
    """Validate and merge the complete P1 model-by-input matrix."""
    raw, _ = _load_p1_config(config)
    p1 = raw["p1_inference"]
    eval_sets = {
        str(name): resolve_project_path(value["path"])
        for name, value in p1["eval_sets"].items()
    }
    summary = merge_p1_shards(
        shard_root=resolve_project_path(shard_root),
        output_root=resolve_project_path(output_root),
        model_variants=tuple(map(str, p1["model_variants"])),
        eval_sets=eval_sets,
        shard_count=shard_count,
        seed=int(raw["seed"]),
        limit=limit,
        expected_split=str(p1.get("expected_split", "test")),
    )
    _print_json(summary)


@app.command("run-api-smoke")
def run_api_smoke(
    config: Path = typer.Option(
        PROJECT_ROOT / "configs/experiments/phase_c.yaml",
        help="Phase C configuration.",
    ),
    protocol: str | None = typer.Option(
        None,
        help="Override the configured protocol.",
    ),
    samples: int | None = typer.Option(
        None,
        min=1,
        help="Override the configured sample count.",
    ),
) -> None:
    """Run the configured deterministic API smoke experiment."""
    raw = read_yaml(config)
    smoke = raw["smoke"]
    generation = raw["generation"]
    settings = load_provider_settings(str(raw["provider"]))
    protocol_name = protocol or str(smoke["protocol"])
    sample_count = samples if samples is not None else int(smoke["samples"])
    output_dir = (
        resolve_project_path(smoke["output_root"])
        / protocol_name
        / f"smoke{sample_count}"
    )
    summary = run_api_inference(
        LongCoQADataset(),
        settings,
        protocol=protocol_name,
        alignment_path=resolve_project_path(raw["alignment"]),
        cache_path=resolve_project_path(raw["cache"]),
        output_dir=output_dir,
        seed=int(raw["seed"]),
        limit=sample_count,
        temperature=float(generation["temperature"]),
        max_tokens_factoid=int(generation["max_tokens_factoid"]),
        max_tokens_non_factoid=int(generation["max_tokens_non_factoid"]),
    )
    _print_json(summary)


@app.command("run-api-suite")
def run_api_protocol_suite(
    config: Path = typer.Option(
        PROJECT_ROOT / "configs/experiments/phase_c.yaml",
        help="Phase C experiment configuration.",
    ),
    samples: int | None = typer.Option(
        None,
        min=1,
        help="Override the configured sample count.",
    ),
    full: bool = typer.Option(
        False,
        "--full/--smoke",
        help="Run the complete P0 dataset.",
    ),
    protocol: list[str] | None = typer.Option(None, "--protocol"),
    output_root: Path | None = typer.Option(
        None,
        help="Override the provider output root.",
    ),
) -> None:
    """Run one checkpointed API client across the configured P0 protocols."""
    raw = read_yaml(config)
    smoke = raw["smoke"]
    generation = raw["generation"]
    sample_count = (
        None
        if full
        else (samples if samples is not None else int(smoke["samples"]))
    )
    protocols = tuple(str(value) for value in (protocol or raw["protocols"]))
    summary = run_api_suite(
        LongCoQADataset(),
        load_provider_settings(str(raw["provider"])),
        protocols=protocols,
        alignment_path=resolve_project_path(raw["alignment"]),
        cache_path=resolve_project_path(raw["cache"]),
        output_root=resolve_project_path(output_root or smoke["output_root"]),
        seed=int(raw["seed"]),
        limit=sample_count,
        temperature=float(generation["temperature"]),
        max_tokens_factoid=int(generation["max_tokens_factoid"]),
        max_tokens_non_factoid=int(generation["max_tokens_non_factoid"]),
    )
    _print_json(summary)


@app.command("run-local-smoke")
def run_local_smoke(
    config: Path = typer.Option(
        PROJECT_ROOT / "configs/experiments/phase_c.yaml",
        help="Phase C experiment configuration.",
    ),
    model_config: Path = typer.Option(
        PROJECT_ROOT / "configs/models/qwen3_5_9b.yaml",
        help="Local vLLM model configuration.",
    ),
    samples: int | None = typer.Option(
        None,
        min=1,
        help="Override the configured sample count for a smaller preflight.",
    ),
) -> None:
    """Run the same deterministic smoke experiment with a local open model."""
    raw = read_yaml(config)
    model = read_yaml(model_config)
    model["path"] = str(resolve_project_path(model["path"]))
    smoke = raw["smoke"]
    generation = raw["generation"]
    sample_count = samples if samples is not None else int(smoke["samples"])
    output_dir = (
        resolve_project_path(model["output_root"])
        / str(smoke["protocol"])
        / f"smoke{sample_count}"
    )
    summary = run_local_inference(
        LongCoQADataset(),
        model,
        protocol=str(smoke["protocol"]),
        alignment_path=resolve_project_path(raw["alignment"]),
        output_dir=output_dir,
        seed=int(raw["seed"]),
        limit=sample_count,
        temperature=float(generation["temperature"]),
        max_tokens_factoid=int(generation["max_tokens_factoid"]),
        max_tokens_non_factoid=int(generation["max_tokens_non_factoid"]),
    )
    _print_json(summary)


@app.command("run-local-suite")
def run_local_protocol_suite(
    config: Path = typer.Option(
        PROJECT_ROOT / "configs/experiments/phase_c.yaml",
        help="Phase C experiment configuration.",
    ),
    model_config: Path = typer.Option(
        PROJECT_ROOT / "configs/models/qwen3_5_9b.yaml",
        help="Local vLLM model configuration.",
    ),
    samples: int | None = typer.Option(
        None,
        min=1,
        help="Override the configured sample count.",
    ),
    full: bool = typer.Option(
        False,
        "--full/--smoke",
        help="Run the complete P0 dataset.",
    ),
    output_root: Path | None = typer.Option(
        None,
        help="Override the model output root.",
    ),
    cuda_visible_devices: str | None = typer.Option(
        None,
        help="Override CUDA_VISIBLE_DEVICES for this lane.",
    ),
    shard_index: int = typer.Option(0, min=0),
    shard_count: int = typer.Option(1, min=1),
) -> None:
    """Run all configured protocols while reusing one local vLLM engine."""
    raw = read_yaml(config)
    model = read_yaml(model_config)
    model["path"] = str(resolve_project_path(model["path"]))
    if cuda_visible_devices is not None:
        model["cuda_visible_devices"] = cuda_visible_devices
    smoke = raw["smoke"]
    generation = raw["generation"]
    sample_count = (
        None
        if full
        else (samples if samples is not None else int(smoke["samples"]))
    )
    summary = run_local_suite(
        LongCoQADataset(),
        model,
        protocols=tuple(str(protocol) for protocol in raw["protocols"]),
        alignment_path=resolve_project_path(raw["alignment"]),
        output_root=resolve_project_path(output_root or model["output_root"]),
        seed=int(raw["seed"]),
        limit=sample_count,
        temperature=float(generation["temperature"]),
        max_tokens_factoid=int(generation["max_tokens_factoid"]),
        max_tokens_non_factoid=int(generation["max_tokens_non_factoid"]),
        shard_index=shard_index,
        shard_count=shard_count,
    )
    _print_json(summary)


@app.command("run-gpt-oss-suite")
def run_gpt_oss_protocol_suite(
    config: Path = typer.Option(
        PROJECT_ROOT / "configs/experiments/phase_c.yaml",
        help="Phase C experiment configuration.",
    ),
    model_config: Path = typer.Option(
        PROJECT_ROOT / "configs/models/gpt_oss_120b.yaml",
        help="Pinned gpt-oss-120b Harmony configuration.",
    ),
    samples: int | None = typer.Option(
        None,
        min=1,
        help="Override the configured sample count.",
    ),
    full: bool = typer.Option(
        False,
        "--full/--smoke",
        help="Run the complete P0 dataset.",
    ),
    protocol: list[str] | None = typer.Option(None, "--protocol"),
    output_root: Path | None = typer.Option(
        None,
        help="Override the model output root.",
    ),
    cuda_visible_devices: str | None = typer.Option(
        None,
        help="Override CUDA_VISIBLE_DEVICES for this one-GPU lane.",
    ),
    shard_index: int = typer.Option(0, min=0),
    shard_count: int = typer.Option(1, min=1),
) -> None:
    """Run gpt-oss with low reasoning and explicit Harmony channel parsing."""
    raw = read_yaml(config)
    model = read_yaml(model_config)
    model["path"] = str(resolve_project_path(model["path"]))
    if cuda_visible_devices is not None:
        model["cuda_visible_devices"] = cuda_visible_devices
    smoke = raw["smoke"]
    generation = raw["generation"]
    sample_count = (
        None
        if full
        else (samples if samples is not None else int(smoke["samples"]))
    )
    protocols = tuple(str(value) for value in (protocol or raw["protocols"]))
    summary = run_gpt_oss_suite(
        LongCoQADataset(),
        model,
        protocols=protocols,
        alignment_path=resolve_project_path(raw["alignment"]),
        output_root=resolve_project_path(output_root or model["output_root"]),
        seed=int(raw["seed"]),
        limit=sample_count,
        temperature=float(generation["temperature"]),
        max_tokens_factoid=int(model["max_tokens_factoid"]),
        max_tokens_non_factoid=int(model["max_tokens_non_factoid"]),
        shard_index=shard_index,
        shard_count=shard_count,
    )
    _print_json(summary)


@app.command("merge-local-shards")
def merge_local_inference_shards(
    shard_root: Path = typer.Option(...),
    output_root: Path = typer.Option(...),
    config: Path = typer.Option(
        PROJECT_ROOT / "configs/experiments/phase_c.yaml",
        help="Phase C experiment configuration.",
    ),
    shard_count: int = typer.Option(8, min=1),
    expected_samples: int = typer.Option(3019, min=1),
) -> None:
    """Validate and merge deterministic local-inference shards."""
    raw = read_yaml(config)
    summary = merge_local_shards(
        resolve_project_path(shard_root),
        resolve_project_path(output_root),
        protocols=tuple(str(protocol) for protocol in raw["protocols"]),
        shard_count=shard_count,
        expected_samples=expected_samples,
    )
    _print_json(summary)


@app.command("judge-inference")
def judge_inference(
    input_path: Path = typer.Option(..., "--input", help="Inference results JSONL."),
    output_dir: Path = typer.Option(..., help="Judgment checkpoint and report directory."),
    config: Path = typer.Option(
        PROJECT_ROOT / "configs/experiments/phase_c.yaml",
        help="Phase C experiment configuration.",
    ),
    limit: int | None = typer.Option(None, min=1, help="Optional preflight sample limit."),
) -> None:
    """Evaluate inference results with the configured structured LLM judge."""
    raw = read_yaml(config)["judge"]
    settings = load_provider_settings(str(raw["provider"]))
    summary = run_llm_judge(
        resolve_project_path(input_path),
        settings,
        cache_path=resolve_project_path(raw["cache"]),
        output_dir=resolve_project_path(output_dir),
        limit=limit,
        max_tokens=int(raw["max_tokens"]),
    )
    _print_json(summary)


@app.command("judge-p1")
def judge_p1(
    config: Path = typer.Option(
        PROJECT_ROOT / "configs/experiments/phase_f.yaml",
        help="Phase F experiment configuration.",
    ),
    output_root: Path | None = typer.Option(
        None, help="Override the configured P1 judgment output root."
    ),
    limit: int | None = typer.Option(None, min=1, help="Optional preflight limit."),
    model_variant: str | None = typer.Option(
        None, help="Single model variant; requires --eval-variant."
    ),
    eval_variant: str | None = typer.Option(
        None, help="Single evaluation variant; requires --model-variant."
    ),
) -> None:
    """Run the structured judge over one P1 cell or the full 4x3 matrix."""
    raw = read_yaml(config)["p1_judge"]
    settings = load_provider_settings(str(raw["provider"]))
    summary = run_p1_judge_matrix(
        settings,
        input_root=resolve_project_path(raw["input_root"]),
        output_root=resolve_project_path(output_root or raw["output_root"]),
        cache_path=resolve_project_path(raw["cache"]),
        expected_samples={
            str(key): int(value) for key, value in raw["expected_samples"].items()
        },
        model_variants=tuple(str(value) for value in raw["model_variants"]),
        eval_variants=tuple(str(value) for value in raw["eval_variants"]),
        model_variant=model_variant,
        eval_variant=eval_variant,
        limit=limit,
        max_tokens=int(raw["max_tokens"]),
    )
    _print_json(summary)


@app.command("evaluate-inference")
def evaluate_inference(
    input_path: Path = typer.Option(..., "--input"),
    output: Path = typer.Option(...),
    bootstrap_iterations: int = typer.Option(2000, min=100),
    seed: int = typer.Option(42),
) -> None:
    """Compute stratified automatic metrics and dialogue-cluster bootstrap CIs."""
    report = evaluate_inference_results(
        resolve_project_path(input_path),
        resolve_project_path(output),
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )
    _print_json(report)


@app.command("evaluate-paper-metrics")
def evaluate_paper_metrics_command(
    input_path: Path = typer.Option(..., "--input", help="Existing inference results JSONL."),
    config: Path = typer.Option(
        PROJECT_ROOT / "configs/metrics/paper_automatic.yaml",
        help="Locked paper metric profiles.",
    ),
    output_root: Path = typer.Option(
        PROJECT_ROOT / "reports/generated/evaluation/paper_metrics",
        help="Versioned paper metric output root.",
    ),
    profile_id: str | None = typer.Option(None, "--profile", help="Run one profile ID."),
    device: str = typer.Option("cpu", help="Device for optional learned metric scorers."),
    batch_size: int = typer.Option(16, min=1, help="Learned metric batch size."),
    bootstrap_iterations: int = typer.Option(2000, min=1),
    seed: int = typer.Option(42),
) -> None:
    """Evaluate paper-compatible lexical metrics over existing model answers."""
    input_path = resolve_project_path(input_path)
    output_root = resolve_project_path(output_root)
    profiles = load_metric_profiles(resolve_project_path(config))
    selected = (
        {profile_id: profiles[profile_id]}
        if profile_id is not None and profile_id in profiles
        else profiles
    )
    if profile_id is not None and profile_id not in profiles:
        raise typer.BadParameter(f"Unknown metric profile: {profile_id}", param_hint="--profile")
    rows = read_jsonl(input_path)
    summaries: dict[str, Any] = {}
    for current_id, metric_profile in selected.items():
        profile_root = output_root / current_id
        if metric_profile.model_id:
            adapter = build_learned_adapter(
                metric_profile,
                device=device,
                batch_size=batch_size,
            )
            scored = score_batch_rows(rows, metric_profile, profile_root, adapter)
        else:
            scored = score_rows(rows, metric_profile, profile_root)
        atomic_write_jsonl(profile_root / "scored_results.jsonl", scored)
        report = aggregate_paper_scores(
            scored,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
        )
        report["input"] = str(input_path)
        report["profile"] = metric_profile.to_dict()
        atomic_write_json(profile_root / "report.json", report)
        visible_row = next((row for row in rows if row.get("status") == "ok"), {})
        export_paper_table(
            [
                {
                    "model": visible_row.get("model", "unknown"),
                    "protocol": visible_row.get("protocol", "unknown"),
                    "report": report,
                }
            ],
            metric_order=metric_profile.metrics,
            csv_path=profile_root / "paper_table.csv",
            markdown_path=profile_root / "paper_table.md",
            metadata_path=profile_root / "paper_table.metadata.json",
        )
        summaries[current_id] = {
            "output": str(profile_root / "report.json"),
            "successful_samples": report["successful_samples"],
        }
    _print_json({"schema_version": "longcoqa-paper-metric-suite-v1", "profiles": summaries})


@app.command("evaluate-rag-quality-smoke")
def evaluate_rag_quality_smoke_command(
    input_path: Path = typer.Option(..., "--input", help="One RAG evaluation cell JSONL."),
    provider: str = typer.Option(..., help="Provider name from configs/providers.yaml."),
    cell: str = typer.Option(..., help="Single model/protocol cell identity."),
    output_root: Path = typer.Option(
        PROJECT_ROOT / "runtime/paper_metrics_api_smoke/rag_quality",
        help="Isolated runtime smoke output root.",
    ),
    cache_path: Path = typer.Option(
        PROJECT_ROOT / "runtime/paper_metrics_api_smoke/rag_quality_cache.sqlite3"
    ),
    smoke: bool = typer.Option(False, "--smoke", help="Required external API safety flag."),
    limit: int = typer.Option(12, min=1),
    max_tokens: int = typer.Option(1024, min=64),
) -> None:
    """Run semantic claim/support evaluation for exactly 12 stratified API rows."""
    input_path = resolve_project_path(input_path)
    output_root = resolve_project_path(output_root)
    validate_api_smoke_request(
        smoke=smoke,
        limit=limit,
        cells=(cell,),
        output_root=output_root,
    )
    rows = read_jsonl(input_path)
    select_stratified_smoke12(rows)
    settings = load_provider_settings(provider)
    with ResponseCache(resolve_project_path(cache_path)) as cache, ProviderClient(
        settings, cache=cache
    ) as client:
        report = run_guarded_api_smoke(
            rows,
            cell=cell,
            output_root=output_root,
            evaluator=make_rag_quality_evaluator(client, max_tokens=max_tokens),
            smoke=smoke,
            limit=limit,
            evaluator_identity={
                "task": "rag_claim_support",
                "provider": settings.safe_summary(),
            },
        )
    _print_json(report)


@app.command("evaluate-judge-smoke")
def evaluate_judge_smoke_command(
    input_path: Path = typer.Option(..., "--input", help="One evaluation cell JSONL."),
    provider: str = typer.Option(..., help="Provider name from configs/providers.yaml."),
    profile_id: str = typer.Option(..., "--profile", help="Locked judge profile ID."),
    cell: str = typer.Option(..., help="Single model/protocol cell identity."),
    config: Path = typer.Option(
        PROJECT_ROOT / "configs/metrics/judge_profiles.yaml",
        help="Locked judge profile definitions.",
    ),
    output_root: Path = typer.Option(
        PROJECT_ROOT / "runtime/paper_metrics_api_smoke/judge",
        help="Isolated runtime smoke output root.",
    ),
    cache_path: Path = typer.Option(
        PROJECT_ROOT / "runtime/paper_metrics_api_smoke/judge_cache.sqlite3"
    ),
    smoke: bool = typer.Option(False, "--smoke", help="Required external API safety flag."),
    limit: int = typer.Option(12, min=1),
    max_tokens: int = typer.Option(256, min=32),
) -> None:
    """Run one locked judge profile over exactly 12 stratified API rows."""
    input_path = resolve_project_path(input_path)
    output_root = resolve_project_path(output_root)
    validate_api_smoke_request(
        smoke=smoke,
        limit=limit,
        cells=(cell,),
        output_root=output_root,
    )
    rows = read_jsonl(input_path)
    select_stratified_smoke12(rows)
    profiles = load_judge_profiles(resolve_project_path(config))
    if profile_id not in profiles:
        raise typer.BadParameter(f"Unknown judge profile: {profile_id}", param_hint="--profile")
    profile = profiles[profile_id]
    settings = load_provider_settings(provider)
    with ResponseCache(resolve_project_path(cache_path)) as cache, ProviderClient(
        settings, cache=cache
    ) as client:
        report = run_guarded_api_smoke(
            rows,
            cell=cell,
            output_root=output_root,
            evaluator=make_judge_evaluator(client, profile, max_tokens=max_tokens),
            smoke=smoke,
            limit=limit,
            evaluator_identity={
                "task": "answer_judge",
                "provider": settings.safe_summary(),
                "profile": profile.to_dict(),
                "profile_hash": profile.profile_hash,
            },
        )
    _print_json(report)


@app.command("export-human-annotation")
def export_human_annotation_command(
    input_path: Path = typer.Option(..., "--input", help="Evaluation rows JSONL."),
    public_output: Path = typer.Option(..., help="Blinded annotation CSV."),
    private_key_output: Path = typer.Option(..., help="Private identity key JSON."),
    blind_salt: str = typer.Option(..., help="Secret salt for stable presentation IDs."),
) -> None:
    """Export a blinded human annotation sheet and a separate private key."""
    rows = read_jsonl(resolve_project_path(input_path))
    public_output = resolve_project_path(public_output)
    private_key_output = resolve_project_path(private_key_output)
    export_human_annotation(
        rows,
        public_path=public_output,
        key_path=private_key_output,
        blind_salt=blind_salt,
    )
    _print_json(
        {
            "samples": len(rows),
            "public_output": str(public_output),
            "private_key_output": str(private_key_output),
        }
    )


@app.command("import-human-annotations")
def import_human_annotations_command(
    input_path: Path = typer.Option(..., "--input", help="Completed annotation CSV."),
    output: Path = typer.Option(..., help="Validated annotation JSONL."),
) -> None:
    """Validate completed human labels and write normalized JSONL."""
    rows = import_human_annotations(resolve_project_path(input_path))
    output = resolve_project_path(output)
    atomic_write_jsonl(output, rows)
    _print_json({"samples": len(rows), "output": str(output)})


@app.command("calibrate-judges")
def calibrate_judges_command(
    judge_input: Path = typer.Option(..., help="Judge scores JSONL."),
    human_input: Path = typer.Option(..., help="Validated human annotations JSONL."),
    output: Path = typer.Option(..., help="Calibration report JSON."),
) -> None:
    """Compare judge scores with completed human annotations."""
    report = calibrate_judges(
        read_jsonl(resolve_project_path(judge_input)),
        read_jsonl(resolve_project_path(human_input)),
    )
    output = resolve_project_path(output)
    atomic_write_json(output, report)
    _print_json({**report, "output": str(output)})


@app.command("evaluate-p1-cpt")
def evaluate_p1_cpt_command(
    formal_root: Path = typer.Option(
        PROJECT_ROOT / "outputs/formal/p1_cpt",
        help="Completed CPT P1 formal output root.",
    ),
    base_root: Path = typer.Option(
        PROJECT_ROOT / "outputs/formal/p1/merged",
        help="Existing Base P1 merged output root.",
    ),
    output_root: Path = typer.Option(
        PROJECT_ROOT / "reports/generated/evaluation/p1_cpt",
        help="CPT evaluation report root.",
    ),
    bootstrap_iterations: int = typer.Option(2000, min=100),
    seed: int = typer.Option(42),
) -> None:
    """Evaluate CPT on all P1 inputs and compare it with the matched Base rows."""
    report = evaluate_p1_cpt(
        resolve_project_path(formal_root),
        resolve_project_path(base_root),
        resolve_project_path(output_root),
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )
    _print_json(report)


@app.command("compare-inference")
def compare_inference(
    left: Path = typer.Option(...),
    right: Path = typer.Option(...),
    output: Path = typer.Option(...),
    bootstrap_iterations: int = typer.Option(2000, min=100),
    seed: int = typer.Option(42),
) -> None:
    """Run a paired dialogue-cluster bootstrap comparison of two answer files."""
    report = compare_inference_results(
        resolve_project_path(left),
        resolve_project_path(right),
        resolve_project_path(output),
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )
    _print_json(report)


@app.command("build-training-data")
def build_training_data(
    config: Path = typer.Option(
        PROJECT_ROOT / "configs/experiments/phase_e.yaml",
        help="Phase E training-data configuration.",
    ),
) -> None:
    """Build leakage-controlled SFT, grounded-SFT, RAFT, and CPT files."""
    raw = read_yaml(config)
    data = raw["data"]
    output_dir = resolve_project_path(data["output_dir"])
    supervised = build_supervised_datasets(
        manifest_path=resolve_project_path(data["p1_manifest"]),
        alignment_path=resolve_project_path(data["alignment"]),
        children_path=resolve_project_path(data["children"]),
        retrieval_path=resolve_project_path(data["hard_negative_retrieval"]),
        bm25_index_path=resolve_project_path(data["bm25_index"]),
        output_dir=output_dir,
        seed=int(raw["seed"]),
        hard_negatives=int(data["hard_negatives"]),
        raft_gold_probability=float(data["raft_gold_probability"]),
        raft_gold_only_probability=float(data["raft_gold_only_probability"]),
        unannotated_evidence_policy=str(data["unannotated_evidence_policy"]),
    )
    cpt = build_cpt_dataset(
        novel_dir=resolve_project_path(data["novel_dir"]),
        output_dir=output_dir,
        encoding=str(data["novel_encoding"]),
        chunk_chars=int(data["cpt_chunk_chars"]),
        overlap_chars=int(data["cpt_overlap_chars"]),
    )
    _print_json({"supervised": supervised, "cpt": cpt})


if __name__ == "__main__":
    app()
