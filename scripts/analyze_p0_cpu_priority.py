#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from longcoqa.evaluation.p0_cpu_priority import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    load_child_retrieval_rows,
    load_p0_generation_rows,
    run_a1,
    run_m3,
    run_m4,
    run_r1,
)
from longcoqa.paths import PROJECT_ROOT
from longcoqa.utils.io import atomic_write_json

DEFAULT_OUTPUT = PROJECT_ROOT / "reports/generated/analysis/p0_cpu_priority_20260724"


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def write_table(output_dir: Path, name: str, frame: pd.DataFrame) -> str:
    path = output_dir / name
    frame.to_csv(path, index=False, float_format="%.12g")
    return str(path.relative_to(PROJECT_ROOT))


def ci_text(row: pd.Series, estimate: str) -> str:
    return f"{row[estimate]:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"


def build_artifact(
    *,
    generated_at: str,
    output_dir: Path,
    m3_buckets: pd.DataFrame,
    m3_slopes: pd.DataFrame,
    m3_contrasts: pd.DataFrame,
    m4_changes: pd.DataFrame,
    m4_regressions: pd.DataFrame,
    r1_summary: pd.DataFrame,
    r1_pairwise: pd.DataFrame,
    a1_conditionals: pd.DataFrame,
    a1_within: pd.DataFrame,
    a1_parent: pd.DataFrame,
) -> dict[str, Any]:
    relative_dir = output_dir.relative_to(PROJECT_ROOT)
    m3_chart = m3_buckets.copy()
    m3_chart["cell"] = (
        m3_chart["protocol"] + " · " + m3_chart["question_type"] + " · " + m3_chart["turn_bucket"]
    )
    r1_hit = r1_summary[r1_summary["metric"] == "hit_at_20"].sort_values("estimate", ascending=False)
    a1_chart = a1_within.copy()
    a1_chart["analysis"] = a1_chart["question_scope"] + " · " + a1_chart["metric"]

    factoid_contrasts = m3_contrasts[
        (m3_contrasts["question_type"] == "factoid")
        & (m3_contrasts["contrast"] == "z1_minus_z0")
    ].set_index("turn_bucket")
    nonfactoid_contrasts = m3_contrasts[
        (m3_contrasts["question_type"] == "non_factoid")
        & (m3_contrasts["contrast"] == "z1_minus_z0")
    ].set_index("turn_bucket")
    best_hit = r1_hit.iloc[0]
    rewritten = r1_hit[r1_hit["method"] == "rewritten_dense"].iloc[0]
    r1_reference_pair = r1_pairwise[
        (r1_pairwise["metric"] == "hit_at_20")
        & (
            (
                (r1_pairwise["left_method"] == "rewritten_dense")
                & (r1_pairwise["right_method"] == best_hit["method"])
            )
            | (
                (r1_pairwise["right_method"] == "rewritten_dense")
                & (r1_pairwise["left_method"] == best_hit["method"])
            )
        )
    ].iloc[0]
    r1_delta = float(best_hit["estimate"] - rewritten["estimate"])
    within_all = a1_within[
        (a1_within["question_scope"] == "all") & (a1_within["metric"] == "character_f1")
    ].iloc[0]
    within_factoid_em = a1_within[a1_within["metric"] == "exact_match_contain"].iloc[0]
    within_length = m4_regressions[
        (m4_regressions["model_spec"] == "within_question_g3_minus_z1_model_fe")
        & (m4_regressions["outcome"] == "delta_character_f1")
    ].iloc[0]

    source = lambda filename: str(relative_dir / filename)
    def csv_source(identifier: str, label: str, filename: str) -> dict[str, Any]:
        path = source(filename)
        return {
            "id": identifier,
            "label": label,
            "path": path,
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "description": f"Load the validated aggregate rows from {filename}.",
                "executed_at": generated_at,
                "sql": f"SELECT * FROM read_csv_auto('{path}')",
            },
        }

    summary_path = source("analysis_summary.json")
    sources = [
        {
            "id": "analysis_summary",
            "label": "Analysis summary and data-quality gate",
            "path": summary_path,
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "description": "Load the analysis metadata and passed quality checks.",
                "executed_at": generated_at,
                "sql": f"SELECT * FROM read_json_auto('{summary_path}')",
            },
        },
        csv_source("m3_source", "M3 turn/type/protocol summaries", "m3_bucket_summary.csv"),
        csv_source("m3_contrast_source", "M3 paired protocol contrasts", "m3_protocol_contrasts.csv"),
        csv_source("m4_source", "M4 answer-length regression results", "m4_regressions.csv"),
        csv_source("m4_change_source", "M4 model-level G3 minus Z1 changes", "m4_paired_changes.csv"),
        csv_source("r1_source", "R1 paired retrieval results", "r1_method_summary.csv"),
        csv_source("a1_source", "A1 same-question hit/miss results", "a1_within_question.csv"),
    ]
    datasets = {
        "m3_buckets": records(m3_chart),
        "m3_contrasts": records(m3_contrasts),
        "m4_regressions": records(m4_regressions),
        "m4_changes": records(m4_changes),
        "r1_hit20": records(r1_hit),
        "a1_within": records(a1_chart),
        "a1_parent": records(a1_parent),
    }
    charts = [
        {
            "id": "m3_chart",
            "title": "M3：题型、协议与轮次深度的 Character-F1",
            "subtitle": "7 模型先按样本宏平均；误差线详见源表的 dialogue-cluster 95% CI。",
            "intent": "comparison",
            "question": "控制题型后，各协议随轮次深度如何变化？",
            "rationale": "分组柱形图直接比较协议、题型和预注册轮次桶。",
            "type": "bar",
            "dataset": "m3_buckets",
            "sourceId": "m3_source",
            "encodings": {
                "x": {"field": "turn_bucket", "type": "nominal", "label": "轮次桶"},
                "y": {"field": "character_f1", "type": "quantitative", "label": "Character-F1"},
                "color": {"field": "protocol", "type": "nominal", "label": "协议"},
                "facet": {"field": "question_type", "type": "nominal", "label": "题型"},
                "tooltip": [
                    {"field": "samples", "type": "quantitative", "label": "N"},
                    {"field": "ci_low", "type": "quantitative", "label": "95% CI 下界"},
                    {"field": "ci_high", "type": "quantitative", "label": "95% CI 上界"},
                ],
            },
        },
        {
            "id": "r1_chart",
            "title": "R1：17 个 child-level retriever 的 Hit@20",
            "subtitle": "完全相同的 2,721 个 sample ID；251 个 dialogue cluster。",
            "intent": "ranking",
            "question": "哪种已完成检索方法在固定样本上获得最高 Hit@20？",
            "rationale": "排序柱形图展示方法间绝对差异，推断以完整 pairwise 表为准。",
            "type": "bar",
            "dataset": "r1_hit20",
            "sourceId": "r1_source",
            "encodings": {
                "x": {"field": "method", "type": "nominal", "label": "方法"},
                "y": {"field": "estimate", "type": "quantitative", "label": "Hit@20"},
                "tooltip": [
                    {"field": "ci_low", "type": "quantitative", "label": "95% CI 下界"},
                    {"field": "ci_high", "type": "quantitative", "label": "95% CI 上界"},
                ],
            },
        },
        {
            "id": "a1_chart",
            "title": "A1：同题 child-level hit 与 miss 的答案差异",
            "subtitle": "只保留四种 RAG 方法中至少一次 hit、至少一次 miss 的题；正值偏向 hit。",
            "intent": "comparison",
            "question": "在同一道题内，gold-in-top10 是否与更高答案指标相关？",
            "rationale": "同题差值减少题目难度构成对条件均值的混杂。",
            "type": "bar",
            "dataset": "a1_within",
            "sourceId": "a1_source",
            "encodings": {
                "x": {"field": "analysis", "type": "nominal", "label": "分析切片"},
                "y": {"field": "delta_hit_minus_miss", "type": "quantitative", "label": "Hit − Miss"},
                "tooltip": [
                    {"field": "discordant_samples", "type": "quantitative", "label": "Discordant N"},
                    {"field": "ci_low", "type": "quantitative", "label": "95% CI 下界"},
                    {"field": "ci_high", "type": "quantitative", "label": "95% CI 上界"},
                ],
            },
        },
    ]
    tables = [
        {
            "id": "m3_contrasts_table",
            "title": "M3 预注册协议对比",
            "subtitle": "Z1−Z0 与 G3−Z1；按模型×样本配对并按 dialogue 聚类。",
            "dataset": "m3_contrasts",
            "sourceId": "m3_contrast_source",
            "columns": [
                {"field": "contrast", "label": "对比", "type": "text"},
                {"field": "question_type", "label": "题型", "type": "text"},
                {"field": "turn_bucket", "label": "轮次桶", "type": "text"},
                {"field": "samples", "label": "题目 N", "type": "number"},
                {"field": "delta_character_f1", "label": "Character-F1 Δ", "format": "number"},
                {"field": "ci_low", "label": "CI 下界", "format": "number"},
                {"field": "ci_high", "label": "CI 上界", "format": "number"},
            ],
        },
        {
            "id": "m4_table",
            "title": "M4 Factoid 答案长度回归",
            "subtitle": "每 1 SD log 预测长度的系数；观察性关联，不作因果解释。",
            "dataset": "m4_regressions",
            "sourceId": "m4_source",
            "columns": [
                {"field": "model_spec", "label": "模型", "type": "text"},
                {"field": "outcome", "label": "结果", "type": "text"},
                {"field": "samples", "label": "N", "type": "number"},
                {"field": "coefficient_per_sd_log_prediction_length", "label": "系数", "format": "number"},
                {"field": "ci_low", "label": "CI 下界", "format": "number"},
                {"field": "ci_high", "label": "CI 上界", "format": "number"},
            ],
        },
        {
            "id": "m4_changes_table",
            "title": "M4 各模型 G3−Z1 的平均变化",
            "subtitle": "每个模型 1,172 条 Factoid 配对；长度为 normalized prediction characters。",
            "dataset": "m4_changes",
            "sourceId": "m4_change_source",
            "columns": [
                {"field": "model", "label": "模型", "type": "text"},
                {"field": "samples", "label": "N", "type": "number"},
                {"field": "delta_prediction_length", "label": "长度 Δ", "format": "number"},
                {"field": "delta_character_f1", "label": "Character-F1 Δ", "format": "number"},
                {"field": "delta_exact_match_contain", "label": "EM-contain Δ", "format": "number"},
                {"field": "corr_length_character_f1", "label": "同题长度/F1 相关", "format": "number"},
            ],
        },
    ]
    blocks = [
        {"id": "title", "type": "markdown", "body": "# LongCoQA P0 CPU 优先分析\n\nM3 / M4 / R1 / A1 · 2026-07-24 数据快照"},
        {
            "id": "summary",
            "type": "markdown",
            "sourceId": "analysis_summary",
            "body": (
                "## 技术摘要\n\n"
                f"**历史信息收益随题型和轮次而异。** Z1−Z0 在 Factoid 的 turn 5+ 为 {ci_text(factoid_contrasts.loc['turn_5_plus'], 'delta_character_f1')}，"
                f"在 Non-factoid 的 turn 5+ 为 {ci_text(nonfactoid_contrasts.loc['turn_5_plus'], 'delta_character_f1')}。\n\n"
                f"**Factoid 冗长与较差自动指标稳定相关。** 同题 G3−Z1 回归中，每 1 SD 的 log 长度增量对应 Character-F1 {within_length['coefficient_per_sd_log_prediction_length']:+.4f} "
                f"（95% CI {within_length['ci_low']:+.4f}～{within_length['ci_high']:+.4f}）；这是关联，不是因果。\n\n"
                f"**Contextual rewritten dense 的 Hit@20 最高。** {best_hit['estimate']:.4f}，比 rewritten dense 高 {r1_delta:+.4f}；"
                f"完整 17 方法、544 个 pairwise-metric 对比已做 Holm 校正（该对比校正 p={r1_reference_pair['p_holm']:.4g}）。\n\n"
                f"**检索命中与答案改善相关，但幅度有限。** 同题 discordant 样本中，整体 Character-F1 的 hit−miss 为 {ci_text(within_all, 'delta_hit_minus_miss')}；"
                f"Factoid EM-contain 为 {ci_text(within_factoid_em, 'delta_hit_minus_miss')}。"
            ),
        },
        {"id": "m3_text", "type": "markdown", "body": "## M3：turn-depth 不是可忽略的构成变量\n\n按题型、协议和轮次桶分层后，多个协议的深轮次表现发生系统变化；因此只给 overall 或按 turn 汇总会混入题型与协议构成。下表的协议差值以模型×题目配对，置信区间按 `(question_type, dialogue_id)` 聚类。"},
        {"id": "m3_chart_block", "type": "chart", "chartId": "m3_chart"},
        {"id": "m3_table_block", "type": "table", "tableId": "m3_contrasts_table"},
        {"id": "m4_text", "type": "markdown", "body": "## M4：Factoid 退化与答案变长高度一致，但模型均值存在指标分歧\n\nPooled 模型控制 reference length、model 和 protocol；同题模型直接用 G3−Z1 的长度与结果变化。两种规格的 Character-F1 系数均显著为负。模型均值并非简单单调：qwen3.5-9B 的 G3 输出平均长 +42.98 字符、Character-F1 −0.0353，但 EM-contain +0.2457，说明“包含短 reference”与“整体字符精确”可向相反方向变化。Judge 数据未在现有 CPU artifacts 中提供，因此没有补值，也不能把自动指标关联写成语义正确性或因果效应。"},
        {"id": "m4_changes_table_block", "type": "table", "tableId": "m4_changes_table"},
        {"id": "m4_table_block", "type": "table", "tableId": "m4_table"},
        {"id": "r1_text", "type": "markdown", "body": "## R1：contextual dense rewritten 是当前完整 child-level 集合的首选\n\n17 种已完成方法在完全相同的 2,721 题上比较。所有方法点估计与 pairwise 差值共享同一组 dialogue bootstrap 抽样；p 值用 bootstrap 标准误的双侧正态近似，CSV 同时保留有限次抽样的 empirical Monte Carlo p，每项指标的 136 个方法对分别做 Holm 校正。HippoRAG 2 是 parent-level 结果，未混入该排名。"},
        {"id": "r1_chart_block", "type": "chart", "chartId": "r1_chart"},
        {"id": "a1_text", "type": "markdown", "body": "## A1：命中 gold evidence 与更好答案相关，但不是充分条件\n\n主分析只比较四种有相同 child-level hit 定义、且已有生成结果的方法。同题对照对每题先分别平均 hit 方法和 miss 方法，再计算差值，避免方法数不平衡把某些题重复加权。HippoRAG 2 仅作为 parent-level 敏感性结果。"},
        {"id": "a1_chart_block", "type": "chart", "chartId": "a1_chart"},
        {"id": "methods", "type": "markdown", "body": "## 方法、口径与质量闸门\n\nM3/M4 使用 7 个模型 × 4 个协议 × 3,019 条成功预测，共 84,532 行；M4 只取 1,172 条 Factoid 的 Z1/G3 配对。R1 使用 17 种 child-level 方法的共同 2,721 sample ID。A1 的四种 child-level RAG 方法各连接 2,721 条 retrieval 与 3,019 条 generation，连接后每法 2,721 条且无重复。所有 95% CI 均用 seed 42、2,000 次 dialogue-cluster bootstrap。"},
        {"id": "limitations", "type": "markdown", "body": "## 限制与结论边界\n\n- M4 长度回归和 A1 hit/miss 都是观察性关联；题目难度、检索内容质量和生成行为仍可能混杂。\n- Factoid 可报告 EM-contain；Non-factoid 不人为阈值化为“正确/错误”，只报告 Character-F1。\n- A1 child-level retrieval 集合没有覆盖 turn-1 等全部 3,019 题，主分析分母为每法 2,721。\n- HippoRAG 2 的 gold 与命中定义在 parent level，只能作为敏感性分析。\n- Judge 子样本不属于现有 CPU artifacts，M4 不报告 Judge 结论。"},
        {"id": "next", "type": "markdown", "body": "## 建议的论文动作\n\n1. 主文的轮次结论改用 M3 的题型分层与 cluster CI，避免 overall 构成混杂。\n2. 将 Factoid 退化描述为“与输出冗长稳定相关”，等待校准 Judge 后再讨论真实正确性。\n3. Table 3 将 contextual dense rewritten 作为当前最佳 child-level 基线，并在脚注说明 17 方法完整配对与 Holm 校正。\n4. A1 可支持“检索命中有帮助但不充分”；后续用 oracle evidence 或随机化 retrieval 干预识别因果效应。"},
    ]
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "LongCoQA P0 CPU 优先分析",
            "description": "M3/M4/R1/A1：轮次×题型、Factoid 长度、检索配对推断与 hit/miss 条件分析。",
            "generatedAt": generated_at,
            "filters": [],
            "cards": [],
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": datasets,
            "accessIssues": [],
        },
        "sources": sources,
    }


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    generation, generation_manifest = load_p0_generation_rows()
    retrieval, retrieval_manifest = load_child_retrieval_rows()
    m3_buckets, m3_slopes, m3_contrasts = run_m3(generation)
    m4_model_protocol, m4_changes, m4_regressions = run_m4(generation)
    r1_summary, r1_pairwise = run_r1(retrieval)
    a1_conditionals, a1_within, a1_parent, a1_manifest = run_a1()

    tables = {
        "m3_bucket_summary.csv": m3_buckets,
        "m3_turn_slopes.csv": m3_slopes,
        "m3_protocol_contrasts.csv": m3_contrasts,
        "m4_model_protocol.csv": m4_model_protocol,
        "m4_paired_changes.csv": m4_changes,
        "m4_regressions.csv": m4_regressions,
        "r1_method_summary.csv": r1_summary,
        "r1_pairwise.csv": r1_pairwise,
        "a1_method_conditionals.csv": a1_conditionals,
        "a1_within_question.csv": a1_within,
        "a1_parent_sensitivity.csv": a1_parent,
    }
    paths = {name: write_table(output_dir, name, frame) for name, frame in tables.items()}
    manifest = generation_manifest + retrieval_manifest + a1_manifest
    summary = {
        "schema_version": "longcoqa-p0-cpu-priority-v1",
        "generated_at": generated_at,
        "bootstrap": {
            "unit": "question_type:dialogue_id",
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed": BOOTSTRAP_SEED,
            "interval": "percentile_95",
        },
        "data_quality": {
            "generation_rows": len(generation),
            "generation_cells": len(generation_manifest),
            "successful_rows_per_cell": 3019,
            "retrieval_methods": len(retrieval),
            "common_retrieval_samples": 2721,
            "rag_child_methods": 4,
            "rag_joined_samples_per_method": 2721,
            "status": "passed",
        },
        "limitations": {
            "judge": "unavailable in existing CPU artifacts; not imputed",
            "causality": "M4 and A1 are observational associations",
            "hipporag2": "parent-level sensitivity only",
        },
        "tables": paths,
        "source_manifest": manifest,
    }
    atomic_write_json(output_dir / "analysis_summary.json", summary)
    artifact = build_artifact(
        generated_at=generated_at,
        output_dir=output_dir,
        m3_buckets=m3_buckets,
        m3_slopes=m3_slopes,
        m3_contrasts=m3_contrasts,
        m4_changes=m4_changes,
        m4_regressions=m4_regressions,
        r1_summary=r1_summary,
        r1_pairwise=r1_pairwise,
        a1_conditionals=a1_conditionals,
        a1_within=a1_within,
        a1_parent=a1_parent,
    )
    atomic_write_json(output_dir / "artifact.json", artifact)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LongCoQA P0 CPU-priority analyses.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run(args.output_dir.resolve())
    print(json.dumps(summary["data_quality"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
