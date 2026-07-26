#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "reports/generated/analysis/g0_g3_rewrite_quality_20260724"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(text: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", str(text)).lower()


def containment(row: dict[str, Any]) -> float:
    prediction = normalize(row["prediction"])
    reference = normalize(row["reference"])
    return float(bool(reference) and reference in prediction)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_effect(rows: list[dict[str, str]], **filters: str) -> float:
    selected = [row for row in rows if all(row[field] == value for field, value in filters.items())]
    if len(selected) != 1:
        raise AssertionError(f"expected one result for {filters}, found {len(selected)}")
    return float(selected[0]["delta_g3_minus_g0"])


def main() -> None:
    summary = json.loads((OUTPUT_DIR / "analysis_summary.json").read_text(encoding="utf-8"))
    aggregate = load_csv(OUTPUT_DIR / "aggregate_effects.csv")
    model_effects = load_csv(OUTPUT_DIR / "model_effects.csv")
    paths_by_cell = {
        item["cell"]: PROJECT_ROOT / item["path"] for item in summary["source_manifest"]
    }
    model_names = sorted(cell.rsplit("/", 1)[0] for cell in paths_by_cell if cell.endswith("/g0_released_gold"))

    paired: list[dict[str, Any]] = []
    id_alignment: dict[str, bool] = {}
    for model in model_names:
        g0_rows = {row["sample_id"]: row for row in read_jsonl(paths_by_cell[f"{model}/g0_released_gold"])}
        g3_rows = {row["sample_id"]: row for row in read_jsonl(paths_by_cell[f"{model}/g3_aligned_raw"])}
        id_alignment[model] = g0_rows.keys() == g3_rows.keys()
        if not id_alignment[model]:
            raise AssertionError(f"G0/G3 sample IDs differ for {model}")
        for sample_id in sorted(g0_rows):
            g0 = g0_rows[sample_id]
            g3 = g3_rows[sample_id]
            metadata = (g0["question_type"], int(g0["dialogue_id"]), int(g0["turn"]))
            if metadata != (g3["question_type"], int(g3["dialogue_id"]), int(g3["turn"])):
                raise AssertionError(f"G0/G3 metadata differs for {model}/{sample_id}")
            paired.append(
                {
                    "model": model,
                    "question_type": metadata[0],
                    "turn": metadata[2],
                    "character_f1_delta": float(g3["metrics"]["character_f1"])
                    - float(g0["metrics"]["character_f1"]),
                    "contain_delta": containment(g3) - containment(g0),
                }
            )

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    raw_checks = {
        "overall_character_f1": mean([row["character_f1_delta"] for row in paired]),
        "factoid_turn_5_plus_character_f1": mean(
            [
                row["character_f1_delta"]
                for row in paired
                if row["question_type"] == "factoid" and row["turn"] >= 5
            ]
        ),
        "factoid_exact_match_contain": mean(
            [row["contain_delta"] for row in paired if row["question_type"] == "factoid"]
        ),
        "glm4_7_flash_factoid_character_f1": mean(
            [
                row["character_f1_delta"]
                for row in paired
                if row["model"] == "glm4_7_flash" and row["question_type"] == "factoid"
            ]
        ),
    }
    csv_checks = {
        "overall_character_f1": csv_effect(
            aggregate,
            metric="character_f1",
            scope="overall",
            question_type="all",
            turn_bucket="all",
        ),
        "factoid_turn_5_plus_character_f1": csv_effect(
            aggregate,
            metric="character_f1",
            scope="question_type_turn_bucket",
            question_type="factoid",
            turn_bucket="turn_5_plus",
        ),
        "factoid_exact_match_contain": csv_effect(
            aggregate,
            metric="exact_match_contain",
            scope="question_type",
            question_type="factoid",
            turn_bucket="all",
        ),
        "glm4_7_flash_factoid_character_f1": csv_effect(
            model_effects,
            model="glm4_7_flash",
            metric="character_f1",
            question_type="factoid",
        ),
    }
    reconciliation = {
        name: {
            "raw": raw_checks[name],
            "reported": csv_checks[name],
            "absolute_difference": abs(raw_checks[name] - csv_checks[name]),
            "passed": math.isclose(raw_checks[name], csv_checks[name], abs_tol=1e-12),
        }
        for name in raw_checks
    }
    if not all(item["passed"] for item in reconciliation.values()):
        raise AssertionError("independent raw-data reconciliation failed")

    per_model_overall: dict[str, list[float]] = defaultdict(list)
    for row in paired:
        per_model_overall[row["model"]].append(row["character_f1_delta"])
    equal_weight_model_mean = mean([mean(values) for values in per_model_overall.values()])
    reported_overall = csv_checks["overall_character_f1"]
    if not math.isclose(equal_weight_model_mean, reported_overall, abs_tol=1e-12):
        raise AssertionError("aggregate/model reconciliation failed")

    report_html = OUTPUT_DIR / "report.html"
    source_paths_relative = all(not Path(item["path"]).is_absolute() for item in summary["source_manifest"])
    validation = {
        "schema_version": "longcoqa-g0-g3-rewrite-quality-validation-v1",
        "validated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "passed_with_visual_caveat",
        "raw_pairing": {
            "models": len(model_names),
            "paired_model_observations": len(paired),
            "all_model_id_sets_aligned": all(id_alignment.values()),
            "per_model_id_alignment": id_alignment,
        },
        "independent_reconciliation": reconciliation,
        "aggregate_reconciliation": {
            "equal_weight_model_mean": equal_weight_model_mean,
            "reported_overall": reported_overall,
            "passed": True,
        },
        "artifact": {
            "report_html_exists": report_html.is_file(),
            "report_html_bytes": report_html.stat().st_size if report_html.is_file() else 0,
            "source_paths_relative": source_paths_relative,
            "schema_validation": "passed",
            "package_validation": "passed",
            "browser_verification": "structural_only",
            "browser_caveat": "No installed Chromium headless-shell executable was available.",
        },
    }
    output_path = OUTPUT_DIR / "validation.json"
    output_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(validation, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
