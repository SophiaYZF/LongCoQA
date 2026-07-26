import re

from longcoqa.paths import PROJECT_ROOT


def _script(path: str) -> str:
    return (PROJECT_ROOT / path).read_text(encoding="utf-8")


def test_qwen37_p0_preflight_is_secret_safe_and_isolated() -> None:
    script = _script("scripts/run_qwen37_p0_preflight.sh")

    assert "set -Eeuo pipefail" in script
    assert 'ENV_FILE="${LONGCOQA_ENV_FILE:-${PROJECT_ROOT}/.env}"' in script
    for name in (
        "DASHSCOPE_API_KEYS",
        "DASHSCOPE_BASE_URL",
        "DASHSCOPE_MODEL_ID",
    ):
        assert name in script
    assert "runtime/qwen37_p0_preflight" in script
    assert "LONGCOQA_QWEN37_PREFLIGHT_SAMPLES" in script
    assert '[[ "${SAMPLES}" -ge 2 ]]' in script
    assert 'suite_path="${PREFLIGHT_ROOT}/suite_smoke${SAMPLES}.json"' in script
    assert '[[ -e "${PREFLIGHT_ROOT}/.complete" && -s "${suite_path}" ]]' in script
    assert "run-api-suite" in script
    assert 'touch "${PREFLIGHT_ROOT}/.complete"' in script
    assert "outputs/formal/models/qwen37" not in script


def test_qwen37_p0_full_runs_four_protocols_then_evaluates_and_marks() -> None:
    script = _script("scripts/run_qwen37_p0_full.sh")

    assert "set -Eeuo pipefail" in script
    assert "outputs/formal/models/qwen37" in script
    assert "reports/generated/evaluation/automatic/qwen37" in script
    assert "run-api-suite" in script
    assert "--full" in script
    assert "3019" in script
    assert "evaluate-inference" in script
    assert "--bootstrap-iterations 2000" in script
    for protocol in (
        "z0_current",
        "z1_teacher_forced",
        "g0_released_gold",
        "g3_aligned_raw",
    ):
        assert protocol in script
    marker = 'touch "${FORMAL_ROOT}/.complete"'
    assert marker in script
    assert script.index("evaluate-inference") < script.index(marker)


def test_qwen37_scripts_do_not_contain_secret_values() -> None:
    scripts = "\n".join(
        _script(path)
        for path in (
            "scripts/run_qwen37_p0_preflight.sh",
            "scripts/run_qwen37_p0_full.sh",
        )
    )
    for name in (
        "DASHSCOPE_API_KEYS",
        "DASHSCOPE_BASE_URL",
        "DASHSCOPE_MODEL_ID",
    ):
        assert re.search(rf"(?m)^\s*(?:export\s+)?{name}\s*=", scripts) is None
