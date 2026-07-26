from __future__ import annotations

from longcoqa.training.capacity import build_trial_specs, choose_trial, classify_status, should_stop


def _trial(batch: int, peak: int, *, status: str = "success", adapter_ok: bool = True) -> dict:
    return {
        "batch_size": batch,
        "gradient_accumulation_steps": {2: 2, 3: 1, 4: 1}[batch],
        "status": status,
        "peak_memory_mib": [peak, peak - 100],
        "adapter_ok": adapter_ok,
        "both_gpus_observed": True,
    }


def test_build_trial_specs_starts_at_two_and_preserves_global_batch_when_possible() -> None:
    specs = build_trial_specs()
    assert [(spec.batch_size, spec.gradient_accumulation_steps) for spec in specs] == [
        (2, 2),
        (3, 1),
        (4, 1),
    ]


def test_choose_trial_rejects_failed_invalid_and_over_limit_trials() -> None:
    trials = [
        _trial(2, 110_000, status="oom"),
        _trial(3, 129_000, adapter_ok=False),
        _trial(4, 134_000),
    ]
    assert choose_trial(trials, total_memory_mib=143_771) is None


def test_choose_trial_selects_stable_trial_closest_to_ninety_percent() -> None:
    trials = [_trial(2, 118_000), _trial(3, 128_900), _trial(4, 131_000)]
    chosen = choose_trial(trials, total_memory_mib=143_771)
    assert chosen is not None
    assert chosen["batch_size"] == 3
    assert chosen["peak_ratio"] == 128_900 / 143_771


def test_should_stop_only_for_target_range_or_hard_failure() -> None:
    assert not should_stop(_trial(2, 110_000), total_memory_mib=143_771)
    assert should_stop(_trial(3, 128_900), total_memory_mib=143_771)
    assert should_stop(_trial(3, 140_000), total_memory_mib=143_771)
    assert should_stop(_trial(3, 100_000, status="oom"), total_memory_mib=143_771)


def test_batch_one_fallback_metadata_is_preserved_when_scan_has_no_candidate() -> None:
    fallback = {
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "status": "baseline_success",
        "peak_memory_mib": [55_921, 55_921],
        "adapter_ok": True,
        "both_gpus_observed": True,
    }
    assert choose_trial([_trial(2, 140_000)], 143_771, fallback=fallback) == {
        **fallback,
        "peak_ratio": 55_921 / 143_771,
        "selection_source": "fallback",
    }


def test_triton_cuda_out_of_memory_is_classified_as_oom() -> None:
    log = "RuntimeError: Triton Error [CUDA]: out of memory"
    assert classify_status(exit_code=1, adapter_ok=False, log_text=log) == "oom"
    assert classify_status(exit_code=0, adapter_ok=True, log_text="") == "success"
