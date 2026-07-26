from __future__ import annotations

import importlib
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from longcoqa.evaluation.paper_metric_profiles import (
    MetricCompatibility,
    MetricProfile,
)

LEXICAL_METRICS = {
    "exact_match",
    "exact_match_contain",
    "recall",
    "bleu_2",
    "rouge_l",
}


@dataclass(frozen=True)
class MetricScore:
    profile_id: str
    profile_hash: str
    compatibility: MetricCompatibility
    scores: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores", MappingProxyType(dict(self.scores)))


@dataclass(frozen=True)
class BatchMetricOutput:
    scores: tuple[Mapping[str, float], ...]
    truncated: tuple[bool, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scores",
            tuple(MappingProxyType(dict(score)) for score in self.scores),
        )


BatchBackend = Callable[..., Any]


def _optional_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise RuntimeError(
            "Learned paper metrics require the optional 'paper-metrics' extra: "
            "pip install -e '.[paper-metrics]'"
        ) from error


def _as_floats(values: Any) -> list[float]:
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [float(value) for value in values]


class _LearnedAdapter:
    metric_name = "score"

    def __init__(
        self,
        profile: MetricProfile,
        *,
        backend: BatchBackend | None = None,
        device: str = "cpu",
        batch_size: int = 16,
    ) -> None:
        if not profile.model_id or not profile.model_revision:
            raise ValueError("Learned metric adapters require a locked model and revision")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.profile = profile
        self.device = device
        self.batch_size = batch_size
        self.backend = backend if backend is not None else self._build_backend()

    def _build_backend(self) -> BatchBackend:
        raise NotImplementedError

    def _validate_inputs(
        self, predictions: Sequence[str], references: Sequence[str]
    ) -> None:
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have the same length")

    def _truncated(
        self, predictions: Sequence[str], references: Sequence[str]
    ) -> tuple[bool, ...]:
        max_length = int(self.profile.options.get("max_length", 512))
        return tuple(
            len(str(prediction)) > max_length or len(str(reference)) > max_length
            for prediction, reference in zip(predictions, references, strict=True)
        )

    @staticmethod
    def _validate_output_length(values: Sequence[Any], expected: int) -> None:
        if len(values) != expected:
            raise ValueError(
                f"Learned metric backend returned {len(values)} scores for {expected} inputs"
            )


class BERTScoreAdapter(_LearnedAdapter):
    metric_name = "bertscore_f1"

    def _build_backend(self) -> BatchBackend:
        module = _optional_module("bert_score")
        hub = _optional_module("huggingface_hub")

        def locked_score(
            predictions: Sequence[str], references: Sequence[str], **kwargs: Any
        ) -> Any:
            options = dict(kwargs)
            model_id = str(options.pop("model_type"))
            revision = str(options.pop("model_revision"))
            model_path = hub.snapshot_download(repo_id=model_id, revision=revision)
            return module.score(
                list(predictions),
                list(references),
                model_type=model_path,
                **options,
            )

        return locked_score

    def score_batch(
        self, predictions: Sequence[str], references: Sequence[str]
    ) -> BatchMetricOutput:
        self._validate_inputs(predictions, references)
        precision, recall, f1 = self.backend(
            list(predictions),
            list(references),
            model_type=self.profile.model_id,
            model_revision=self.profile.model_revision,
            device=self.device,
            batch_size=self.batch_size,
            num_layers=int(self.profile.options["num_layers"]),
            verbose=False,
        )
        precision_values = _as_floats(precision)
        recall_values = _as_floats(recall)
        f1_values = _as_floats(f1)
        self._validate_output_length(precision_values, len(predictions))
        self._validate_output_length(recall_values, len(predictions))
        self._validate_output_length(f1_values, len(predictions))
        precision_name = str(
            self.profile.options.get("precision_metric", "bertscore_precision")
        )
        recall_name = str(
            self.profile.options.get("recall_metric", "bertscore_recall")
        )
        f1_name = str(self.profile.options.get("f1_metric", "bertscore_f1"))
        scores = tuple(
            {
                precision_name: precision_value,
                recall_name: recall_value,
                f1_name: f1_value,
            }
            for precision_value, recall_value, f1_value in zip(
                precision_values, recall_values, f1_values, strict=True
            )
        )
        return BatchMetricOutput(scores=scores, truncated=self._truncated(predictions, references))


def _transformers_bart_backend(
    predictions: Sequence[str],
    references: Sequence[str],
    **kwargs: Any,
) -> list[float]:
    torch = _optional_module("torch")
    transformers = _optional_module("transformers")
    model_id = str(kwargs["model_id"])
    revision = str(kwargs["model_revision"])
    device = str(kwargs["device"])
    direction = str(kwargs["direction"])
    max_length = int(kwargs["max_length"])
    batch_size = int(kwargs["batch_size"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_id, revision=revision)
    model.to(device)
    model.eval()
    sources = references if direction == "reference_to_prediction" else predictions
    targets = predictions if direction == "reference_to_prediction" else references
    values: list[float] = []
    for start in range(0, len(sources), batch_size):
        source_batch = list(sources[start : start + batch_size])
        target_batch = list(targets[start : start + batch_size])
        encoded = tokenizer(
            source_batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        target = tokenizer(
            text_target=target_batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )["input_ids"].to(device)
        with torch.inference_mode():
            logits = model(**encoded, labels=target).logits
            log_probs = torch.log_softmax(logits, dim=-1)
            mask = target.ne(tokenizer.pad_token_id)
            token_scores = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            sequence_scores = (token_scores * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        values.extend(_as_floats(sequence_scores))
    return values


class BARTScoreAdapter(_LearnedAdapter):
    metric_name = "bartscore"

    def _build_backend(self) -> BatchBackend:
        _optional_module("torch")
        _optional_module("transformers")
        return _transformers_bart_backend

    def score_batch(
        self, predictions: Sequence[str], references: Sequence[str]
    ) -> BatchMetricOutput:
        self._validate_inputs(predictions, references)
        direction = str(
            self.profile.options.get("direction", "reference_to_prediction")
        )
        if direction not in {"reference_to_prediction", "prediction_to_reference"}:
            raise ValueError(f"Unsupported BARTScore direction: {direction}")
        max_length = int(self.profile.options.get("max_length", 1024))
        values = _as_floats(
            self.backend(
                list(predictions),
                list(references),
                model_id=self.profile.model_id,
                model_revision=self.profile.model_revision,
                device=self.device,
                batch_size=self.batch_size,
                max_length=max_length,
                direction=direction,
            )
        )
        self._validate_output_length(values, len(predictions))
        return BatchMetricOutput(
            scores=tuple({self.metric_name: value} for value in values),
            truncated=self._truncated(predictions, references),
        )


def _transformers_bleurt_backend(
    predictions: Sequence[str],
    references: Sequence[str],
    **kwargs: Any,
) -> list[float]:
    torch = _optional_module("torch")
    bleurt = _optional_module("bleurt_pytorch")
    transformers = _optional_module("transformers")
    transformers.logging.set_verbosity_error()
    model_id = str(kwargs["model_id"])
    revision = str(kwargs["model_revision"])
    device = str(kwargs["device"])
    max_length = int(kwargs["max_length"])
    batch_size = int(kwargs["batch_size"])
    tokenizer = bleurt.BleurtTokenizer.from_pretrained(model_id, revision=revision)
    model = bleurt.BleurtForSequenceClassification.from_pretrained(
        model_id, revision=revision
    )
    model.to(device)
    model.eval()
    values: list[float] = []
    for start in range(0, len(predictions), batch_size):
        prediction_batch = list(predictions[start : start + batch_size])
        reference_batch = list(references[start : start + batch_size])
        encoded = tokenizer(
            reference_batch,
            prediction_batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            logits = model(**encoded).logits.squeeze(-1)
        values.extend(_as_floats(logits))
    return values


class BLEURTAdapter(_LearnedAdapter):
    metric_name = "bleurt"

    def _build_backend(self) -> BatchBackend:
        _optional_module("torch")
        _optional_module("bleurt_pytorch")
        return _transformers_bleurt_backend

    def score_batch(
        self, predictions: Sequence[str], references: Sequence[str]
    ) -> BatchMetricOutput:
        self._validate_inputs(predictions, references)
        max_length = int(self.profile.options.get("max_length", 512))
        values = _as_floats(
            self.backend(
                list(predictions),
                list(references),
                model_id=self.profile.model_id,
                model_revision=self.profile.model_revision,
                device=self.device,
                batch_size=self.batch_size,
                max_length=max_length,
            )
        )
        self._validate_output_length(values, len(predictions))
        return BatchMetricOutput(
            scores=tuple({self.metric_name: value} for value in values),
            truncated=self._truncated(predictions, references),
        )


def build_learned_adapter(
    profile: MetricProfile,
    *,
    device: str = "cpu",
    batch_size: int = 16,
) -> _LearnedAdapter:
    metrics = set(profile.metrics)
    if "bertscore_f1" in metrics:
        return BERTScoreAdapter(profile, device=device, batch_size=batch_size)
    if metrics == {"bartscore"}:
        return BARTScoreAdapter(profile, device=device, batch_size=batch_size)
    if metrics == {"bleurt"}:
        return BLEURTAdapter(profile, device=device, batch_size=batch_size)
    raise ValueError(f"No learned metric adapter for profile {profile.profile_id}")


def _normalize(text: str, profile: MetricProfile) -> str:
    if profile.normalization != "zh_alnum_lower":
        raise ValueError(f"Unsupported normalization: {profile.normalization}")
    if profile.tokenization != "character":
        raise ValueError(f"Unsupported tokenization: {profile.tokenization}")
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", str(text)).lower()


def _recall(prediction: str, reference: str) -> float:
    if not prediction or not reference:
        return float(prediction == reference)
    overlap = sum((Counter(prediction) & Counter(reference)).values())
    return overlap / len(reference)


def _ngrams(text: str, size: int) -> Counter[str]:
    return Counter(text[index : index + size] for index in range(len(text) - size + 1))


def _bleu_2(prediction: str, reference: str, profile: MetricProfile) -> float:
    if not prediction or not reference:
        return float(prediction == reference)
    smoothing = profile.options.get("bleu_smoothing", "add_one")
    if smoothing != "add_one":
        raise ValueError(f"Unsupported BLEU smoothing: {smoothing}")
    precisions: list[float] = []
    for size in (1, 2):
        predicted = _ngrams(prediction, size)
        expected = _ngrams(reference, size)
        overlap = sum((predicted & expected).values())
        total = sum(predicted.values())
        precisions.append((overlap + 1) / (total + 1))
    brevity_penalty = (
        1.0
        if len(prediction) >= len(reference)
        else math.exp(1 - len(reference) / len(prediction))
    )
    return brevity_penalty * math.sqrt(precisions[0] * precisions[1])


def _rouge_l(prediction: str, reference: str) -> float:
    if not prediction or not reference:
        return float(prediction == reference)
    previous = [0] * (len(reference) + 1)
    for predicted_character in prediction:
        current = [0]
        for index, reference_character in enumerate(reference, start=1):
            if predicted_character == reference_character:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    longest = previous[-1]
    precision = longest / len(prediction)
    recall = longest / len(reference)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def paper_lexical_scores(
    prediction: str,
    reference: str,
    profile: MetricProfile,
) -> MetricScore:
    unknown = set(profile.metrics) - LEXICAL_METRICS
    if unknown:
        raise ValueError(f"Unsupported paper metric(s): {', '.join(sorted(unknown))}")
    predicted = _normalize(prediction, profile)
    expected = _normalize(reference, profile)
    available = {
        "exact_match": float(predicted == expected),
        "exact_match_contain": float(bool(expected) and expected in predicted),
        "recall": _recall(predicted, expected),
        "bleu_2": _bleu_2(predicted, expected, profile),
        "rouge_l": _rouge_l(predicted, expected),
    }
    return MetricScore(
        profile_id=profile.profile_id,
        profile_hash=profile.profile_hash,
        compatibility=profile.compatibility,
        scores={metric: available[metric] for metric in profile.metrics},
    )
