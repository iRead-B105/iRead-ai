"""Validated GMS generation with deterministic child-safe fallback content."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from .generation_models import (
    TrainingCandidateRequest,
    TrainingCandidateResponse,
)
from .mock_generators import generate_training_candidates as mock_training_candidates
from .providers import GenerationProviderError, GMSTextProvider

_UNSAFE_TERMS = (
    "자살",
    "죽여",
    "살해",
    "피투성이",
    "성관계",
    "마약",
    "담배",
    "술을 마",
)


@dataclass(frozen=True, slots=True)
class ProviderResult:
    value: Any
    provider: str
    fallback: bool


def generate_training(
    request: TrainingCandidateRequest,
    provider: GMSTextProvider | None,
) -> ProviderResult:
    if provider is None:
        return ProviderResult(mock_training_candidates(request), "mock", False)

    def generate() -> TrainingCandidateResponse:
        document = provider.generate_json(
            schema_name="iread_training_candidates",
            schema=TrainingCandidateResponse.model_json_schema(),
            system_prompt=(
                "당신은 한국어 아동 문해 훈련 문항 생성기입니다. 응답은 JSON Schema만 "
                "따르며, 아동에게 안전하고 공포스럽지 않은 표현을 사용합니다. "
                "targetFeatures를 우선 연습하고 excludedFeatures는 포함하지 않습니다. "
                "정답과 선택지는 서로 모순되거나 중복되면 안 됩니다."
            ),
            user_prompt=json.dumps(
                request.model_dump(mode="json"), ensure_ascii=False, sort_keys=True
            ),
        )
        result = TrainingCandidateResponse.model_validate(document)
        if result.type != request.trainingType or len(result.data) != request.count:
            raise ValueError("training response type or count did not match request")
        _validate_output_template(request, result)
        _reject_unsafe(result.model_dump_json())
        return result

    return _with_fallback(
        generate,
        lambda: mock_training_candidates(request),
        provider.model,
    )


def _with_fallback(
    generate: Callable[[], Any],
    fallback: Callable[[], Any],
    provider_name: str,
) -> ProviderResult:
    try:
        return ProviderResult(generate(), provider_name, False)
    except (GenerationProviderError, ValidationError, TypeError, ValueError):
        return ProviderResult(fallback(), "safe-mock", True)


def _validate_output_template(
    request: TrainingCandidateRequest,
    response: TrainingCandidateResponse,
) -> None:
    template_data = request.outputTemplate.get("data")
    if not isinstance(template_data, list) or not template_data:
        return
    example = template_data[0]
    if not isinstance(example, dict):
        return
    required_keys = set(example)
    if required_keys and any(
        not required_keys.issubset(item) for item in response.data
    ):
        raise ValueError("training candidate did not match outputTemplate keys")


def _reject_unsafe(text: str) -> None:
    if any(term in text for term in _UNSAFE_TERMS):
        raise ValueError("generated content failed the child-safety filter")
