from __future__ import annotations

from dataclasses import dataclass

from iread_ai.curriculum_models import ReadingFeatureCategory
from iread_ai.devtools.training_review_catalog import TRAINING_REVIEW_CATALOG

RETIRED_TEMPLATE_IDS = frozenset({6, 14, 24, 26, 32, 34})


@dataclass(frozen=True, slots=True)
class CurriculumTemplateSpec:
    template_id: int
    training_type: str
    name: str
    group: str
    stage: int
    supported_categories: tuple[ReadingFeatureCategory, ...]
    suggested_feature: str
    selectable: bool


def _stage(template_id: int) -> int:
    if template_id <= 3:
        return 1
    if template_id <= 13:
        return 2
    if template_id <= 18:
        return 3
    if template_id <= 20:
        return 4
    if template_id == 21:
        return 5
    if template_id <= 26:
        return 6
    if template_id <= 29:
        return 7
    return 8


def _supported_categories(
    template_id: int, feature_code: str
) -> tuple[ReadingFeatureCategory, ...]:
    primary = feature_code.split(".", 1)[0]
    categories: list[ReadingFeatureCategory] = []
    if primary in {"GRAPHEME", "SYLLABLE", "PHONOLOGY", "WORD", "SENTENCE"}:
        categories.append(primary)  # type: ignore[arg-type]

    stage = _stage(template_id)
    if stage <= 2:
        categories.extend(("GRAPHEME", "SYLLABLE"))
    elif stage <= 4:
        categories.extend(("GRAPHEME", "SYLLABLE", "WORD"))
    elif stage <= 6:
        categories.extend(("SYLLABLE", "PHONOLOGY", "WORD"))
    else:
        categories.extend(("PHONOLOGY", "WORD", "SENTENCE"))
    return tuple(dict.fromkeys(categories))


CURRICULUM_TEMPLATE_CATALOG = tuple(
    CurriculumTemplateSpec(
        template_id=spec.template_id,
        training_type=spec.training_type,
        name=spec.name,
        group=spec.group,
        stage=_stage(spec.template_id),
        supported_categories=_supported_categories(spec.template_id, spec.suggested_feature),
        suggested_feature=spec.suggested_feature,
        selectable=spec.template_id not in RETIRED_TEMPLATE_IDS,
    )
    for spec in TRAINING_REVIEW_CATALOG
)


CATALOG_BY_ID = {spec.template_id: spec for spec in CURRICULUM_TEMPLATE_CATALOG}
