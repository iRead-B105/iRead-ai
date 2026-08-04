from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from threading import Lock

from .contracts import (
    LexiconItem,
    LexiconPaletteRequest,
    LexiconPaletteResponse,
    LexiconStatusResponse,
)
from .features import canonical_feature_code, feature_matches
from .repository import LexiconRepository


class LexiconUnavailableError(RuntimeError):
    pass


class LexiconPaletteService:
    def __init__(self, database_path: Path | str) -> None:
        self.database_path = Path(database_path)
        self._repository: LexiconRepository | None = None
        self._load_error: str | None = None
        self._cache: OrderedDict[str, LexiconPaletteResponse] = OrderedDict()
        self._cache_lock = Lock()
        try:
            self._repository = LexiconRepository(self.database_path)
        except (FileNotFoundError, RuntimeError, OSError) as exception:
            self._load_error = str(exception)

    def close(self) -> None:
        if self._repository is not None:
            self._repository.close()

    def status(self) -> LexiconStatusResponse:
        if self._repository is None:
            return LexiconStatusResponse(
                status="UNAVAILABLE",
                databasePath=str(self.database_path),
                reason=self._load_error or "lexicon database is unavailable",
            )
        metadata = self._repository.metadata()
        metrics = self._repository.metrics()
        return LexiconStatusResponse(
            status="READY",
            databasePath=str(self._repository.path),
            databaseVersion=metadata.get("source_version"),
            analyzerVersion=metadata.get("analyzer_version"),
            lexemeCount=metrics["lexemes"],
            formCount=metrics["forms"],
            featureCount=metrics["features"],
            reviewRequiredCount=metrics["review"],
        )

    def build_palette(self, request: LexiconPaletteRequest) -> LexiconPaletteResponse:
        if self._repository is None:
            raise LexiconUnavailableError(self._load_error or "lexicon database is unavailable")
        cache_key = request.model_dump_json(exclude={"requestId"})
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                return cached.model_copy(update={"requestId": request.requestId})
        targets = tuple(canonical_feature_code(item.featureCode) for item in request.targetFeatures)
        excluded = tuple(canonical_feature_code(code) for code in request.excludedFeatures)
        mastered = tuple(canonical_feature_code(code) for code in request.masteredFeatures)
        target_weights = {
            canonical_feature_code(item.featureCode): item.weaknessScore * item.confidence
            for item in request.targetFeatures
        }
        candidates = self._repository.palette_candidates(
            excluded_features=excluded,
            target_features=targets,
            semantic_tags=tuple(request.semanticTags),
            parts_of_speech=tuple(request.partsOfSpeech),
            min_syllables=request.minSyllables,
            max_syllables=request.maxSyllables,
            max_batchim_ratio=request.maxBatchimRatio,
            strict_pronunciation=request.strictPronunciation,
            require_target=request.requireTarget,
            include_inflections=request.includeInflections,
            limit=request.limit,
        )
        items = [
            self._score_candidate(candidate, target_weights, mastered) for candidate in candidates
        ]
        items.sort(key=lambda item: (-item.score, item.syllableCount, item.surface, item.formId))
        metadata = self._repository.metadata()
        response = LexiconPaletteResponse(
            requestId=request.requestId,
            databaseVersion=metadata.get("source_version", "unknown"),
            analyzerVersion=metadata.get("analyzer_version", "unknown"),
            items=items[: request.limit],
        )
        with self._cache_lock:
            self._cache[cache_key] = response
            self._cache.move_to_end(cache_key)
            while len(self._cache) > 128:
                self._cache.popitem(last=False)
        return response

    @staticmethod
    def _score_candidate(
        candidate: dict[str, object],
        target_weights: dict[str, float],
        mastered: tuple[str, ...],
    ) -> LexiconItem:
        features = dict(candidate["features"])
        score = float(candidate["story_score"]) * 4
        reasons = [f"동화 적합도 {candidate['story_score']}"]
        if candidate["story_tier"] == "CORE":
            score += 12
            reasons.append("핵심 아동 어휘")
        matched_targets: list[str] = []
        for target, weight in target_weights.items():
            if any(feature_matches(actual, target) for actual in features):
                score += 24 * weight
                matched_targets.append(target)
        if matched_targets:
            reasons.append("목표 특징 " + ", ".join(matched_targets))
        unmastered_complexity = sum(
            1
            for actual in features
            if actual.startswith("GRAPHEME.")
            and mastered
            and not any(feature_matches(actual, target) for target in target_weights)
            and not any(feature_matches(actual, known) for known in mastered)
        )
        if unmastered_complexity:
            score -= min(unmastered_complexity * 1.5, 9)
            reasons.append(f"미숙달 글자 특징 {unmastered_complexity}개 감점")
        score -= float(candidate["batchim_ratio"]) * 6
        return LexiconItem(
            formId=int(candidate["form_id"]),
            headword=str(candidate["headword"]),
            surface=str(candidate["written_form"]),
            pronunciation=(
                str(candidate["pronunciation"]) if candidate["pronunciation"] is not None else None
            ),
            partOfSpeech=str(candidate["part_of_speech"]),
            definition=str(candidate["definition"]),
            storyTier=str(candidate["story_tier"]),
            semanticTags=[tag for tag in str(candidate["story_tags"]).split(",") if tag],
            syllableCount=int(candidate["written_syllable_count"]),
            batchimCount=int(candidate["batchim_count"]),
            batchimRatio=float(candidate["batchim_ratio"]),
            pronunciationStatus=str(candidate["pronunciation_status"]),
            features={str(code): int(count) for code, count in features.items()},
            score=round(score, 3),
            reasons=reasons,
        )


__all__ = ["LexiconPaletteService", "LexiconUnavailableError"]
