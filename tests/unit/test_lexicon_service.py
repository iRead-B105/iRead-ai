from __future__ import annotations

import sqlite3
from pathlib import Path

from iread_ai.lexicon.contracts import LexiconPaletteRequest, LexiconTargetFeature
from iread_ai.lexicon.features import canonical_feature_code, feature_matches
from iread_ai.lexicon.service import LexiconPaletteService


def _create_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            PRAGMA user_version = 1;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE lexemes (
                id INTEGER PRIMARY KEY,
                headword TEXT NOT NULL,
                part_of_speech TEXT NOT NULL,
                definition TEXT NOT NULL,
                story_tier TEXT NOT NULL,
                story_score INTEGER NOT NULL,
                story_tags TEXT NOT NULL
            );
            CREATE TABLE word_forms (
                id INTEGER PRIMARY KEY,
                lexeme_id INTEGER NOT NULL,
                form_kind TEXT NOT NULL,
                written_form TEXT NOT NULL,
                pronunciation TEXT,
                written_syllable_count INTEGER NOT NULL,
                batchim_count INTEGER NOT NULL,
                batchim_ratio REAL NOT NULL,
                pronunciation_status TEXT NOT NULL
            );
            CREATE TABLE form_features (
                form_id INTEGER NOT NULL,
                feature_code TEXT NOT NULL,
                occurrence_count INTEGER NOT NULL,
                confidence REAL NOT NULL DEFAULT 1,
                PRIMARY KEY (form_id, feature_code)
            ) WITHOUT ROWID;
            INSERT INTO metadata VALUES ('source_version', 'test-1');
            INSERT INTO metadata VALUES ('analyzer_version', 'test-analyzer');
            INSERT INTO lexemes VALUES (
                1, '꿀', '명사', '달콤한 먹을거리', 'CORE', 16, 'FOOD,NATURE'
            );
            INSERT INTO lexemes VALUES (2, '꽃', '명사', '식물의 꽃', 'CORE', 16, 'NATURE');
            INSERT INTO lexemes VALUES (
                3, '나비', '명사', '날개가 있는 곤충', 'CORE', 15, 'ANIMAL,NATURE'
            );
            INSERT INTO word_forms VALUES (1, 1, 'LEMMA', '꿀', '꿀', 1, 1, 1, 'NO_CHANGE');
            INSERT INTO word_forms VALUES (2, 2, 'LEMMA', '꽃', '꼳', 1, 1, 1, 'EXPLAINED');
            INSERT INTO word_forms VALUES (3, 3, 'LEMMA', '나비', '나비', 2, 0, 0, 'NO_CHANGE');
            INSERT INTO form_features VALUES (1, 'ONSET_ㄲ', 1, 1);
            INSERT INTO form_features VALUES (1, 'CODA_ㄹ', 1, 1);
            INSERT INTO form_features VALUES (1, 'HAS_BATCHIM', 1, 1);
            INSERT INTO form_features VALUES (2, 'ONSET_ㄲ', 1, 1);
            INSERT INTO form_features VALUES (2, 'CODA_ㅊ', 1, 1);
            INSERT INTO form_features VALUES (2, 'PHONO_LIAISON', 1, .95);
            INSERT INTO form_features VALUES (3, 'ONSET_ㄴ', 1, 1);
            INSERT INTO form_features VALUES (3, 'NO_BATCHIM', 1, 1);
            """
        )


def test_feature_codes_use_project_taxonomy_and_exact_leaf_matching() -> None:
    assert canonical_feature_code("ONSET_ㄲ") == "GRAPHEME.ONSET.TENSE.ㄲ"
    assert feature_matches("GRAPHEME.ONSET.TENSE.ㄲ", "GRAPHEME.ONSET.TENSE")
    assert not feature_matches("GRAPHEME.ONSET.TENSE", "GRAPHEME.ONSET.TENSE.ㄲ")
    assert canonical_feature_code("SYLLABLE.COMPLEX_VOWEL") == "GRAPHEME.VOWEL.COMPOUND"
    assert canonical_feature_code("SYLLABLE.TENSE_ONSET") == "GRAPHEME.ONSET.TENSE"
    assert (
        canonical_feature_code("PHONOLOGY.LIAISON.CODA_TO_SILENT_ONSET")
        == "PHONO_LIAISON"
    )
    assert (
        canonical_feature_code("PHONOLOGY.FINAL_NEUTRALIZATION.TO_ㄷ")
        == "PHONO_CODA_NEUTRALIZATION"
    )


def test_palette_hard_filters_excluded_features_and_requires_target(tmp_path: Path) -> None:
    database = tmp_path / "lexicon.sqlite3"
    _create_database(database)
    service = LexiconPaletteService(database)

    result = service.build_palette(
        LexiconPaletteRequest(
            requestId="palette-1",
            targetFeatures=[
                LexiconTargetFeature(
                    featureCode="GRAPHEME.ONSET.TENSE.ㄲ",
                    weaknessScore=0.8,
                    confidence=0.9,
                )
            ],
            excludedFeatures=["PHONO_LIAISON"],
            requireTarget=True,
            strictPronunciation=True,
            limit=10,
        )
    )

    assert [item.surface for item in result.items] == ["꿀"]
    assert result.items[0].features["GRAPHEME.ONSET.TENSE.ㄲ"] == 1
    assert "PHONO_LIAISON" not in result.items[0].features


def test_status_reports_unavailable_database_without_crashing(tmp_path: Path) -> None:
    service = LexiconPaletteService(tmp_path / "missing.sqlite3")

    status = service.status()

    assert status.status == "UNAVAILABLE"
    assert status.lexemeCount == 0


def test_palette_cache_reuses_the_same_policy_with_a_new_request_id(
    tmp_path: Path,
) -> None:
    database = tmp_path / "lexicon.sqlite3"
    _create_database(database)
    service = LexiconPaletteService(database)
    first_request = LexiconPaletteRequest(
        requestId="palette-first",
        targetFeatures=[LexiconTargetFeature(featureCode="GRAPHEME.ONSET.TENSE.ㄲ")],
        requireTarget=True,
        strictPronunciation=True,
        limit=10,
    )

    first = service.build_palette(first_request)
    second = service.build_palette(first_request.model_copy(update={"requestId": "palette-second"}))

    assert first.requestId == "palette-first"
    assert second.requestId == "palette-second"
    assert [item.surface for item in second.items] == [item.surface for item in first.items]
    assert len(service._cache) == 1


def test_registered_word_validation_reports_only_unknown_content_words(
    tmp_path: Path,
) -> None:
    database = tmp_path / "lexicon.sqlite3"
    _create_database(database)
    service = LexiconPaletteService(database)

    unknown = service.unknown_content_words(["나비 꽃 꿀", "까나를"])

    assert unknown == ["까나"]
