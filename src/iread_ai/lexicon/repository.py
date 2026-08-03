from __future__ import annotations

import sqlite3
import threading
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .features import canonicalize_features, storage_feature_codes

EXPECTED_SCHEMA_VERSION = 1


def _placeholders(values: Iterable[object]) -> str:
    return ",".join("?" for _ in values)


class LexiconRepository:
    def __init__(self, database_path: Path | str) -> None:
        self.path = Path(database_path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._connection = sqlite3.connect(
            f"{self.path.as_uri()}?mode=ro",
            uri=True,
            check_same_thread=False,
            timeout=10,
        )
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
        if version != EXPECTED_SCHEMA_VERSION:
            self.close()
            raise RuntimeError(
                f"unsupported lexicon schema version: {version}; expected {EXPECTED_SCHEMA_VERSION}"
            )
        required = {"metadata", "lexemes", "word_forms", "form_features"}
        tables = {
            str(row[0])
            for row in self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if not required.issubset(tables):
            self.close()
            raise RuntimeError("lexicon database is missing required tables")

    def close(self) -> None:
        self._connection.close()

    def metadata(self) -> dict[str, str]:
        with self._lock:
            rows = self._connection.execute("SELECT key, value FROM metadata").fetchall()
        return {str(row["key"]): str(row["value"]) for row in rows}

    def metrics(self) -> dict[str, int]:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM lexemes) AS lexemes,
                    (SELECT COUNT(*) FROM word_forms) AS forms,
                    (SELECT COUNT(*) FROM form_features) AS features,
                    (
                        SELECT COUNT(*) FROM word_forms
                        WHERE pronunciation_status IN ('PARTIALLY_EXPLAINED', 'UNALIGNED')
                    ) AS review
                """
            ).fetchone()
        return {key: int(row[key]) for key in row.keys()}

    def palette_candidates(
        self,
        *,
        excluded_features: tuple[str, ...],
        target_features: tuple[str, ...],
        semantic_tags: tuple[str, ...],
        parts_of_speech: tuple[str, ...],
        min_syllables: int,
        max_syllables: int,
        max_batchim_ratio: float,
        strict_pronunciation: bool,
        require_target: bool,
        include_inflections: bool,
        limit: int,
    ) -> list[dict[str, Any]]:
        clauses = [
            "l.story_tier IN ('CORE', 'SUPPORT')",
            "f.written_syllable_count BETWEEN ? AND ?",
            "f.batchim_ratio <= ?",
        ]
        params: list[Any] = [min_syllables, max_syllables, max_batchim_ratio]
        if not include_inflections:
            clauses.append("f.form_kind = 'LEMMA'")
        if strict_pronunciation:
            clauses.append("f.pronunciation_status IN ('NO_CHANGE', 'EXPLAINED')")
        if parts_of_speech:
            clauses.append(f"l.part_of_speech IN ({_placeholders(parts_of_speech)})")
            params.extend(parts_of_speech)
        if semantic_tags:
            tag_clauses = ["(',' || l.story_tags || ',') LIKE ?" for _ in semantic_tags]
            clauses.append(f"({' OR '.join(tag_clauses)})")
            params.extend(f"%,{tag},%" for tag in semantic_tags)

        excluded_storage = tuple(
            dict.fromkeys(
                stored for feature in excluded_features for stored in storage_feature_codes(feature)
            )
        )
        if excluded_storage:
            clauses.append(
                f"""
                NOT EXISTS (
                    SELECT 1 FROM form_features ef
                    WHERE ef.form_id = f.id
                      AND ef.feature_code IN ({_placeholders(excluded_storage)})
                )
                """
            )
            params.extend(excluded_storage)

        target_storage = tuple(
            dict.fromkeys(
                stored for feature in target_features for stored in storage_feature_codes(feature)
            )
        )
        if require_target and target_storage:
            clauses.append(
                f"""
                EXISTS (
                    SELECT 1 FROM form_features tf
                    WHERE tf.form_id = f.id
                      AND tf.feature_code IN ({_placeholders(target_storage)})
                )
                """
            )
            params.extend(target_storage)

        candidate_limit = min(max(limit * 12, 100), 1000)
        params.append(candidate_limit)
        sql = f"""
            SELECT
                f.id AS form_id,
                l.headword,
                f.written_form,
                f.pronunciation,
                l.part_of_speech,
                l.definition,
                l.story_tier,
                l.story_score,
                l.story_tags,
                f.written_syllable_count,
                f.batchim_count,
                f.batchim_ratio,
                f.pronunciation_status
            FROM word_forms f
            JOIN lexemes l ON l.id = f.lexeme_id
            WHERE {" AND ".join(clauses)}
            ORDER BY
                CASE l.story_tier WHEN 'CORE' THEN 0 ELSE 1 END,
                l.story_score DESC,
                f.written_syllable_count,
                l.headword,
                f.id
            LIMIT ?
        """
        with self._lock:
            rows = self._connection.execute(sql, params).fetchall()
            form_ids = [int(row["form_id"]) for row in rows]
            features_by_form: defaultdict[int, list[tuple[str, int]]] = defaultdict(list)
            for offset in range(0, len(form_ids), 400):
                batch = form_ids[offset : offset + 400]
                if not batch:
                    continue
                feature_rows = self._connection.execute(
                    f"""
                    SELECT form_id, feature_code, occurrence_count
                    FROM form_features
                    WHERE form_id IN ({_placeholders(batch)})
                    ORDER BY form_id, feature_code
                    """,
                    batch,
                ).fetchall()
                for feature_row in feature_rows:
                    features_by_form[int(feature_row["form_id"])].append(
                        (str(feature_row["feature_code"]), int(feature_row["occurrence_count"]))
                    )

        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["features"] = canonicalize_features(features_by_form.get(int(item["form_id"]), ()))
            result.append(item)
        return result


__all__ = ["EXPECTED_SCHEMA_VERSION", "LexiconRepository"]
