from __future__ import annotations

import os
import sqlite3
import tempfile
import threading
from collections import defaultdict
from importlib.resources import files
from pathlib import Path

from .models import LearningUnit
from .seed import (
    BANK_VERSION,
    CONFUSION_PAIRS,
    unit_features,
    unit_parts,
    unit_seeds,
)

_INITIALIZE_LOCK = threading.Lock()


def default_database_path() -> Path:
    configured = os.getenv("AI_TRAINING_ITEM_DB_PATH", "").strip()
    if configured:
        return Path(configured)
    return Path(tempfile.gettempdir()) / "iread-ai" / "basic-training-items.sqlite3"


class SQLiteLearningUnitRepository:
    def __init__(self, database_path: Path | str | None = None) -> None:
        self._database_path = Path(database_path or default_database_path())
        self._initialize()

    @property
    def database_path(self) -> Path:
        return self._database_path

    def find_all_active(self) -> tuple[LearningUnit, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, unit_type, surface, spoken_text, pronunciation,
                       onset, vowel, coda, difficulty, familiarity, trace_asset_key
                FROM learning_units
                WHERE is_active = 1
                ORDER BY difficulty, familiarity DESC, id
                """
            ).fetchall()
            features: dict[int, set[str]] = defaultdict(set)
            for row in connection.execute(
                """
                SELECT learning_unit_id, feature_code
                FROM learning_unit_features
                WHERE is_verified = 1
                ORDER BY learning_unit_id, feature_code
                """
            ):
                features[int(row["learning_unit_id"])].add(str(row["feature_code"]))
            confusions: dict[int, list[int]] = defaultdict(list)
            for row in connection.execute(
                """
                SELECT learning_unit_id, confusing_unit_id
                FROM learning_unit_confusions
                ORDER BY learning_unit_id, priority, confusing_unit_id
                """
            ):
                confusions[int(row["learning_unit_id"])].append(int(row["confusing_unit_id"]))
        return tuple(
            LearningUnit(
                id=int(row["id"]),
                unit_type=str(row["unit_type"]),
                surface=str(row["surface"]),
                spoken_text=str(row["spoken_text"]),
                pronunciation=str(row["pronunciation"]),
                onset=row["onset"],
                vowel=row["vowel"],
                coda=row["coda"],
                difficulty=int(row["difficulty"]),
                familiarity=int(row["familiarity"]),
                trace_asset_key=row["trace_asset_key"],
                feature_codes=frozenset(features[int(row["id"])]),
                confusion_ids=tuple(confusions[int(row["id"])]),
            )
            for row in rows
        )

    def counts(self) -> dict[str, int]:
        with self._connect() as connection:
            return {
                "units": int(
                    connection.execute("SELECT COUNT(*) FROM learning_units").fetchone()[0]
                ),
                "features": int(
                    connection.execute("SELECT COUNT(*) FROM learning_unit_features").fetchone()[0]
                ),
                "confusions": int(
                    connection.execute("SELECT COUNT(*) FROM learning_unit_confusions").fetchone()[
                        0
                    ]
                ),
            }

    def _initialize(self) -> None:
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        with _INITIALIZE_LOCK, self._connect() as connection:
            schema = (
                files("iread_ai.training_bank").joinpath("schema.sql").read_text(encoding="utf-8")
            )
            connection.executescript(schema)
            stored_version = connection.execute(
                "SELECT value FROM bank_metadata WHERE key = 'bank_version'"
            ).fetchone()
            if stored_version is not None and stored_version[0] != BANK_VERSION:
                self._replace_seed(connection)
                connection.execute(
                    "UPDATE bank_metadata SET value = ? WHERE key = 'bank_version'",
                    (BANK_VERSION,),
                )
            if stored_version is None:
                self._seed(connection)
                connection.execute(
                    "INSERT INTO bank_metadata(key, value) VALUES ('bank_version', ?)",
                    (BANK_VERSION,),
                )
            connection.commit()

    def _replace_seed(self, connection: sqlite3.Connection) -> None:
        connection.execute("DELETE FROM learning_unit_confusions")
        connection.execute("DELETE FROM learning_unit_features")
        connection.execute("DELETE FROM learning_units")
        self._seed(connection)

    def _seed(self, connection: sqlite3.Connection) -> None:
        unit_ids: dict[tuple[str, str], int] = {}
        for seed in unit_seeds():
            onset, vowel, coda = unit_parts(seed)
            cursor = connection.execute(
                """
                INSERT INTO learning_units(
                    unit_type, surface, spoken_text, pronunciation,
                    onset, vowel, coda, difficulty, familiarity,
                    trace_asset_key, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'CURATED_V1')
                """,
                (
                    seed.unit_type,
                    seed.surface,
                    seed.spoken_text,
                    seed.pronunciation,
                    onset,
                    vowel,
                    coda,
                    seed.difficulty,
                    seed.familiarity,
                    seed.trace_asset_key,
                ),
            )
            unit_id = int(cursor.lastrowid)
            unit_ids[(seed.unit_type, seed.surface)] = unit_id
            connection.executemany(
                """
                INSERT INTO learning_unit_features(
                    learning_unit_id, feature_code, occurrence_count, is_verified
                ) VALUES (?, ?, 1, 1)
                """,
                ((unit_id, feature) for feature in sorted(unit_features(seed))),
            )
        for unit_type, first, second, confusion_type, priority in CONFUSION_PAIRS:
            first_id = unit_ids[(unit_type, first)]
            second_id = unit_ids[(unit_type, second)]
            connection.executemany(
                """
                INSERT INTO learning_unit_confusions(
                    learning_unit_id, confusing_unit_id, confusion_type, priority
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    (first_id, second_id, confusion_type, priority),
                    (second_id, first_id, confusion_type, priority),
                ),
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection


__all__ = ["SQLiteLearningUnitRepository", "default_database_path"]
