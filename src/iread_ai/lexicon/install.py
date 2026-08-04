from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from .repository import LexiconRepository


def install_database(source: Path, destination: Path) -> dict[str, int | str]:
    source_repository = LexiconRepository(source)
    try:
        source_metadata = source_repository.metadata()
        source_metrics = source_repository.metrics()
    finally:
        source_repository.close()

    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        with closing(sqlite3.connect(temporary)) as connection:
            integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise RuntimeError(f"lexicon database integrity check failed: {integrity}")
        installed_repository = LexiconRepository(temporary)
        try:
            installed_metrics = installed_repository.metrics()
        finally:
            installed_repository.close()
        if installed_metrics != source_metrics:
            raise RuntimeError("installed lexicon database metrics do not match source")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "destination": str(destination.resolve()),
        "sourceVersion": source_metadata.get("source_version", "unknown"),
        **source_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Install a validated iRead lexicon database")
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("local-output/lexicon/story-lexicon.sqlite3"),
    )
    args = parser.parse_args()
    result = install_database(args.source_db, args.destination)
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
