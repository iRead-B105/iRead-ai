from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from iread_ai.devtools.service_story_catalog import ServiceStoryFixture

READING_PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "fluent": {
        "label": "잘 읽는 아이",
        "description": "회피 규칙 없이 자연스러운 어휘와 생생한 사건을 우선해요.",
        "version": 4,
        "skills": [],
    },
    "balanced": {
        "label": "조금 어려워하는 아이",
        "description": "재미는 유지하면서 된소리, 겹받침, 연음을 조금 줄여요.",
        "version": 1,
        "skills": [
            {
                "code": "HAS_TENSE_ONSET",
                "role": "LIMITED",
                "maxOccurrences": 2,
                "targetMin": None,
                "targetMax": None,
                "unitPenalty": 1.2,
            },
            {
                "code": "PHONO_LIAISON",
                "role": "LIMITED",
                "maxOccurrences": 4,
                "targetMin": None,
                "targetMax": None,
                "unitPenalty": 1.4,
            },
            {
                "code": "HAS_COMPLEX_CODA",
                "role": "LIMITED",
                "maxOccurrences": 1,
                "targetMin": None,
                "targetMax": None,
                "unitPenalty": 1.4,
            },
        ],
    },
    "beginner": {
        "label": "많이 어려워하는 아이",
        "description": "겹받침은 피하고 된소리와 연음을 크게 줄여요.",
        "version": 5,
        "skills": [
            {
                "code": "HAS_COMPLEX_CODA",
                "role": "EXCLUDED",
                "maxOccurrences": 0,
                "targetMin": None,
                "targetMax": None,
                "unitPenalty": 2.0,
            },
            {
                "code": "HAS_TENSE_ONSET",
                "role": "LIMITED",
                "maxOccurrences": 1,
                "targetMin": None,
                "targetMax": None,
                "unitPenalty": 1.8,
            },
            {
                "code": "PHONO_LIAISON",
                "role": "LIMITED",
                "maxOccurrences": 1,
                "targetMin": None,
                "targetMax": None,
                "unitPenalty": 2.0,
            },
            {
                "code": "CODA_ㅆ",
                "role": "LIMITED",
                "maxOccurrences": 1,
                "targetMin": None,
                "targetMax": None,
                "unitPenalty": 1.6,
            },
        ],
    },
}


def build_generation_profile(
    story: ServiceStoryFixture,
    preset_key: str,
) -> dict[str, Any]:
    if preset_key not in READING_PROFILE_PRESETS:
        raise KeyError(f"Unknown reading profile preset: {preset_key}")
    preset = READING_PROFILE_PRESETS[preset_key]
    profile = {
        "schemaVersion": 2,
        "generationProfileVersion": int(preset["version"]),
        "sourceReadingProfileVersion": int(preset["version"]),
        "compilerVersion": "service-simulator-v1",
        "contentContract": {
            "sentenceCount": 4,
            "preferredWrittenSyllables": {"min": 55, "max": 70},
            "acceptedWrittenSyllables": {"min": 50, "max": 75},
            "directDialogueCount": 1,
        },
        "skills": copy.deepcopy(preset["skills"]),
        "protectedTerms": [character.name for character in story.characters],
    }
    canonical = json.dumps(
        profile,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    profile["policyHash"] = f"sha256:{digest}"
    return profile


__all__ = ["READING_PROFILE_PRESETS", "build_generation_profile"]
