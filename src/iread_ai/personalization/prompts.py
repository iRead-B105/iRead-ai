from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

BASELINE_PROMPT_MODE = "baseline"
PERSONALIZED_PROMPT_MODE = "personalized"

_FEATURE_DESCRIPTIONS = {
    "HAS_BATCHIM": "받침이 있는 한글 음절",
    "HAS_COMPLEX_CODA": "ㄳ, ㄵ, ㄺ, ㅄ처럼 두 자음으로 된 겹받침",
    "HAS_DOUBLE_CODA": "ㄲ 또는 ㅆ 받침",
    "HAS_TENSE_ONSET": "ㄲ, ㄸ, ㅃ, ㅆ, ㅉ으로 시작하는 된소리 음절",
    "HAS_ASPIRATED_ONSET": "ㅋ, ㅌ, ㅍ, ㅊ으로 시작하는 거센소리 음절",
    "HAS_COMPOUND_VOWEL": "ㅘ, ㅙ, ㅚ, ㅝ, ㅞ, ㅟ, ㅢ가 든 음절",
    "HAS_GLIDE_VOWEL": "ㅑ, ㅒ, ㅕ, ㅖ, ㅛ, ㅠ가 든 음절",
    "PHONO_LIAISON": "받침 뒤에 모음이 와서 소리가 이어지는 연음",
    "PHONO_NASALIZATION": "받침이 ㄴ, ㅁ, ㅇ 계열로 바뀌는 비음화",
    "PHONO_PALATALIZATION": "ㄷ·ㅌ이 ㅣ 계열 앞에서 ㅈ·ㅊ으로 바뀌는 구개음화",
    "PHONO_TENSIFICATION": "뒤 자음이 된소리로 바뀌는 된소리되기",
    "PHONO_LIQUIDIZATION": "ㄴ과 ㄹ이 만나 ㄹㄹ로 바뀌는 유음화",
    "PHONO_ASPIRATION": "ㅎ의 영향으로 거센소리가 나는 격음화",
    "PHONO_H_DELETION": "ㅎ 소리가 발음에서 사라지는 현상",
    "PHONO_CLUSTER_SIMPLIFICATION": "겹받침에서 한 자음만 나는 단순화",
    "PHONO_CODA_NEUTRALIZATION": "받침이 대표 받침 소리로 바뀌는 규칙",
}

DEFAULT_REPAIR_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "page_repair.md"
)


def load_repair_prompt(path: Path | None = None) -> str:
    return _load_prompt(path or DEFAULT_REPAIR_PROMPT_PATH, "repair")


def _load_prompt(prompt_path: Path, prompt_name: str) -> str:
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"{prompt_name} prompt must not be empty")
    if len(prompt.encode("utf-8")) > 64 * 1024:
        raise ValueError(
            f"{prompt_name} prompt must not exceed 64 KiB"
        )
    return prompt


def build_repair_user_prompt(
    *,
    context: Any,
    profile: Any,
    source_candidate: Any,
    repair_plan: Mapping[str, Any],
) -> str:
    profile_document = _jsonable(profile)
    if not isinstance(profile_document, Mapping):
        raise TypeError(
            "generation profile must serialize to an object"
        )
    source_document = _jsonable(source_candidate)
    if not isinstance(source_document, Mapping):
        raise TypeError("source candidate must serialize to an object")
    normalized_source = {
        "candidate_id": source_document.get(
            "candidate_id",
            source_document.get("candidateId"),
        ),
        "sentences": _jsonable(
            source_document.get("sentences", [])
        ),
    }
    document = {
        "task": "repair_story_page",
        "source_candidate": normalized_source,
        "page_locks": _context_document(context),
        "content_contract": _jsonable(
            profile_document.get(
                "content_contract",
                profile_document.get("contentContract", {}),
            )
        ),
        "reading_policy_hints": _policy_hints(profile_document),
        "repair_plan": _jsonable(repair_plan),
    }
    return json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_reading_policy_hints(
    profile: Any,
) -> dict[str, list[dict[str, Any]]]:
    return _policy_hints(_jsonable(profile))


def _context_document(context: Any) -> dict[str, Any]:
    if hasattr(context, "to_dict"):
        raw = context.to_dict()
    else:
        raw = _jsonable(context)
    if not isinstance(raw, Mapping):
        raise TypeError(
            "page generation context must serialize to an object"
        )
    return {
        str(key): _jsonable(value)
        for key, value in raw.items()
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple | list | set | frozenset):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json", by_alias=True))
    if is_dataclass(value):
        return _jsonable(asdict(value))
    raise TypeError(
        f"value is not JSON serializable: {type(value).__name__}"
    )


def _policy_hints(
    profile_document: Any,
) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(profile_document, Mapping):
        raise TypeError(
            "generation profile must serialize to an object"
        )
    raw_skills = profile_document.get("skills", [])
    if not isinstance(raw_skills, list):
        raw_skills = (
            list(raw_skills)
            if isinstance(raw_skills, tuple)
            else []
        )

    hints: dict[str, list[dict[str, Any]]] = {
        "excluded": [],
        "limited": [],
        "target": [],
    }
    role_to_key = {
        "EXCLUDED": "excluded",
        "LIMITED": "limited",
        "TARGET": "target",
    }
    for raw_skill in raw_skills:
        if not isinstance(raw_skill, Mapping):
            continue
        role = str(raw_skill.get("role", "")).upper()
        key = role_to_key.get(role)
        if key is None:
            continue
        code = str(raw_skill.get("code", ""))
        hints[key].append(
            {
                "code": code,
                "description": _describe_feature(code),
                "min_occurrences": int(
                    _first_non_none(
                        raw_skill,
                        "min_occurrences",
                        "minOccurrences",
                        "target_min",
                        "targetMin",
                    )
                    or 0
                ),
                "max_occurrences": int(
                    _first_non_none(
                        raw_skill,
                        "max_occurrences",
                        "maxOccurrences",
                        "target_max",
                        "targetMax",
                    )
                    or 0
                ),
            }
        )
    return hints


def _first_non_none(
    document: Mapping[str, Any],
    *keys: str,
) -> Any:
    for key in keys:
        value = document.get(key)
        if value is not None:
            return value
    return 0


def _describe_feature(code: str) -> str:
    if code.startswith("ONSET_"):
        return f"초성이 {code.removeprefix('ONSET_')}인 한글 음절"
    if code.startswith("NUCLEUS_"):
        return f"중성이 {code.removeprefix('NUCLEUS_')}인 한글 음절"
    if code.startswith("CODA_"):
        coda = code.removeprefix("CODA_")
        if coda == "ㅆ":
            return "받침이 ㅆ인 음절(예: 했, 있, 었)"
        return f"받침이 {coda}인 한글 음절"
    return _FEATURE_DESCRIPTIONS.get(code, code)


__all__ = [
    "BASELINE_PROMPT_MODE",
    "DEFAULT_REPAIR_PROMPT_PATH",
    "PERSONALIZED_PROMPT_MODE",
    "build_reading_policy_hints",
    "build_repair_user_prompt",
    "load_repair_prompt",
]
