from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from iread_ai.devtools.training_comparison_data import (
    ORIGINAL_MOCKS,
    SUPPORTED_PREFIXES,
    TRAINING_TYPES,
)
from iread_ai.personalization.analyzer import KoreanReadingAnalyzer
from iread_ai.training_personalization import select_training_candidate

ROOT_DIR = Path(__file__).resolve().parents[3]
BACKEND_TEMPLATE_PATH = (
    ROOT_DIR.parent
    / "backend"
    / "src"
    / "main"
    / "resources"
    / "training-templates.json"
)
PROFILE_SPECS = {
    2115: {"name": "AI기초형", "difficulty": 1},
    2116: {"name": "AI진행형", "difficulty": 3},
    2117: {"name": "AI숙달형", "difficulty": 5},
}


@dataclass(frozen=True, slots=True)
class FeatureProfile:
    feature_code: str
    weakness_score: float
    confidence: float
    evidence_count: int


def _mysql_rows() -> list[str]:
    sql = """
        SELECT p.student_id, f.feature_code,
               ROUND(COALESCE(p.weakness_score, 0) / 1000.0, 4),
               COALESCE(p.confidence, 0), COALESCE(p.evidence_count, 0)
          FROM student_feature_profiles p
          JOIN reading_features f ON f.id = p.reading_features_id
         WHERE p.student_id IN (2115, 2116, 2117)
         ORDER BY p.student_id, f.feature_code
    """.strip().replace("\n", " ")
    command = (
        'mysql --batch --skip-column-names '
        '--default-character-set=utf8mb4 '
        '--user="$MYSQL_USER" --password="$MYSQL_PASSWORD" '
        '--database="$MYSQL_DATABASE" '
        f"--execute='{sql}'"
    )
    completed = subprocess.run(
        ["docker", "exec", "iread-mysql", "sh", "-lc", command],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return [line for line in completed.stdout.splitlines() if line.strip()]


def load_backend_profiles() -> dict[int, list[FeatureProfile]]:
    profiles: dict[int, list[FeatureProfile]] = defaultdict(list)
    for line in _mysql_rows():
        student_id, code, weakness, confidence, evidence = line.split("\t")
        profiles[int(student_id)].append(
            FeatureProfile(
                feature_code=code,
                weakness_score=float(weakness),
                confidence=float(confidence),
                evidence_count=int(evidence),
            )
        )
    missing = set(PROFILE_SPECS).difference(profiles)
    if missing:
        raise RuntimeError(f"Backend profile rows were missing for students: {sorted(missing)}")
    return dict(profiles)


def load_backend_templates() -> dict[str, dict[str, Any]]:
    payload = json.loads(BACKEND_TEMPLATE_PATH.read_text(encoding="utf-8"))
    return {
        item["prompt"]["trainingType"]: item
        for item in payload.get("templates", [])
    }


def _feature_family(code: str) -> str:
    parts = code.split(".")
    return ".".join(parts[:3]) if len(parts) >= 3 else code


def select_targets(
    profiles: list[FeatureProfile],
    training_type: str,
) -> list[dict[str, Any]]:
    prefixes = SUPPORTED_PREFIXES.get(training_type, ())
    compatible = sorted(
        (
            profile
            for profile in profiles
            if profile.evidence_count > 0
            and any(profile.feature_code.startswith(prefix) for prefix in prefixes)
        ),
        key=lambda profile: (
            profile.weakness_score,
            profile.confidence,
            profile.evidence_count,
        ),
        reverse=True,
    )
    selected: list[FeatureProfile] = []
    families: set[str] = set()
    for profile in compatible:
        family = _feature_family(profile.feature_code)
        if family in families:
            continue
        selected.append(profile)
        families.add(family)
        if len(selected) == 2:
            break
    for profile in compatible:
        if profile not in selected:
            selected.append(profile)
        if len(selected) == 2:
            break
    return [
        {
            "featureCode": profile.feature_code,
            "weaknessScore": profile.weakness_score,
            "confidence": profile.confidence,
            "evidenceCount": profile.evidence_count,
        }
        for profile in selected
    ]


def build_request(
    student_id: int,
    training_type: str,
    profiles: list[FeatureProfile],
    templates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    spec = PROFILE_SPECS[student_id]
    targets = select_targets(profiles, training_type)
    target_text = ", ".join(item["featureCode"] for item in targets) or "없음"
    prompt = templates[training_type]["prompt"]
    return {
        "requestId": (
            f"profile-benchmark-{student_id}-{training_type.lower()}-"
            f"{uuid.uuid4().hex[:8]}"
        ),
        "schemaVersion": 2,
        "trainingType": training_type,
        "count": 5,
        "difficulty": spec["difficulty"],
        "targetFeatures": targets,
        "excludedFeatures": [],
        "additionalPrompt": (
            f"{prompt.get('additionalPrompt', '')}\n\n"
            f"{spec['name']} 아동의 현재 목표({target_text})를 연습하되, "
            "정답 관계가 명확하고 서로 다른 문항 후보를 만드세요."
        ),
        "outputTemplate": prompt["outputTemplate"],
        "useLexicon": True,
    }


def _compact(value: str) -> str:
    return "".join(value.split()).rstrip(".?!。？！")


_ONSETS = tuple("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
_VOWELS = tuple("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
_CODAS = (
    "",
    "ㄱ",
    "ㄲ",
    "ㄳ",
    "ㄴ",
    "ㄵ",
    "ㄶ",
    "ㄷ",
    "ㄹ",
    "ㄺ",
    "ㄻ",
    "ㄼ",
    "ㄽ",
    "ㄾ",
    "ㄿ",
    "ㅀ",
    "ㅁ",
    "ㅂ",
    "ㅄ",
    "ㅅ",
    "ㅆ",
    "ㅇ",
    "ㅈ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
)


def _compose_jamo(parts: list[Any]) -> str | None:
    values = [str(value) for value in parts]
    if len(values) not in {2, 3}:
        return None
    try:
        onset_index = _ONSETS.index(values[0])
        vowel_index = _VOWELS.index(values[1])
        coda_index = _CODAS.index(values[2]) if len(values) == 3 else 0
    except ValueError:
        return None
    return chr(0xAC00 + (onset_index * 21 + vowel_index) * 28 + coda_index)


def answer_relation_issues(training_type: str, item: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    choices = item.get("choices")
    answer_index = item.get("answerIndex")
    if answer_index is not None and isinstance(choices, list):
        if not isinstance(answer_index, int) or not 0 <= answer_index < len(choices):
            issues.append("answerIndex가 choices 범위를 벗어남")

    if training_type in {"PHONEME_BLEND", "SYLLABLE_BLEND"}:
        parts = item.get("audioParts", [])
        order = item.get("answerOrder", [])
        cards = item.get("cards", [])
        result = str(item.get("result", ""))
        joined_parts = "".join(map(str, parts))
        composed_parts = _compose_jamo(parts) if training_type == "PHONEME_BLEND" else None
        if result not in {joined_parts, composed_parts}:
            issues.append("audioParts를 합친 값과 result가 다름")
        try:
            ordered_parts = [cards[index] for index in order]
            joined_order = "".join(str(value) for value in ordered_parts)
            composed_order = (
                _compose_jamo(ordered_parts) if training_type == "PHONEME_BLEND" else None
            )
            if result not in {joined_order, composed_order}:
                issues.append("answerOrder로 조립한 값과 result가 다름")
        except (IndexError, TypeError):
            issues.append("answerOrder가 cards 범위를 벗어남")

    if training_type == "FINAL_CONSONANT_DELETE":
        source = str(item.get("source", ""))
        result = str(item.get("result", ""))
        if not source or not result or source == result:
            issues.append("받침 제거 전후 값이 올바르지 않음")
    elif training_type == "SYLLABLE_DELETE":
        syllables = item.get("syllables", [])
        delete_index = item.get("deleteIndex")
        if isinstance(syllables, list) and isinstance(delete_index, int):
            if 0 <= delete_index < len(syllables):
                actual = "".join(
                    str(value) for index, value in enumerate(syllables) if index != delete_index
                )
                if _compact(actual) != _compact(str(item.get("result", ""))):
                    issues.append("deleteIndex 적용 결과와 result가 다름")
            else:
                issues.append("deleteIndex가 syllables 범위를 벗어남")
    elif training_type == "SYLLABLE_REPLACE":
        source = str(item.get("source", ""))
        replace_index = item.get("replaceIndex")
        if isinstance(replace_index, int) and 0 <= replace_index < len(source):
            selected = None
            if isinstance(choices, list) and isinstance(answer_index, int):
                if 0 <= answer_index < len(choices):
                    selected = str(choices[answer_index])
            if selected is None:
                issues.append("교체 정답을 찾을 수 없음")
            else:
                actual = source[:replace_index] + selected + source[replace_index + 1 :]
                if actual != str(item.get("result", "")):
                    issues.append("교체 결과와 result가 다름")
        else:
            issues.append("replaceIndex가 source 범위를 벗어남")
    elif training_type == "SENTENCE_ASSEMBLY":
        cards = item.get("cards", [])
        order = item.get("answerOrder", [])
        try:
            actual = " ".join(str(cards[index]) for index in order)
            if _compact(actual) != _compact(str(item.get("completedSentence", ""))):
                issues.append("cards 조립 결과와 completedSentence가 다름")
        except (IndexError, TypeError):
            issues.append("answerOrder가 cards 범위를 벗어남")
    elif training_type == "FILL_IN_THE_BLANK":
        sentence = str(item.get("sentence", ""))
        if "{{blank}}" not in sentence:
            issues.append("sentence에 {{blank}}가 없음")
        elif isinstance(choices, list) and isinstance(answer_index, int):
            if 0 <= answer_index < len(choices):
                actual = sentence.replace("{{blank}}", str(choices[answer_index]))
                if _compact(actual) != _compact(str(item.get("completedSentence", ""))):
                    issues.append("빈칸 정답 적용 결과와 completedSentence가 다름")
    return issues


def _mock_fit(
    training_type: str,
    targets: list[dict[str, Any]],
    difficulty: int,
) -> dict[str, Any]:
    mock = ORIGINAL_MOCKS[training_type]
    candidate = {**mock["content"], **mock["answer"]}
    _, evidence = select_training_candidate(
        [candidate],
        target_features=(item["featureCode"] for item in targets),
        excluded_features=(),
        recommended_words=(),
        analyzer=None,
        lexicon_applied=False,
        training_type=training_type,
        difficulty=difficulty,
    )
    fit = evidence.candidates[0].model_dump()
    return {
        "item": candidate,
        "fit": fit,
        "answerRelationIssues": answer_relation_issues(training_type, candidate),
    }


def _distributed_targets(
    training_type: str,
    candidate_index: int,
    targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(targets) <= 1 or training_type in {"WORD_READING", "WORD_CHAIN_READING"}:
        return targets
    return [targets[0 if candidate_index < 3 else 1]]


def _analysis_target_code(code: str) -> str | None:
    phonology_prefixes = {
        "PHONOLOGY.LIAISON": "PHONOLOGY.LIAISON",
        "PHONOLOGY.NASALIZATION": "PHONOLOGY.NASALIZATION",
        "PHONOLOGY.LIQUIDIZATION": "PHONOLOGY.LIQUIDIZATION",
        "PHONOLOGY.PALATALIZATION": "PHONOLOGY.PALATALIZATION",
        "PHONOLOGY.TENSIFICATION": "PHONOLOGY.TENSIFICATION",
        "PHONOLOGY.ASPIRATION": "PHONOLOGY.ASPIRATION",
        "PHONOLOGY.FINAL_NEUTRALIZATION": "PHONO_CODA_NEUTRALIZATION",
    }
    for prefix, analysis_code in phonology_prefixes.items():
        if code == prefix or code.startswith(f"{prefix}."):
            return analysis_code
    aliases = {
        "SYLLABLE.COMPLEX_VOWEL": "GRAPHEME.VOWEL.COMPOUND",
        "SYLLABLE.TENSE_ONSET": "GRAPHEME.ONSET.TENSE",
    }
    if code in aliases:
        return aliases[code]
    if code in {"SENTENCE.FLUENCY", "WORD.PHONOLOGICALLY_CHANGED"}:
        return None
    return code


def _analysis_target_codes(targets: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(
        analysis_code
        for target in targets
        if (analysis_code := _analysis_target_code(target["featureCode"])) is not None
    )


def generate_candidate_set(
    client: httpx.Client,
    endpoint: str,
    api_key: str,
    payload: dict[str, Any],
    analyzer: KoreanReadingAnalyzer,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = client.post(
        endpoint,
        headers={
            "X-API-Key": api_key,
            "Idempotency-Key": payload["requestId"],
            "Content-Type": "application/json",
        },
        json=payload,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    response.raise_for_status()
    body = response.json()
    if body.get("type") != payload["trainingType"] or len(body.get("data", [])) != 5:
        raise ValueError("Candidate API did not return five items of the requested type")
    candidates: list[dict[str, Any]] = []
    for index, candidate in enumerate(body["data"]):
        assigned = _distributed_targets(
            payload["trainingType"], index, payload["targetFeatures"]
        )
        _, evidence = select_training_candidate(
            [candidate],
            target_features=_analysis_target_codes(assigned),
            excluded_features=payload["excludedFeatures"],
            recommended_words=(),
            analyzer=analyzer,
            lexicon_applied=False,
            training_type=payload["trainingType"],
            difficulty=payload["difficulty"],
        )
        candidates.append(
            {
                "candidateIndex": index,
                "assignedTargets": [item["featureCode"] for item in assigned],
                "item": candidate,
                "fit": evidence.candidates[0].model_dump(),
                "answerRelationIssues": answer_relation_issues(
                    payload["trainingType"], candidate
                ),
            }
        )
    unique_count = len(
        {
            json.dumps(candidate["item"], ensure_ascii=False, sort_keys=True)
            for candidate in candidates
        }
    )
    target_pass_count = sum(
        candidate["fit"]["targetLoadStatus"] in {"PASS", "NOT_APPLICABLE"}
        for candidate in candidates
    )
    length_applicable = [
        candidate
        for candidate in candidates
        if candidate["fit"]["lengthStatus"] != "NOT_APPLICABLE"
    ]
    length_pass_count = sum(
        candidate["fit"]["lengthStatus"] == "PASS" for candidate in length_applicable
    )
    answer_relation_pass_count = sum(
        not candidate["answerRelationIssues"] for candidate in candidates
    )
    return {
        "elapsedMs": elapsed_ms,
        "headerProvider": response.headers.get("X-AI-Provider", "unknown"),
        "response": body,
        "candidates": candidates,
        "uniqueCandidateCount": unique_count,
        "targetPassCount": target_pass_count,
        "targetPassRate": target_pass_count / 5,
        "lengthApplicableCount": len(length_applicable),
        "lengthPassCount": length_pass_count,
        "lengthPassRate": (
            length_pass_count / len(length_applicable) if length_applicable else 1.0
        ),
        "answerRelationPassCount": answer_relation_pass_count,
        "answerRelationPassRate": answer_relation_pass_count / 5,
    }


def _status(mock: dict[str, Any], generated: dict[str, Any]) -> str:
    if generated.get("error"):
        return "ERROR"
    if generated["answerRelationPassCount"] < 5 or generated["uniqueCandidateCount"] < 5:
        return "REGRESSION"
    mock_fit = mock["fit"]
    mock_pass = mock_fit["targetLoadStatus"] in {"PASS", "NOT_APPLICABLE"}
    mock_length = mock_fit["lengthStatus"] in {"PASS", "NOT_APPLICABLE"}
    generated_pass = generated["targetPassRate"] >= 0.8
    generated_length = generated["lengthPassRate"] >= 0.8
    if generated_pass and generated_length and (not mock_pass or not mock_length):
        return "IMPROVED"
    if not generated_pass or not generated_length:
        return "PARTIAL"
    return "EQUIVALENT"


def _profile_summary(
    student_id: int,
    profiles: list[FeatureProfile],
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    assessed = [profile for profile in profiles if profile.evidence_count > 0]
    statuses = Counter(item["comparisonStatus"] for item in items)
    generated_items = [item["generated"] for item in items if not item["generated"].get("error")]
    target_candidate_count = sum(item["targetPassCount"] for item in generated_items)
    total_candidate_count = len(generated_items) * 5
    length_applicable = [
        candidate
        for item in generated_items
        for candidate in item["candidates"]
        if candidate["fit"]["lengthStatus"] != "NOT_APPLICABLE"
    ]
    length_passes = sum(
        candidate["fit"]["lengthStatus"] == "PASS" for candidate in length_applicable
    )
    relation_candidate_count = sum(
        item["answerRelationPassCount"] for item in generated_items
    )
    return {
        "studentId": student_id,
        "name": PROFILE_SPECS[student_id]["name"],
        "difficulty": PROFILE_SPECS[student_id]["difficulty"],
        "featureCount": len(profiles),
        "assessedFeatureCount": len(assessed),
        "averageAssessedWeakness": round(
            sum(profile.weakness_score for profile in assessed) / max(len(assessed), 1), 4
        ),
        "topWeaknesses": [
            {
                "featureCode": profile.feature_code,
                "weaknessScore": profile.weakness_score,
            }
            for profile in sorted(
                assessed, key=lambda value: value.weakness_score, reverse=True
            )[:8]
        ],
        "comparisonStatusCounts": dict(statuses),
        "targetPassCount": target_candidate_count,
        "targetCandidateCount": total_candidate_count,
        "targetPassRate": round(
            target_candidate_count / max(total_candidate_count, 1), 4
        ),
        "lengthApplicableCount": len(length_applicable),
        "lengthPassCount": length_passes,
        "lengthPassRate": round(length_passes / max(len(length_applicable), 1), 4),
        "answerRelationPassCount": relation_candidate_count,
        "answerRelationCandidateCount": total_candidate_count,
        "answerRelationPassRate": round(
            relation_candidate_count / max(total_candidate_count, 1), 4
        ),
        "averageGenerationMs": round(
            sum(item.get("generated", {}).get("elapsedMs", 0) for item in items)
            / len(items),
            1,
        ),
    }


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# 아동 프로필별 훈련 목데이터·AI 생성 비교",
        "",
        f"생성 시각: {report['generatedAt']}",
        "",
        "## 프로필별 자동 평가",
        "",
        (
            "| 아동 | 난이도 | 측정 특성 | 목표 적재 통과 | 길이 통과 | "
            "정답 관계 통과 | 개선/동등/부분/회귀/오류 | 평균 생성 |"
        ),
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for summary in report["profileSummaries"]:
        counts = summary["comparisonStatusCounts"]
        lines.append(
            f"| {summary['name']} | {summary['difficulty']} | "
            f"{summary['assessedFeatureCount']}/{summary['featureCount']} | "
            f"{summary['targetPassCount']}/{summary['targetCandidateCount']} | "
            f"{summary['lengthPassCount']}/{summary['lengthApplicableCount']} | "
            f"{summary['answerRelationPassCount']}/"
            f"{summary['answerRelationCandidateCount']} | "
            f"{counts.get('IMPROVED', 0)}/{counts.get('EQUIVALENT', 0)}/"
            f"{counts.get('PARTIAL', 0)}/{counts.get('REGRESSION', 0)}/"
            f"{counts.get('ERROR', 0)} | "
            f"{summary['averageGenerationMs']:.0f} ms |"
        )
    lines.extend(
        [
            "",
            "## 훈련별 결과",
            "",
            "| 번호 | 훈련 | 기초형 | 진행형 | 숙달형 | 프로필별 결과 변화 |",
            "|---:|---|---|---|---|---|",
        ]
    )
    for row in report["trainingSummaries"]:
        lines.append(
            f"| {row['templateId']} | {row['name']} | {row['statuses']['2115']} | "
            f"{row['statuses']['2116']} | {row['statuses']['2117']} | "
            f"{'예' if row['variesAcrossProfiles'] else '아니오'} |"
        )
    lines.extend(
        [
            "",
            "## 판정 해석",
            "",
            "- `IMPROVED`: 목데이터가 놓친 목표 적재 또는 읽기 길이를 생성기가 충족했습니다.",
            "- `EQUIVALENT`: 두 결과가 자동 검증 기준에서 같은 수준입니다.",
            "- `PARTIAL`: 정답 관계는 안전하지만 목표 또는 길이 통과율이 80% 미만입니다.",
            "- `REGRESSION`: 생성 결과의 목표·길이·정답 관계가 목데이터보다 나빠졌습니다.",
            "- `ERROR`: API 생성에 실패했습니다.",
            (
                "- 자연스러움과 교육적 적합성은 자동 점수만으로 확정하지 않고 "
                "상세 JSON의 실제 문항을 함께 검토해야 합니다."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def run(endpoint: str, api_key: str, output_dir: Path) -> tuple[Path, Path, dict[str, Any]]:
    profiles = load_backend_profiles()
    templates = load_backend_templates()
    analyzer = KoreanReadingAnalyzer()
    analyzer.warmup()
    records: dict[str, list[dict[str, Any]]] = {}
    total = len(PROFILE_SPECS) * len(TRAINING_TYPES)
    completed_count = 0
    with httpx.Client(timeout=120) as client:
        for student_id, feature_profiles in profiles.items():
            student_records: list[dict[str, Any]] = []
            for template_id, group, name, training_type in TRAINING_TYPES:
                payload = build_request(
                    student_id, training_type, feature_profiles, templates
                )
                mock = _mock_fit(
                    training_type,
                    payload["targetFeatures"],
                    PROFILE_SPECS[student_id]["difficulty"],
                )
                try:
                    generated = generate_candidate_set(
                        client, endpoint, api_key, payload, analyzer
                    )
                except (httpx.HTTPError, KeyError, ValueError) as exception:
                    generated = {"error": str(exception)}
                status = _status(mock, generated)
                student_records.append(
                    {
                        "templateId": template_id,
                        "group": group,
                        "name": name,
                        "trainingType": training_type,
                        "request": payload,
                        "originalMock": mock,
                        "generated": generated,
                        "comparisonStatus": status,
                    }
                )
                completed_count += 1
                print(
                    f"[{completed_count:03d}/{total}] {PROFILE_SPECS[student_id]['name']} "
                    f"{template_id:02d} {training_type}: {status}",
                    flush=True,
                )
            records[str(student_id)] = student_records

    profile_summaries = [
        _profile_summary(student_id, profiles[student_id], records[str(student_id)])
        for student_id in PROFILE_SPECS
    ]
    training_summaries: list[dict[str, Any]] = []
    for template_id, group, name, training_type in TRAINING_TYPES:
        rows = {
            str(student_id): next(
                item
                for item in records[str(student_id)]
                if item["trainingType"] == training_type
            )
            for student_id in PROFILE_SPECS
        }
        fingerprints = {
            json.dumps(
                row.get("generated", {})
                .get("response", {})
                .get("data", []),
                ensure_ascii=False,
                sort_keys=True,
            )
            for row in rows.values()
        }
        training_summaries.append(
            {
                "templateId": template_id,
                "group": group,
                "name": name,
                "trainingType": training_type,
                "statuses": {
                    student_id: row["comparisonStatus"]
                    for student_id, row in rows.items()
                },
                "variesAcrossProfiles": len(fingerprints) > 1,
            }
        )

    report = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(UTC).isoformat(),
        "endpoint": endpoint,
        "profileSummaries": profile_summaries,
        "trainingSummaries": training_summaries,
        "profiles": {
            str(student_id): {
                **PROFILE_SPECS[student_id],
                "features": [
                    {
                        "featureCode": profile.feature_code,
                        "weaknessScore": profile.weakness_score,
                        "confidence": profile.confidence,
                        "evidenceCount": profile.evidence_count,
                    }
                    for profile in feature_profiles
                ],
                "items": records[str(student_id)],
            }
            for student_id, feature_profiles in profiles.items()
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"training-profile-comparison-{stamp}.json"
    markdown_path = output_dir / f"training-profile-comparison-{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    return json_path, markdown_path, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8081/api/v1/trainings/candidates",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "local-output" / "training-profile-comparison",
    )
    arguments = parser.parse_args()
    load_dotenv(ROOT_DIR / ".env")
    api_key = (
        os.getenv("AI_INTERNAL_API_KEY", "").strip()
        or os.getenv("AI_API_KEY", "").strip()
        or "local-development-key"
    )
    json_path, markdown_path, report = run(
        arguments.endpoint, api_key, arguments.output_dir
    )
    print(json.dumps(report["profileSummaries"], ensure_ascii=False, indent=2))
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
