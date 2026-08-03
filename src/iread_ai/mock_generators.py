from __future__ import annotations

import hashlib
import json
from typing import Any

from .generation_models import (
    ContinueStoryRequest,
    EvaluateTrainingRequest,
    EvaluateTrainingResponse,
    GeneratedStoryLine,
    GenerateStoryRequest,
    GenerateStoryResponse,
    GenerateTrainingRequest,
    GenerateTrainingResponse,
    SpeechSynthesisRequest,
    SpeechTranscriptionResponse,
    StoryBranchOption,
    StoryBranchPrompt,
    TrainingCandidateRequest,
    TrainingCandidateResponse,
)

WORDS = ["사과", "나무", "바다", "토끼", "모자"]
SENTENCES = [
    "아기가 사과를 먹는다.",
    "누나가 나무를 본다.",
    "토끼가 들판을 달린다.",
    "아이가 모자를 쓴다.",
    "강아지가 문을 닫는다.",
]
SYLLABLES = ["가", "너", "도", "무", "비"]
FINALS = ["감", "눈", "달", "밤", "집"]

ONSETS = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]
VOWELS = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ",
    "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
]
CODAS = [
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ",
    "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ",
    "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]


def _rotate(values: list[str], index: int) -> list[str]:
    offset = index % len(values)
    return values[offset:] + values[:offset]


def _choice_values(correct: str, pool: list[str], index: int) -> list[str]:
    values = [correct]
    values.extend(value for value in pool if value != correct and value not in values)
    return _rotate(values[:3], index)


def _decompose(syllable: str) -> list[str]:
    offset = ord(syllable[0]) - 0xAC00
    if not 0 <= offset <= 11171:
        raise ValueError(f"완성형 한글 음절이 아닙니다: {syllable}")
    result = [ONSETS[offset // 588], VOWELS[offset % 588 // 28]]
    coda = CODAS[offset % 28]
    if coda:
        result.append(coda)
    return result


def _compose(onset: str, vowel: str, coda: str = "") -> str:
    value = 0xAC00 + ONSETS.index(onset) * 588 + VOWELS.index(vowel) * 28
    value += CODAS.index(coda)
    return chr(value)


def _initial(text: str) -> str:
    return _decompose(text[0])[0]


def _final(text: str) -> str:
    parts = _decompose(text[0])
    return parts[2] if len(parts) == 3 else ""


def _sound_choice(index: int, correct: str, pool: list[str]) -> dict[str, Any]:
    choices = _choice_values(correct, pool, index)
    return {"audioText": correct, "choices": choices, "answerIndex": choices.index(correct)}


def _trace(index: int, kind: str) -> dict[str, Any]:
    if kind == "VOWEL":
        target = ["ㅏ", "ㅓ", "ㅗ", "ㅜ", "ㅣ"][index]
        return {
            "vowelType": "BASIC",
            "target": target,
            "soundText": target,
            "traceAssetKey": f"vowel_{index}",
        }
    if kind == "CONSONANT":
        target = ["ㄱ", "ㄴ", "ㄷ", "ㅁ", "ㅅ"][index]
        return {
            "consonantType": "BASIC",
            "target": target,
            "soundText": target,
            "traceAssetKey": f"consonant_{index}",
        }
    target = SYLLABLES[index]
    return {
        "syllableType": "WITHOUT_FINAL",
        "target": target,
        "soundText": target,
        "traceAssetKey": f"syllable_{index}",
    }


def _candidate(training_type: str, index: int) -> dict[str, Any]:
    if training_type == "VOWEL_TRACE":
        return _trace(index, "VOWEL")
    if training_type == "CONSONANT_TRACE":
        return _trace(index, "CONSONANT")
    if training_type == "SYLLABLE_TRACE":
        return _trace(index, "SYLLABLE")
    if training_type == "CONSONANT_SOUND_CHOICE":
        target = ["ㄱ", "ㄴ", "ㄷ", "ㅁ", "ㅅ"][index]
        return _sound_choice(index, target, ["ㄱ", "ㄴ", "ㄷ", "ㅁ", "ㅅ", "ㅂ", "ㅈ"])
    if training_type == "VOWEL_SOUND_CHOICE":
        target = ["ㅏ", "ㅓ", "ㅗ", "ㅜ", "ㅣ"][index]
        return _sound_choice(index, target, ["ㅏ", "ㅓ", "ㅗ", "ㅜ", "ㅣ", "ㅡ", "ㅐ"])
    if training_type == "CONSONANT_VOWEL_CLASSIFICATION":
        consonant = index % 2 == 0
        audio = ["ㄱ", "ㄴ", "ㄷ"][index // 2] if consonant else ["ㅏ", "ㅓ"][index // 2]
        return {
            "audioText": audio,
            "choices": ["CONSONANT", "VOWEL"],
            "answerIndex": 0 if consonant else 1,
        }
    if training_type in {"SYLLABLE_INITIAL_CHOICE", "WORD_INITIAL_CHOICE"}:
        text = SYLLABLES[index] if training_type.startswith("SYLLABLE") else WORDS[index]
        correct = _initial(text)
        choices = _choice_values(
            correct, ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㅈ", "ㅎ"], index
        )
        return {"audioText": text, "choices": choices, "answerIndex": choices.index(correct)}
    if training_type == "SAME_INITIAL_WORD_CHOICE":
        correct = ["사슴", "나비", "바구니", "토마토", "무지개"][index]
        choices = _rotate([correct, "기차", "연필"], index)
        return {
            "targetType": "WORD",
            "targetAudioText": WORDS[index],
            "choiceType": "WORD",
            "choices": [{"text": value, "imagePrompt": ""} for value in choices],
            "answerIndex": choices.index(correct),
        }
    if training_type in {"FINAL_CONSONANT_CHOICE", "WORD_FINAL_SOUND_CHOICE"}:
        text = FINALS[index] if training_type.startswith("FINAL") else ["곰", "문", "밥", "집", "공"][index]
        correct = _final(text[-1])
        choices = _choice_values(correct, ["ㄱ", "ㄴ", "ㄹ", "ㅁ", "ㅂ", "ㅇ"], index)
        return {"audioText": text, "choices": choices, "answerIndex": choices.index(correct)}
    if training_type == "FINAL_CONSONANT_COMPARISON":
        audio = FINALS[index]
        offset = ord(audio) - 0xAC00
        onset_medial = offset // 28 * 28
        choices = [audio]
        for coda_index in [1, 4, 7, 8, 16, 17, 21]:
            value = chr(0xAC00 + onset_medial + coda_index)
            if value not in choices:
                choices.append(value)
            if len(choices) == 3:
                break
        return {"audioText": audio, "choices": choices, "answerIndex": 0}
    if training_type == "SIMILAR_SOUND_CHOICE":
        plains = ["ㄱ", "ㄷ", "ㅂ", "ㅈ", "ㅅ"]
        aspirated = {"ㄱ": "ㅋ", "ㄷ": "ㅌ", "ㅂ": "ㅍ", "ㅈ": "ㅊ", "ㅅ": "ㅆ"}
        return {
            "soundGroup": "PLAIN_ASPIRATED",
            "audioText": ["가", "다", "바", "자", "사"][index],
            "choices": [plains[index], aspirated[plains[index]]],
            "answerIndex": 0,
        }
    if training_type in {"PHONEME_BLEND", "SYLLABLE_BLEND"}:
        result = SYLLABLES[index] if training_type == "PHONEME_BLEND" else WORDS[index]
        parts = _decompose(result) if training_type == "PHONEME_BLEND" else list(result)
        distractor_pool = ["ㄴ", "ㅓ", "ㅁ", "ㅗ"] if training_type == "PHONEME_BLEND" else ["가", "너", "도", "마"]
        distractor = next(value for value in distractor_pool if value not in parts)
        return {
            "audioParts": parts,
            "cards": parts + [distractor],
            "answerOrder": list(range(len(parts))),
            "result": result,
        }
    if training_type == "BASIC_SYLLABLE_BUILD":
        result = SYLLABLES[index]
        initial, medial = _decompose(result)
        initial_choices = _choice_values(
            initial, ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ"], 0
        )
        medial_choices = _choice_values(
            medial, ["ㅏ", "ㅓ", "ㅗ", "ㅜ", "ㅡ", "ㅣ"], 0
        )
        return {
            "targetAudioText": result,
            "initialChoices": initial_choices,
            "medialChoices": medial_choices,
            "initialAnswerIndex": initial_choices.index(initial),
            "medialAnswerIndex": medial_choices.index(medial),
            "result": result,
        }
    if training_type in {"FINAL_SYLLABLE_BUILD", "DOUBLE_FINAL_BUILD"}:
        result = FINALS[index] if training_type.startswith("FINAL") else ["닭", "삶", "값", "앉", "몫"][index]
        initial, medial, final = _decompose(result)
        initial_choices = _choice_values(
            initial, ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ"], 0
        )
        medial_choices = _choice_values(
            medial, ["ㅏ", "ㅓ", "ㅗ", "ㅜ", "ㅡ", "ㅣ"], 0
        )
        final_choices = _choice_values(
            final, ["ㄱ", "ㄴ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ"], 0
        )
        return {
            "targetAudioText": result,
            "initialChoices": initial_choices,
            "medialChoices": medial_choices,
            "finalChoices": final_choices,
            "initialAnswerIndex": initial_choices.index(initial),
            "medialAnswerIndex": medial_choices.index(medial),
            "finalAnswerIndex": final_choices.index(final),
            "result": result,
        }
    if training_type == "FINAL_CONSONANT_DELETE":
        source = FINALS[index]
        parts = _decompose(source)
        result = _compose(parts[0], parts[1])
        return {
            "source": source,
            "targetAudioText": result,
            "removableUnits": parts,
            "answerIndex": 2,
            "result": result,
        }
    if training_type == "SYLLABLE_DELETE":
        source = WORDS[index]
        delete_index = index % len(source)
        result = "".join(value for position, value in enumerate(source) if position != delete_index)
        return {
            "source": source,
            "targetAudioText": result,
            "syllables": list(source),
            "deleteIndex": delete_index,
            "result": result,
        }
    if training_type == "SYLLABLE_REPLACE":
        source = ["사과", "나비", "바다", "토끼", "모자"][index]
        result = ["나과", "나무", "바보", "코끼", "과자"][index]
        replace_index = [0, 1, 1, 0, 0][index]
        replacement = result[replace_index]
        return {
            "source": source,
            "targetAudioText": result,
            "replaceIndex": replace_index,
            "choices": [replacement, "가", "너"],
            "answerIndex": 0,
            "result": result,
        }
    if training_type == "WORD_READING":
        return {
            "readingOrder": "SEQUENTIAL" if index % 2 == 0 else "FREE",
            "words": _rotate(WORDS, index)[:3],
        }
    if training_type == "NONWORD_READING":
        return {
            "words": [
                {"text": WORDS[index], "isNonword": False},
                {"text": ["사푸", "누바", "버누", "토마기", "모파"][index], "isNonword": True},
            ]
        }
    if training_type == "DIFFICULT_WORD_PREVIEW":
        difficult = ["국물", "먹는다", "닫는다", "협력", "꽃잎"][index]
        sentence = [
            "아기가 시원한 국물을 먹는다.",
            "동생은 아침밥을 먹는다.",
            "아이가 조용히 문을 닫는다.",
            "친구들이 힘을 모아 협력한다.",
            "봄에는 예쁜 꽃잎이 날린다.",
        ][index]
        return {"difficultWords": [{"word": difficult, "syllables": list(difficult)}], "sentence": sentence}
    if training_type == "SENTENCE_READING":
        sentence = SENTENCES[index]
        return {"sentence": sentence, "tokens": sentence[:-1].split(" ")}
    if training_type == "SHORT_PASSAGE_READING":
        second = [
            "아기는 천천히 꼭꼭 씹었다.",
            "나무 위에서 새가 노래했다.",
            "들판의 풀들이 바람에 흔들렸다.",
            "토끼의 친구가 함께 웃었다.",
            "모자가 바람에 살짝 흔들렸다.",
        ][index]
        return {"sentences": [SENTENCES[index], second]}
    if training_type == "SENTENCE_ASSEMBLY":
        sentence = SENTENCES[index]
        ordered = sentence.split(" ")
        cards = ordered[-1:] + ordered[:-1]
        return {
            "cards": cards,
            "answerOrder": [cards.index(value) for value in ordered],
            "completedSentence": sentence,
        }
    if training_type == "FILL_IN_THE_BLANK":
        choices = _rotate(["모자", "기차", "바다", "나비", "사과"], index)[:3]
        answer = choices[0]
        return {
            "sentence": "책상 위에 {{blank}}를 놓았어요.",
            "inputType": "CHOICE",
            "choices": choices,
            "answerIndex": 0,
            "acceptedAnswers": [answer],
            "completedSentence": f"책상 위에 {answer}를 놓았어요.",
        }
    if training_type == "IMAGE_SENTENCE_MATCH":
        sentence = SENTENCES[index]
        return {
            "imagePrompt": sentence[:-1] + " 장면",
            "choices": [sentence, "고양이는 방에 앉아 있다.", "비행기가 힘차게 날아간다."],
            "answerIndex": 0,
        }
    if training_type == "SENTENCE_REPEAT":
        return {"sentence": SENTENCES[index], "emotion": "HAPPY" if index % 2 == 0 else "CALM"}
    if training_type == "WORD_CHAIN_READING":
        return {"words": _rotate(WORDS, index)[:3], "requiredOrder": "SEQUENTIAL"}
    if training_type == "PHRASE_READING":
        sentence = SENTENCES[index]
        tokens = sentence[:-1].split(" ")
        return {"sentence": sentence, "phrases": [tokens[0], " ".join(tokens[1:]) + "."]}
    if training_type == "REPEATED_SENTENCE_READING":
        return {"sentence": SENTENCES[index], "repeatCount": 2 + index % 2}
    if training_type == "SHORT_STORY_READING":
        return {
            "title": f"{WORDS[index]} 이야기",
            "sentences": [
                {
                    "speaker": "NARRATOR",
                    "text": "강아지가 숲길에서 작은 문을 찾았어요.",
                    "emotion": "CALM",
                },
                {
                    "speaker": "CHARACTER",
                    "text": "안에는 무엇이 있을까? 하고 물었어요.",
                    "emotion": "SURPRISED",
                },
                {
                    "speaker": "NARRATOR",
                    "text": "친구와 문을 열자 밝은 별이 나왔어요.",
                    "emotion": "HAPPY",
                },
            ],
        }
    raise ValueError(f"지원하지 않는 trainingType입니다: {training_type}")


def _training_offset(request: TrainingCandidateRequest) -> int:
    policy = {
        "difficulty": request.difficulty,
        "targets": sorted(
            (item.featureCode, item.weaknessScore, item.confidence)
            for item in request.targetFeatures
        ),
        "excluded": sorted(request.excludedFeatures),
        "prompt": request.additionalPrompt,
    }
    digest = hashlib.sha256(
        json.dumps(policy, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).digest()
    return (digest[0] + request.difficulty - 1) % 5


def generate_training_candidates(request: TrainingCandidateRequest) -> TrainingCandidateResponse:
    offset = _training_offset(request)
    data = [
        _candidate(request.trainingType, (offset + index) % 5)
        for index in range(request.count)
    ]
    return TrainingCandidateResponse(type=request.trainingType, data=data)


def generate_legacy_training(request: GenerateTrainingRequest) -> GenerateTrainingResponse:
    spec = request.inputData.get("generationSpec") or {}
    question_count = max(1, int(spec.get("questionCount", 5)))
    expected_words = request.inputData.get("expectedWords") or WORDS
    questions: list[dict[str, Any]] = []
    for index in range(question_count):
        item = expected_words[index % len(expected_words)]
        word = item if isinstance(item, str) else item.get("wordName", WORDS[index % len(WORDS)])
        questions.append(
            {
                "questionId": f"q-{index + 1:03d}",
                "sequence": index + 1,
                "problem": {"instruction": f"{word}를 소리 내어 읽어 보세요.", "targetText": word},
                "answer": {"canonicalText": word, "correctText": word},
            }
        )
    return GenerateTrainingResponse(
        requestId=request.requestId,
        schemaVersion=request.schemaVersion,
        generatedData={
            "version": request.schemaVersion,
            "questionType": spec.get("questionType", "MOCK_READING"),
            "questions": questions,
        },
    )


STORY_SCRIPTS: dict[str, dict[str, list[str]]] = {
    "토끼와 거북이": {
        "start": [
            "어느 숲 속, 빠른 토끼와 느린 거북이가 살고 있었어요.",
            "어느 날 토끼가 거북이를 놀리며 달리기 시합을 하자고 했어요.",
            "거북이는 흔쾌히 시합을 시작했고, 토끼는 훌쩍 앞서나갔어요.",
            "쉬운 시합이라 자만한 토끼는 길가에 주저앉아 잠을 잤어요.",
            "잠든 토끼를 깨울까요, 아니면 가만히 둘까요?",
        ],
        "continue": [
            "토끼는 잠에서 늦게 깨어 거북이를 뒤쫓았어요.",
            "하지만 거북이는 이미 결승선 가까이 다가가 있었지요.",
            "거북이는 한 걸음 한 걸음 멈추지 않고 걸었어요.",
            "마침내 거북이가 먼저 결승선을 통과했어요.",
            "토끼는 부끄러워 고개를 들지 못했답니다.",
        ],
    },
    "개미와 배짱이": {
        "start": [
            "따뜻한 여름, 부지런한 개미와 노래하는 배짱이가 살고 있었어요.",
            "개미는 매일 먹이를 모으느라 바빴어요.",
            "배짱이는 노래만 부르며 놀고 먹었지요.",
            "가을이 깊어지고 추운 겨울이 다가오고 있었어요.",
            "배짱이는 개미에게 먹이를 나눠달라고 할까요, 혼자 견뎌야 할까요?",
        ],
        "continue": [
            "배짱이는 찬 바람을 맞으며 개미네 집을 찾아갔어요.",
            '"개미야, 추운 겨울을 날 먹이를 조금만 나눠줄 수 없을까?"',
            "개미는 마음을 다잡며 지난여름을 떠올렸어요.",
            "개미는 배짱이에게 다음부터는 함께 일하는 법을 알려주었어요.",
            "다음해 여름, 배짱이도 부지런히 먹이를 모았답니다.",
        ],
    },
    "노인과 바다": {
        "start": [
            "바닷가 마을에 바다를 사랑하는 늙은 어부가 있었어요.",
            "어부는 여든네 날 동안 물고기를 한 마리도 잡지 못했어요.",
            "어느 날, 어부는 아주 먼 바다로 나아갔어요.",
            "큰 낚싯바늘에 자신보다 큰 물고기가 걸려들었지요.",
            "거대한 물고기를 그냥 놓아줄까요, 끝까지 잡아볼까요?",
        ],
        "continue": [
            "어부는 작은 배에 매달린 물고기와 며칠을 보냈어요.",
            "물고기는 마침내 지쳐 어부에게 길을 열어주었어요.",
            "돌아오는 길, 상어 떼가 피 냄새를 맡고 따라왔어요.",
            "어부는 노를 단단히 쥐고 끝까지 포기하지 않았어요.",
            "비록 남은 건 뼈뿐이었지만, 어부의 마음은 단단했답니다.",
        ],
    },
    "신데렐라": {
        "start": [
            "옛날에 착하고 예쁜 신데렐라라는 소녀가 살고 있었어요.",
            "계모와 언니들은 신데렐라에게 잔심부름만 시켰어요.",
            "어느 날, 왕자님의 무도회 초대장이 왔어요.",
            "신데렐라는 혼자 남아 슬퍼하고 있었지요.",
            "슬픈 신데렐라를 도와줄까요, 아니면 그냥 둘까요?",
        ],
        "continue": [
            "그때 반짝, 요정 할머니가 나타나셨어요.",
            "호박은 마차로, 쥐는 말로, 누더기는 예쁜 옷으로 변했어요.",
            '"밤 열두 시가 되면 꼭 돌아와라." 할머니가 당부하셨지요.',
            "무도회에서 왕자님과 춤을 추는 사이 시계가 열두 시를 쳤어요.",
            "떨어진 유리구두 한 짝이 왕자님을 신데렐라에게 이끌었답니다.",
        ],
    },
    "별주부전": {
        "start": [
            "깊은 바다 용궁에 용왕님이 크게 앓아누웠어요.",
            "약은 토끼의 간뿐이라는 말에 별주부가 육지로 갔어요.",
            "별주부는 꾀로 토끼를 꼬드겨 용궁으로 데려갔어요.",
            "토끼는 위험을 눈치채고 마음을 다잡았어요.",
            "토끼는 속아 간을 내주어야 할까요, 꾀로 빠져나갈까요?",
        ],
        "continue": [
            '토끼는 웃으며 말했어요. "내 간은 육지 바위 틈에 두고 왔단다."',
            "별주부는 하는 수 없이 토끼를 다시 육지로 데려갔어요.",
            "육지에 오자 토끼는 별주부를 멋지게 놀려 주었지요.",
            "별주부는 부끄럽고 미안해 빈 손으로 돌아갔어요.",
            "그 뒤로 토끼의 꾀와 별주부의 뜻밖 수고가 전해졌답니다.",
        ],
    },
    "아기돼지 삼형제": {
        "start": [
            "숲 속에 아기돼지 삼형제가 살고 있었어요.",
            "첫째는 짚으로, 둘째는 나무로 집을 지었어요.",
            "셋째는 부지런히 벽돌로 튼튼한 집을 지었지요.",
            "어느 날 무서운 늑대가 나타나 크게 불었어요.",
            "짚집과 나무집이 흔들려요. 벽돌집으로 달아날까요, 그 자리를 지킬까요?",
        ],
        "continue": [
            "두 형제는 숨이 턱에 차 벽돌집으로 달려갔어요.",
            "늑대는 벽돌집도 불어 넘기려고 안간힘을 썼어요.",
            "하지만 튼튼한 벽돌집은 꿈쩍도 하지 않았어요.",
            "막내돼지의 꾀로 늑대는 굴뚝으로 털썩 떨어졌지요.",
            "삼형제는 다시는 부지런함을 잊지 않았답니다.",
        ],
    },
}


def _story_branch_prompt() -> StoryBranchPrompt:
    return StoryBranchPrompt(
        subtitle="별빛 숲의 갈림길",
        options=[
            StoryBranchOption(optionNo=1, label="반짝이는 별빛 길로 간다"),
            StoryBranchOption(optionNo=2, label="작은 친구가 가리킨 숲길로 간다"),
            StoryBranchOption(optionNo=3, label="맑은 시냇물 길을 따라간다"),
        ]
    )


def _story_script(request: GenerateStoryRequest, key: str) -> list[str]:
    """title로 동화 스크립트를 찾는다. 없으면 generic 대사로 폴백."""
    script = STORY_SCRIPTS.get(request.storyTemplate.title)
    if script:
        return script[key]
    title = request.storyTemplate.title
    if key == "start":
        return [
            f"{title}의 문이 천천히 열렸어요.",
            "주인공은 반짝이는 길을 따라 조심조심 걸었어요.",
            "길 끝에서 도움이 필요한 작은 친구를 만났어요.",
            "두 친구는 힘을 합쳐 숨겨진 지도를 찾았어요.",
            "이제 어느 길로 가면 좋을지 말해 볼까요?",
        ]
    return [
        "주인공은 용기를 내어 새로운 길을 골랐어요.",
        "작은 친구는 기쁘게 고개를 끄덕이며 앞장섰어요.",
        "선택한 길 끝에서 빛나는 열쇠를 발견했어요.",
        "열쇠로 문을 열자 잃어버린 보물이 모습을 드러냈어요.",
        "모두 함께 기뻐하며 멋진 모험을 마치고 돌아왔어요.",
    ]


def generate_story(request: GenerateStoryRequest) -> GenerateStoryResponse:
    start_script = _story_script(request, "start")
    contents = [*start_script[:3], start_script[-1]]
    return GenerateStoryResponse(
        requestId=request.requestId,
        schemaVersion=request.schemaVersion,
        nextProgress=4,
        completed=False,
        lines=[
            GeneratedStoryLine(
                content=content,
                requiresBranchInput=index == 3,
                branchPrompt=_story_branch_prompt() if index == 3 else None,
            )
            for index, content in enumerate(contents)
        ],
    )


def _clean_branch_intent(intent: str) -> str:
    cleaned = intent.strip().strip('"\'“”‘’').rstrip(".?!。？！ ")
    return cleaned or "친구들이 함께 힘을 모으는 것"


def _continuation_segment(request: ContinueStoryRequest) -> tuple[list[str], bool]:
    page_count = len(request.history)
    page_in_day = page_count % 10
    day = page_count // 10 + 1
    title = request.storyTemplate.title
    intent = _clean_branch_intent(request.branchIntent)

    if page_in_day == 0:
        return ([
            f"{title}의 {day}일차 아침이 밝았어요.",
            "어제의 선택을 기억한 친구들이 다시 길을 나섰어요.",
            "길 앞에는 생각하지 못한 새로운 일이 기다리고 있었어요.",
            "친구들은 어떻게 하면 좋을지 샛별이의 생각을 기다렸어요.",
        ], True)

    if page_in_day == 4:
        return ([
            f"샛별이는 ‘{intent}’라고 정했고, 이야기에서도 그 선택이 그대로 이루어졌어요.",
            "친구들은 그 선택을 믿고 힘차게 앞으로 나아갔어요.",
            "선택한 길에서 새로운 친구와 중요한 단서를 만났어요.",
            "조금 전의 선택은 이야기 속 사건과 결과를 바꾸었어요.",
            "이제 다음에는 어떤 일이 일어나면 좋을지 말해 볼까요?",
        ], True)

    if page_in_day == 9:
        if page_count == 99:
            return ([
                f"마지막에도 ‘{intent}’ 선택이 이루어지며 모두가 기쁜 결말을 맞았어요.",
            ], False)
        return ([
            f"‘{intent}’ 선택이 이루어지며 오늘의 모험이 즐겁게 마무리되었어요.",
        ], False)

    raise ValueError(
        f"story continuation must start after page 4, 9, or 10 (received {page_count})"
    )


def continue_story(request: ContinueStoryRequest) -> GenerateStoryResponse:
    contents, branch_on_last_line = _continuation_segment(request)
    next_progress = min(len(request.history) + len(contents), 100)
    return GenerateStoryResponse(
        requestId=request.requestId,
        schemaVersion=request.schemaVersion,
        nextProgress=next_progress,
        completed=next_progress == 100,
        lines=[
            GeneratedStoryLine(
                content=content,
                requiresBranchInput=branch_on_last_line and index == len(contents) - 1,
                branchPrompt=(
                    _story_branch_prompt()
                    if branch_on_last_line and index == len(contents) - 1
                    else None
                ),
            )
            for index, content in enumerate(contents)
        ],
    )


def evaluate_training(request: EvaluateTrainingRequest) -> EvaluateTrainingResponse:
    """결정적 평가: result.questions의 정답 비율로 accuracy(0~100)를 계산한다.

    questions 구조를 알 수 없거나 비어 있으면 데모 통과를 가정해 100.0을 반환한다.
    실제 AI 평가 모델 연동은 별도 후속 작업에서 replaces 이 결정적 mock을 대체한다.
    """
    questions = request.result.get("questions") or []
    if questions:
        correct = sum(
            1
            for item in questions
            if isinstance(item, dict)
            and (item.get("isCorrect") or item.get("correctionConfirmed"))
        )
        accuracy = round(correct / len(questions) * 100, 2)
    else:
        accuracy = 100.0
    return EvaluateTrainingResponse(
        requestId=request.requestId,
        schemaVersion=request.schemaVersion,
        accuracy=accuracy,
    )


# 결정적 STT/TTS mock. 실제 음성 인식·합성은 Azure Speech 연동(P3-F)에서 대체한다.
_TRANSCRIBE_FALLBACK = "친구를 따라간다"
# 재생 불가 자리표시자 오디오(ID3 스텁). 백엔드 MockSpeechProcessor와 동등하며,
# 실제 재생 가능 오디오는 Azure TTS(P3-F)에서 제공한다.
_SILENT_AUDIO_PLACEHOLDER = b"ID3\x03\x00\x00\x00\x00\x00\x00"


def transcribe_speech_mock(
    request_id: str, expected_text: str | None
) -> SpeechTranscriptionResponse:
    transcript = (
        expected_text.strip()
        if expected_text and expected_text.strip()
        else _TRANSCRIBE_FALLBACK
    )
    duration_ms = max(300, len(transcript) * 250)
    return SpeechTranscriptionResponse(
        requestId=request_id,
        transcript=transcript,
        confidence=1.0,
        durationMs=duration_ms,
    )


def synthesize_speech_mock(request: SpeechSynthesisRequest) -> tuple[bytes, int]:
    duration_ms = max(400, len(request.text) * 250)
    return _SILENT_AUDIO_PLACEHOLDER, duration_ms
