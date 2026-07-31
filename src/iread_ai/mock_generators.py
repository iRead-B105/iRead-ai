from __future__ import annotations

from typing import Any

from .generation_models import (
    ContinueStoryRequest,
    GeneratedStoryLine,
    GenerateStoryRequest,
    GenerateStoryResponse,
    GenerateTrainingRequest,
    GenerateTrainingResponse,
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
        answer = WORDS[index]
        return {
            "sentence": "책상 위에 {{blank}} 그림이 있다.",
            "inputType": "CHOICE",
            "choices": [answer, "기차", "연필"],
            "answerIndex": 0,
            "acceptedAnswers": [answer],
            "completedSentence": f"책상 위에 {answer} 그림이 있다.",
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
                {"speaker": "NARRATOR", "text": SENTENCES[index], "emotion": "CALM"},
                {"speaker": "CHARACTER", "text": "오늘은 정말 즐거워!", "emotion": "HAPPY"},
            ],
        }
    raise ValueError(f"지원하지 않는 trainingType입니다: {training_type}")


def generate_training_candidates(request: TrainingCandidateRequest) -> TrainingCandidateResponse:
    data = [_candidate(request.trainingType, index % 5) for index in range(request.count)]
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


def generate_story(request: GenerateStoryRequest) -> GenerateStoryResponse:
    title = request.storyTemplate.title
    contents = [
        f"{title}의 문이 천천히 열렸어요.",
        "주인공은 반짝이는 길을 따라 조심조심 걸었어요.",
        "길 끝에서 도움이 필요한 작은 친구를 만났어요.",
        "두 친구는 힘을 합쳐 숨겨진 지도를 찾았어요.",
        "이제 어느 길로 가면 좋을지 말해 볼까요?",
    ]
    return GenerateStoryResponse(
        requestId=request.requestId,
        schemaVersion=request.schemaVersion,
        nextProgress=50,
        completed=False,
        lines=[
            GeneratedStoryLine(content=content, requiresBranchInput=index == 4)
            for index, content in enumerate(contents)
        ],
    )


def continue_story(request: ContinueStoryRequest) -> GenerateStoryResponse:
    intent = request.branchIntent.strip()
    contents = [
        f'주인공은 "{intent}"라고 말하며 새로운 길을 골랐어요.',
        "작은 친구는 기쁘게 고개를 끄덕이며 앞장섰어요.",
        "선택한 길 끝에서 빛나는 열쇠를 발견했어요.",
        "열쇠로 문을 열자 잃어버린 보물이 모습을 드러냈어요.",
        "모두 함께 기뻐하며 멋진 모험을 마치고 돌아왔어요.",
    ]
    return GenerateStoryResponse(
        requestId=request.requestId,
        schemaVersion=request.schemaVersion,
        nextProgress=100,
        completed=True,
        lines=[GeneratedStoryLine(content=content, requiresBranchInput=False) for content in contents],
    )
