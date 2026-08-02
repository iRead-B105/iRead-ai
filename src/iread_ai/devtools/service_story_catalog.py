from __future__ import annotations

from dataclasses import dataclass

SOURCE_NOTE = (
    "널리 알려진 고전·전래 이야기의 사건 뼈대만 사용하며, "
    "현대 번역의 문장이나 캐릭터 디자인은 사용하지 않습니다."
)


@dataclass(frozen=True, slots=True)
class StoryCharacterFixture:
    character_id: str
    name: str
    role: str
    immutable_traits: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StoryPagePlanFixture:
    locked_event: str
    required_character_ids: tuple[str, ...]
    required_concepts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StoryBeatFixture:
    beat_id: str
    goal: str
    question_focus: str | None
    allowed_branch_slots: tuple[str, ...]
    pages: tuple[
        StoryPagePlanFixture,
        StoryPagePlanFixture,
        StoryPagePlanFixture,
        StoryPagePlanFixture,
    ]
    concluding: bool = False


@dataclass(frozen=True, slots=True)
class ServiceStoryFixture:
    template_id: int
    version: int
    title: str
    source_title: str
    source_note: str
    description: str
    context: str
    initial_summary: str
    characters: tuple[StoryCharacterFixture, ...]
    beats: tuple[StoryBeatFixture, ...]

    @property
    def total_chapters(self) -> int:
        return len(self.beats)


def _character(
    character_id: str,
    name: str,
    role: str,
    *traits: str,
) -> StoryCharacterFixture:
    return StoryCharacterFixture(character_id, name, role, tuple(traits))


def _plan(
    locked_event: str,
    characters: tuple[str, ...],
    *concepts: str,
) -> StoryPagePlanFixture:
    return StoryPagePlanFixture(locked_event, characters, tuple(concepts))


def _beat(
    beat_id: str,
    goal: str,
    question_focus: str | None,
    slots: tuple[str, ...],
    pages: tuple[
        StoryPagePlanFixture,
        StoryPagePlanFixture,
        StoryPagePlanFixture,
        StoryPagePlanFixture,
    ],
    *,
    concluding: bool = False,
) -> StoryBeatFixture:
    return StoryBeatFixture(
        beat_id=beat_id,
        goal=goal,
        question_focus=question_focus,
        allowed_branch_slots=slots,
        pages=pages,
        concluding=concluding,
    )


STORY_CATALOG: tuple[ServiceStoryFixture, ...] = (
    ServiceStoryFixture(
        template_id=1001,
        version=2,
        title="토끼와 거북이",
        source_title="이솝 우화 「토끼와 거북이」",
        source_note=SOURCE_NOTE,
        description="빠른 토끼와 꾸준한 거북이가 숲길 경주를 해요.",
        context=(
            "숲속에서 펼쳐지는 신나는 경주 이야기예요. 토끼의 빠름과 "
            "거북이의 꾸준함을 모두 장점으로 다뤄요. 처음에는 토끼가 거북이의 "
            "느린 걸음을 얕보는 원작 갈등을 분명히 보여 주되, 거친 모욕이나 "
            "부상 없이 경주와 화해의 핵심 사건을 지켜요."
        ),
        initial_summary="이야기는 숲속 경주가 제안되기 전부터 시작해요.",
        characters=(
            _character(
                "hare",
                "토끼",
                "주인공",
                "거북이보다 훨씬 빠르다",
                "처음에는 거북이의 느린 걸음을 얕본다",
                "마지막에는 거북이를 진심으로 인정한다",
            ),
            _character(
                "tortoise",
                "거북이",
                "주인공",
                "천천히 간다",
                "쉽게 포기하지 않는다",
                "상대를 놀리지 않는다",
            ),
        ),
        beats=(
            _beat(
                "race-challenge",
                "토끼가 거북이의 느린 걸음을 얕보며 도발하고, 거북이가 경주로 답해요.",
                "둘이 동시에 외칠 짧은 출발 구호의 내용. 누가 외칠지는 고르지 않아요.",
                ("CHEER_PHRASE",),
                (
                    _plan(
                        "토끼가 거북이의 느린 걸음을 얕보며 자신이 쉽게 이길 거라고 말해요.",
                        ("hare", "tortoise"),
                        "거북이의 느린 걸음을 얕보는 토끼",
                        "토끼의 도발",
                    ),
                    _plan(
                        "거북이가 느려도 끝까지 갈 수 있다며 토끼에게 경주로 답해요.",
                        ("hare", "tortoise"),
                        "거북이의 도전",
                        "끝까지 가기",
                    ),
                    _plan(
                        "두 친구가 큰 나무에서 언덕 너머 종까지 같은 길로 달리기로 해요.",
                        ("hare", "tortoise"),
                        "출발선",
                        "결승선",
                    ),
                    _plan(
                        "토끼와 거북이가 같은 출발선에 나란히 서서 시작할 말을 기다려요.",
                        ("hare", "tortoise"),
                        "같은 출발선",
                        "시작할 말을 기다리는 두 친구",
                    ),
                ),
            ),
            _beat(
                "different-paces",
                "직전 답의 시작 말을 반영해 토끼는 앞서고 거북이는 자기 리듬을 찾아요.",
                "다음 장에서 거북이가 걸음마다 되풀이할 짧은 리듬 말의 내용",
                ("WALKING_RHYTHM", "SELF_TALK"),
                (
                    _plan(
                        "직전 답에서 정한 말을 둘이 외치고 토끼가 빠르게 앞서가요.",
                        ("hare", "tortoise"),
                        "직전 답에서 나온 시작 말",
                        "토끼의 빠른 출발",
                    ),
                    _plan(
                        "거북이가 숨을 고르며 자기에게 맞는 일정한 걸음을 찾아요.",
                        ("tortoise",),
                        "거북이의 리듬",
                    ),
                    _plan(
                        "토끼가 멀리 앞선 뒤 뒤를 돌아보고 여유를 부려요.",
                        ("hare", "tortoise"),
                        "토끼의 방심",
                    ),
                    _plan(
                        "거북이는 멀어진 토끼를 보아도 멈추지 않고 언덕으로 향해요.",
                        ("hare", "tortoise"),
                        "꾸준한 전진",
                    ),
                ),
            ),
            _beat(
                "nap-and-passing",
                "직전 답의 리듬을 이어 가는 동안 토끼가 잠들고 거북이가 곁을 지나가요.",
                "잠에서 깬 토끼가 앞선 거북이를 보고 처음 외칠 짧은 말의 내용",
                ("DIALOGUE",),
                (
                    _plan(
                        "거북이가 직전 답의 리듬 말을 되뇌며 큰 나무 쪽으로 다가와요.",
                        ("hare", "tortoise"),
                        "직전 답의 리듬 말",
                        "다가오는 거북이",
                    ),
                    _plan(
                        "앞서간 토끼가 큰 나무 그늘에서 잠깐만 쉬겠다며 누워요.",
                        ("hare",),
                        "큰 나무",
                        "토끼의 쉼",
                    ),
                    _plan(
                        "리듬 말이 가까워지는 동안 토끼가 잠들고 거북이가 곁을 지나가요.",
                        ("hare", "tortoise"),
                        "직전 답이 남긴 결과",
                        "잠든 토끼",
                        "거북이의 추월",
                    ),
                    _plan(
                        "거북이가 결승선의 종을 바라보며 마지막 언덕을 올라요.",
                        ("tortoise",),
                        "마지막 언덕",
                        "결승선 종",
                    ),
                ),
            ),
            _beat(
                "last-chase",
                "직전 답의 말을 외치며 토끼가 깨어 추격하지만 거북이가 먼저 결승선을 통과해요.",
                "종을 울린 거북이와 헐레벌떡 온 토끼가 주고받을 첫마디",
                ("DIALOGUE", "CHEER_PHRASE"),
                (
                    _plan(
                        "토끼가 잠에서 깨어 직전 답의 말을 외치며 거북이가 앞선 것을 발견해요.",
                        ("hare", "tortoise"),
                        "직전 답에서 나온 토끼의 첫마디",
                        "토끼의 깨달음",
                    ),
                    _plan(
                        "토끼가 힘껏 달리고 거북이는 마지막 걸음을 멈추지 않아요.",
                        ("hare", "tortoise"),
                        "마지막 추격",
                    ),
                    _plan(
                        "거북이가 토끼보다 먼저 결승선의 종을 울려요.",
                        ("hare", "tortoise"),
                        "거북이의 완주",
                    ),
                    _plan(
                        "토끼도 결승선에 도착해 거북이 앞에서 숨을 골라요.",
                        ("hare", "tortoise"),
                        "토끼의 도착",
                    ),
                ),
            ),
            _beat(
                "respectful-ending",
                "토끼가 사과하고 두 친구가 서로의 장점을 인정해요.",
                None,
                (),
                (
                    _plan(
                        "직전 대화에 이어 토끼가 자만했던 일을 솔직히 사과해요.",
                        ("hare", "tortoise"),
                        "토끼의 사과",
                    ),
                    _plan(
                        "토끼가 거북이의 꾸준함을 칭찬하고 거북이는 토끼의 빠름을 인정해요.",
                        ("hare", "tortoise"),
                        "서로의 장점",
                    ),
                    _plan(
                        "두 친구가 결승선에서 나란히 쉬며 경주를 돌아봐요.",
                        ("hare", "tortoise"),
                        "함께 쉬기",
                    ),
                    _plan(
                        "토끼와 거북이가 다음에는 서로 응원하며 달리기로 약속해요.",
                        ("hare", "tortoise"),
                        "우정",
                        "새 약속",
                    ),
                ),
                concluding=True,
            ),
        ),
    ),
    ServiceStoryFixture(
        template_id=1002,
        version=2,
        title="개미와 베짱이",
        source_title="이솝 우화 「개미와 베짱이」",
        source_note=SOURCE_NOTE,
        description="일과 음악을 좋아하는 두 친구가 겨울을 함께 준비해요.",
        context=(
            "들판에서 계절을 건너는 이야기예요. 개미의 준비성과 베짱이의 "
            "음악을 모두 쓸모 있는 재능으로 다루고, 굶주림이나 모욕 대신 "
            "갑작스러운 비와 협력으로 갈등을 만들어요."
        ),
        initial_summary="이야기는 여름 들판에서 개미와 베짱이가 만나기 전부터 시작해요.",
        characters=(
            _character(
                "ant",
                "개미",
                "주인공",
                "부지런하다",
                "겨울을 미리 준비한다",
                "필요할 때 친구를 돕는다",
            ),
            _character(
                "grasshopper",
                "베짱이",
                "주인공",
                "노래와 연주를 좋아한다",
                "밝고 다정하다",
                "경험을 통해 준비하는 법을 배운다",
            ),
        ),
        beats=(
            _beat(
                "summer-meeting",
                "개미와 베짱이가 들판에서 만나 서로 다른 일을 알게 돼요.",
                "다음 날 씨앗을 나를 때 둘이 맞춰 부를 짧은 일 노래",
                ("SONG_STYLE", "GREETING"),
                (
                    _plan(
                        "개미가 씨앗을 나르다가 풀잎 위에서 노래하는 베짱이를 만나요.",
                        ("ant", "grasshopper"),
                        "여름 들판",
                        "첫 만남",
                    ),
                    _plan(
                        "개미는 겨울 준비를 설명하고 베짱이는 음악을 만드는 일을 들려줘요.",
                        ("ant", "grasshopper"),
                        "겨울 준비",
                        "음악",
                    ),
                    _plan(
                        "바람에 씨앗이 굴러가자 베짱이가 노랫소리로 방향을 알려 줘요.",
                        ("ant", "grasshopper"),
                        "굴러가는 씨앗",
                        "노래로 돕기",
                    ),
                    _plan(
                        "두 친구가 씨앗을 되찾고 서로의 재능을 더 알고 싶어 해요.",
                        ("ant", "grasshopper"),
                        "서로 다른 재능",
                    ),
                ),
            ),
            _beat(
                "work-song",
                "아이디어를 반영한 노래가 실제로 일을 돕고 둘이 하루를 함께 보내요.",
                "먹구름을 본 베짱이가 개미에게 보낼 짧고 재미있는 경고 노랫소리",
                ("SIGNAL_SOUND", "SONG_STYLE"),
                (
                    _plan(
                        "베짱이가 고른 리듬을 연주하고 개미가 그 박자에 맞춰 씨앗을 날라요.",
                        ("ant", "grasshopper"),
                        "일을 돕는 노래",
                    ),
                    _plan(
                        "베짱이가 가벼운 씨앗을 모으고 개미가 저장할 곳을 정해요.",
                        ("ant", "grasshopper"),
                        "역할 나누기",
                    ),
                    _plan(
                        "둘이 작은 저장 구멍을 발견하지만 입구가 풀잎에 가려져 있어요.",
                        ("ant", "grasshopper"),
                        "가려진 저장 구멍",
                    ),
                    _plan(
                        "두 친구가 힘과 소리를 함께 써서 저장 구멍을 찾아 열어요.",
                        ("ant", "grasshopper"),
                        "협력",
                    ),
                ),
            ),
            _beat(
                "weather-warning",
                "직전 답의 경고 노랫소리가 울린 뒤 갑작스러운 비가 둘의 준비를 시험해요.",
                "빗방울이 떨어지자 둘이 먼저 낚아챌 물건과 구할 방법",
                ("ACTION_ORDER", "HELPFUL_ITEM"),
                (
                    _plan(
                        "베짱이가 직전 답의 경고 노랫소리를 내자 "
                        "먹구름과 찬 바람이 들판에 나타나요.",
                        ("ant", "grasshopper"),
                        "직전 답의 경고 노랫소리",
                        "먹구름",
                        "찬 바람",
                    ),
                    _plan(
                        "첫 빗방울이 떨어져 씨앗 자루와 베짱이의 악기가 젖기 시작해요.",
                        ("ant", "grasshopper"),
                        "젖는 씨앗",
                        "젖는 악기",
                    ),
                    _plan(
                        "개미와 베짱이가 큰 잎 아래로 물건을 옮기지만 바람이 더 세져요.",
                        ("ant", "grasshopper"),
                        "큰 잎",
                        "세진 바람",
                    ),
                    _plan(
                        "둘이 모든 물건을 지키려면 먼저 무엇을 할지 정해야 해요.",
                        ("ant", "grasshopper"),
                        "급한 선택",
                    ),
                ),
            ),
            _beat(
                "rain-rescue",
                "직전 선택으로 씨앗과 악기를 구하고 안전한 집으로 돌아가요.",
                "집 안에서 젖은 악기와 씨앗을 말리며 흉내 낼 재미있는 빗방울 소리",
                ("SIGNAL_SOUND", "SONG_STYLE"),
                (
                    _plan(
                        "직전 선택을 바로 실행해 씨앗과 악기 중 급한 것을 먼저 안전하게 해요.",
                        ("ant", "grasshopper"),
                        "선택 실행",
                    ),
                    _plan(
                        "개미가 작은 길을 찾고 베짱이가 소리로 서로의 위치를 알려요.",
                        ("ant", "grasshopper"),
                        "빗길",
                        "위치 신호",
                    ),
                    _plan(
                        "거센 빗물이 길을 막지만 둘이 잎사귀 다리를 놓아요.",
                        ("ant", "grasshopper"),
                        "빗물",
                        "잎사귀 다리",
                    ),
                    _plan(
                        "두 친구가 물건을 가지고 개미 집 앞에 무사히 도착해요.",
                        ("ant", "grasshopper"),
                        "안전한 도착",
                    ),
                ),
            ),
            _beat(
                "winter-together",
                "직전 답의 빗방울 소리로 젖은 물건을 말리며 일과 음악을 함께 나눠요.",
                "따뜻한 겨울밤 둘이 함께 부를 노래나 장난스러운 한마디",
                ("DIALOGUE", "SONG_STYLE"),
                (
                    _plan(
                        "베짱이가 직전 답의 빗방울 소리를 연주하고 "
                        "개미가 젖은 씨앗과 악기를 따뜻한 곳에 펼쳐 말려요.",
                        ("ant", "grasshopper"),
                        "직전 답의 빗방울 소리",
                        "씨앗과 악기 말리기",
                    ),
                    _plan(
                        "베짱이는 저장실 정리를 돕고 개미는 악기 줄을 고쳐 줘요.",
                        ("ant", "grasshopper"),
                        "서로 돕기",
                    ),
                    _plan(
                        "겨울이 와도 집에는 먹을거리와 작은 음악이 함께 있어요.",
                        ("ant", "grasshopper"),
                        "따뜻한 겨울",
                    ),
                    _plan(
                        "두 친구가 서로의 재능 덕분에 겨울이 즐겁다고 느껴요.",
                        ("ant", "grasshopper"),
                        "고마움",
                    ),
                ),
            ),
            _beat(
                "spring-promise",
                "봄이 오고 둘이 다음 여름의 일과 음악을 함께 계획해요.",
                None,
                (),
                (
                    _plan(
                        "직전 대화에 이어 개미와 베짱이가 서로에게 고맙다고 말해요.",
                        ("ant", "grasshopper"),
                        "감사",
                    ),
                    _plan(
                        "봄 햇살 아래에서 새 씨앗이 저장실 밖으로 옮겨져요.",
                        ("ant", "grasshopper"),
                        "봄",
                        "새 씨앗",
                    ),
                    _plan(
                        "둘이 일할 시간과 노래할 시간을 함께 정해요.",
                        ("ant", "grasshopper"),
                        "새 계획",
                    ),
                    _plan(
                        "개미와 베짱이가 들판에서 힘찬 새 노래로 봄을 시작해요.",
                        ("ant", "grasshopper"),
                        "우정",
                        "봄 노래",
                    ),
                ),
                concluding=True,
            ),
        ),
    ),
    ServiceStoryFixture(
        template_id=1003,
        version=2,
        title="사자와 생쥐",
        source_title="이솝 우화 「사자와 생쥐」",
        source_note=SOURCE_NOTE,
        description="작은 생쥐의 용기와 솜씨가 큰 사자를 구해요.",
        context=(
            "작은 약속이 큰 힘이 되는 숲속 우정 이야기예요. 힘의 크기보다 약속과 "
            "세심한 솜씨의 가치를 보여 주며, 그물 사건은 무섭거나 다치는 "
            "묘사 없이 안전하게 다뤄요."
        ),
        initial_summary="이야기는 사자가 큰 나무 아래에서 쉬기 전부터 시작해요.",
        characters=(
            _character(
                "lion",
                "사자",
                "주인공",
                "힘이 세다",
                "처음에는 작은 도움을 대수롭지 않게 여긴다",
                "한 약속은 지킨다",
            ),
            _character(
                "mouse",
                "생쥐",
                "주인공",
                "몸집이 작다",
                "용감하다",
                "매듭처럼 세밀한 것을 잘 다룬다",
            ),
        ),
        beats=(
            _beat(
                "unexpected-meeting",
                "생쥐가 사자를 깨우고 사과한 뒤 자신이 도울 방법을 말할 기회를 얻어요.",
                "작은 생쥐가 큰 사자에게 당차게 건넬 약속 한마디",
                ("DIALOGUE", "PROMISE"),
                (
                    _plan(
                        "먹이를 찾던 생쥐가 실수로 잠든 사자의 앞발에 부딪혀요.",
                        ("lion", "mouse"),
                        "뜻밖의 만남",
                    ),
                    _plan(
                        "사자가 깨어 생쥐를 바라보고 생쥐는 실수를 솔직히 사과해요.",
                        ("lion", "mouse"),
                        "생쥐의 사과",
                    ),
                    _plan(
                        "생쥐가 몸집은 작아도 언젠가 사자를 도울 수 있다며 말할 기회를 청해요.",
                        ("lion", "mouse"),
                        "도움의 약속",
                    ),
                    _plan(
                        "사자가 앞발의 힘을 풀고 생쥐가 어떤 약속을 할지 흥미롭게 기다려요.",
                        ("lion", "mouse"),
                        "약속을 기다리는 사자",
                    ),
                ),
            ),
            _beat(
                "small-skill",
                "직전 답의 약속을 들은 사자가 생쥐를 놓아주고, "
                "생쥐의 작은 솜씨가 숲에서 실제로 빛나요.",
                "멀리 떨어져도 서로 도움을 청할 수 있는 둘만의 짧은 신호",
                ("SIGNAL", "HELP_REQUEST"),
                (
                    _plan(
                        "생쥐가 직전 답의 약속을 건네자 사자가 생쥐를 놓아주고 "
                        "생쥐는 숲길로 돌아가요.",
                        ("lion", "mouse"),
                        "직전 답의 약속",
                        "사자가 놓아준 생쥐",
                    ),
                    _plan(
                        "생쥐가 새 둥지에 걸린 가는 끈을 발견해 매듭을 풀어요.",
                        ("mouse",),
                        "가는 끈",
                        "작은 솜씨",
                    ),
                    _plan(
                        "사자가 그 모습을 보고 작은 발도 중요한 일을 한다는 것을 알아요.",
                        ("lion", "mouse"),
                        "사자의 깨달음",
                    ),
                    _plan(
                        "두 친구가 각자의 힘이 필요한 때를 위해 멀리서도 통할 신호를 정하려 해요.",
                        ("lion", "mouse"),
                        "도움 신호를 정하기 직전",
                    ),
                ),
            ),
            _beat(
                "lion-in-net",
                "직전 답의 도움 신호를 연습한 뒤 사자가 빈 그물에 걸리고 "
                "생쥐가 신호를 듣고 달려와요.",
                "생쥐가 가장 먼저 갉아 볼 그물의 매듭",
                ("ACTION_ORDER", "ROPE_TARGET"),
                (
                    _plan(
                        "사자와 생쥐가 직전 답의 도움 신호를 주고받아 본 뒤 각자의 숲길로 떠나요.",
                        ("lion", "mouse"),
                        "직전 답의 도움 신호",
                    ),
                    _plan(
                        "사자가 숲길을 걷다가 풀에 가려진 빈 그물에 발이 걸리고 "
                        "당길수록 매듭이 단단해져요.",
                        ("lion",),
                        "빈 그물",
                        "걸린 발",
                        "단단해진 매듭",
                    ),
                    _plan(
                        "사자가 다치지 않은 채 둘만의 도움 신호를 보내고 생쥐가 듣고 달려와요.",
                        ("lion", "mouse"),
                        "둘만의 도움 신호",
                        "달려오는 생쥐",
                    ),
                    _plan(
                        "생쥐가 그물 앞에 도착해 서로 얽힌 매듭들을 자세히 살펴요.",
                        ("lion", "mouse"),
                        "생쥐의 도착",
                        "매듭 살피기",
                    ),
                ),
            ),
            _beat(
                "mouse-rescue",
                "생쥐가 고른 순서로 매듭을 풀고 사자가 그물 밖으로 나와요.",
                "그물에서 나온 사자가 생쥐에게 놀라서 할 첫마디",
                ("DIALOGUE", "THANKS"),
                (
                    _plan(
                        "생쥐가 직전 선택에 따라 느슨한 줄부터 갉거나 매듭부터 풀어요.",
                        ("lion", "mouse"),
                        "선택한 구조 방법",
                    ),
                    _plan(
                        "사자가 생쥐의 신호에 맞춰 몸을 천천히 움직여 틈을 넓혀요.",
                        ("lion", "mouse"),
                        "협력",
                    ),
                    _plan(
                        "마지막 줄이 풀리고 사자가 안전하게 그물 밖으로 나와요.",
                        ("lion", "mouse"),
                        "구조 성공",
                    ),
                    _plan(
                        "사자가 작은 친구의 큰 도움을 떠올리며 생쥐를 바라봐요.",
                        ("lion", "mouse"),
                        "고마움",
                    ),
                ),
            ),
            _beat(
                "equal-friends",
                "사자와 생쥐가 서로를 동등한 친구로 인정하고 함께 숲을 살펴요.",
                None,
                (),
                (
                    _plan(
                        "사자가 직전 답의 첫마디로 생쥐에게 진심을 전하고 약속을 기억해요.",
                        ("lion", "mouse"),
                        "직전 답의 첫마디",
                        "사자의 감사",
                    ),
                    _plan(
                        "생쥐는 누구나 서로 다른 방식으로 도울 수 있다고 답해요.",
                        ("lion", "mouse"),
                        "생쥐의 대답",
                    ),
                    _plan(
                        "둘이 위험한 빈 그물을 한쪽으로 정리해 다른 동물이 걸리지 않게 해요.",
                        ("lion", "mouse"),
                        "숲길 정리",
                    ),
                    _plan(
                        "사자와 생쥐가 크기와 상관없이 서로 돕는 친구가 돼요.",
                        ("lion", "mouse"),
                        "진짜 우정",
                    ),
                ),
                concluding=True,
            ),
        ),
    ),
    ServiceStoryFixture(
        template_id=1004,
        version=2,
        title="해와 바람",
        source_title="이솝 우화 「북풍과 해」",
        source_note=SOURCE_NOTE,
        description="해와 바람이 힘을 쓰는 서로 다른 방법을 배워요.",
        context=(
            "해와 바람이 솜씨를 겨루는 하늘 이야기예요. 나그네를 괴롭히는 경쟁이 되지 "
            "않도록 안전 규칙을 먼저 세우고, 억지보다 따뜻함이 효과적인 순간을 "
            "보여 준 뒤 두 힘이 협력해요."
        ),
        initial_summary="이야기는 해와 바람이 구름 위에서 만나기 전부터 시작해요.",
        characters=(
            _character("sun", "해", "주인공", "따뜻하다", "차분하다", "기다릴 줄 안다"),
            _character(
                "wind",
                "바람",
                "주인공",
                "빠르고 힘이 세다",
                "처음에는 힘만 중요하다고 여긴다",
                "다른 방법을 배운다",
            ),
            _character(
                "traveler",
                "나그네",
                "조연",
                "외투를 입고 길을 걷는다",
                "스스로 편안한 행동을 선택한다",
            ),
        ),
        beats=(
            _beat(
                "strength-debate",
                "해와 바람이 나그네의 외투를 두고 안전한 겨루기 규칙을 정해요.",
                "해와 바람의 겨루기를 시작하고 멈출 재미있는 신호",
                ("FAIR_RULE", "STOP_SIGNAL"),
                (
                    _plan(
                        "해와 바람이 구름 위에서 누가 더 힘이 센지 이야기해요.",
                        ("sun", "wind"),
                        "힘에 관한 대화",
                    ),
                    _plan(
                        "외투를 입은 나그네가 아래 길로 걸어오는 것을 발견해요.",
                        ("sun", "wind", "traveler"),
                        "나그네",
                        "외투",
                    ),
                    _plan(
                        "해와 바람이 나그네가 스스로 외투를 벗게 하는 겨루기를 제안해요.",
                        ("sun", "wind", "traveler"),
                        "겨루기 제안",
                    ),
                    _plan(
                        "두 친구가 나그네를 다치게 하지 않는 규칙을 정하고 "
                        "바람이 시작 신호를 기다려요.",
                        ("sun", "wind", "traveler"),
                        "안전 규칙",
                        "신호를 기다리는 바람",
                    ),
                ),
            ),
            _beat(
                "wind-tries",
                "직전 답의 신호로 시작한 바람이 불어 보지만 외투는 더 단단히 여며져요.",
                "바람이 해에게 차례를 건네며 외칠 장난스러운 한마디",
                ("DIALOGUE", "START_SIGNAL"),
                (
                    _plan(
                        "직전 답의 시작 신호가 울리자 바람이 나그네 곁에 산들바람을 보내요.",
                        ("wind", "traveler"),
                        "직전 답의 시작 신호",
                        "산들바람",
                    ),
                    _plan(
                        "외투가 살짝 흔들리지만 나그네는 계속 길을 걸어요.",
                        ("wind", "traveler"),
                        "흔들리는 외투",
                    ),
                    _plan(
                        "바람이 조금 세지자 나그네가 외투를 더 단단히 여며요.",
                        ("wind", "traveler"),
                        "여민 외투",
                    ),
                    _plan(
                        "바람은 멈춤 신호를 지키고 힘만으로는 안 된다는 것을 깨달아요.",
                        ("sun", "wind", "traveler"),
                        "멈춤",
                        "바람의 깨달음",
                    ),
                ),
            ),
            _beat(
                "sun-warms",
                "직전 답으로 차례를 받은 해가 천천히 따뜻한 빛을 보내 나그네가 편안함을 느껴요.",
                "따뜻해진 나그네가 외투를 벗고 쉬고 싶은 곳",
                ("REST_PLACE", "SENSORY_DETAIL"),
                (
                    _plan(
                        "바람이 직전 답의 한마디를 외치자 해가 구름 사이로 부드러운 빛을 비춰요.",
                        ("sun", "wind", "traveler"),
                        "직전 답의 한마디",
                        "부드러운 햇빛",
                    ),
                    _plan(
                        "길과 공기가 따뜻해지고 나그네의 어깨가 편안해져요.",
                        ("sun", "traveler"),
                        "따뜻한 공기",
                    ),
                    _plan(
                        "나그네가 스스로 외투 단추를 풀고 그늘이 있는 곳을 찾아요.",
                        ("sun", "traveler"),
                        "스스로 푼 외투",
                    ),
                    _plan(
                        "나그네가 외투를 벗어 팔에 걸고 어디에서 쉴지 주변을 둘러봐요.",
                        ("sun", "wind", "traveler"),
                        "외투 벗기",
                        "쉴 곳을 고르기 직전",
                    ),
                ),
            ),
            _beat(
                "sudden-cloud",
                "직전 답의 장소로 향하는 나그네 앞을 큰 구름이 가리자 해와 바람이 함께 도와요.",
                "해와 바람이 나그네의 남은 길에 함께 만들어 줄 포근한 날씨",
                ("TEAM_ACTION", "WEATHER_STYLE"),
                (
                    _plan(
                        "나그네가 직전 답의 쉼터로 향하는 순간 "
                        "큰 구름이 몰려와 앞길을 어둡게 가려요.",
                        ("sun", "wind", "traveler"),
                        "직전 답의 쉼터",
                        "큰 구름",
                        "가려진 길",
                    ),
                    _plan(
                        "바람이 구름 가장자리를 밀고 해가 틈으로 빛을 보내요.",
                        ("sun", "wind"),
                        "첫 협력",
                    ),
                    _plan(
                        "나그네가 다시 보이는 길을 따라 안전하게 걸어요.",
                        ("sun", "wind", "traveler"),
                        "밝아진 길",
                    ),
                    _plan(
                        "해와 바람은 서로 다른 힘을 함께 쓰면 더 도움이 된다는 것을 알아요.",
                        ("sun", "wind"),
                        "서로 다른 힘",
                    ),
                ),
            ),
            _beat(
                "shared-strength",
                "직전 답의 날씨를 함께 만들며 해와 바람이 서로를 인정하고 나그네를 도와요.",
                None,
                (),
                (
                    _plan(
                        "해와 바람이 직전 답의 포근한 날씨를 함께 만들며 서로의 방법을 칭찬해요.",
                        ("sun", "wind", "traveler"),
                        "직전 답의 포근한 날씨",
                        "서로의 인정",
                    ),
                    _plan(
                        "둘이 알맞은 빛과 바람으로 나그네가 걷기 좋은 길을 만들어요.",
                        ("sun", "wind", "traveler"),
                        "알맞은 빛과 바람",
                    ),
                    _plan(
                        "나그네가 편안한 걸음으로 마을 가까이 도착해요.",
                        ("sun", "wind", "traveler"),
                        "마을 도착",
                    ),
                    _plan(
                        "해와 바람이 힘은 쓰는 방법에 따라 달라진다는 것을 기억해요.",
                        ("sun", "wind"),
                        "힘의 올바른 쓰임",
                    ),
                ),
                concluding=True,
            ),
        ),
    ),
    ServiceStoryFixture(
        template_id=1005,
        version=2,
        title="시골 쥐와 도시 쥐",
        source_title="이솝 우화 「시골 쥐와 도시 쥐」",
        source_note=SOURCE_NOTE,
        description="두 쥐가 서로의 생활을 경험하고 각자의 행복을 이해해요.",
        context=(
            "서로 다른 두 친구가 오가는 여행 이야기예요. 시골과 도시 중 "
            "하나를 낮추지 않고, 화려하지만 시끄러운 도시와 소박하지만 "
            "편안한 시골의 차이를 경험하게 해요."
        ),
        initial_summary="이야기는 도시 쥐가 시골 친구를 찾아가기 전부터 시작해요.",
        characters=(
            _character(
                "country_mouse",
                "시골 쥐",
                "주인공",
                "소박하다",
                "조용한 곳을 좋아한다",
                "손님을 따뜻하게 맞는다",
            ),
            _character(
                "city_mouse",
                "도시 쥐",
                "주인공",
                "활발하다",
                "새로운 것을 잘 안다",
                "친구의 선택을 존중한다",
            ),
            _character(
                "house_cat",
                "집고양이",
                "갈등 인물",
                "큰 집을 천천히 돌아다닌다",
                "쥐들을 다치게 하지 않는다",
            ),
        ),
        beats=(
            _beat(
                "country-welcome",
                "도시 쥐가 시골 집을 방문하고 소박한 음식이 놓인 식탁에 마주 앉아요.",
                "도시 쥐가 시골 식탁에서 가장 먼저 맛보고 싶은 한입",
                ("FOOD_DETAIL", "WELCOME_PHRASE"),
                (
                    _plan(
                        "도시 쥐가 들판 길을 지나 시골 쥐의 작은 집에 도착해요.",
                        ("country_mouse", "city_mouse"),
                        "도시 쥐의 방문",
                    ),
                    _plan(
                        "시골 쥐가 친구를 반갑게 맞고 조용한 집을 안내해요.",
                        ("country_mouse", "city_mouse"),
                        "따뜻한 환영",
                    ),
                    _plan(
                        "두 친구가 곡식과 열매를 작은 식탁에 차려요.",
                        ("country_mouse", "city_mouse"),
                        "소박한 음식",
                    ),
                    _plan(
                        "도시 쥐가 고요한 소리를 들으며 식탁의 어느 음식부터 맛볼지 둘러봐요.",
                        ("country_mouse", "city_mouse"),
                        "시골의 고요함",
                        "첫 음식을 고르기 직전",
                    ),
                ),
            ),
            _beat(
                "city-invitation",
                "음식 선택을 반영한 식사 뒤 도시 쥐가 친구를 도시로 초대해요.",
                "시골 쥐가 여행에 챙길 작은 물건",
                ("TRAVEL_ITEM", "DIALOGUE"),
                (
                    _plan(
                        "두 친구가 고른 음식을 나눠 먹으며 서로 사는 곳을 이야기해요.",
                        ("country_mouse", "city_mouse"),
                        "함께한 식사",
                    ),
                    _plan(
                        "도시 쥐가 밝은 불빛과 넓은 식탁이 있는 도시 집을 들려줘요.",
                        ("country_mouse", "city_mouse"),
                        "도시 집 이야기",
                    ),
                    _plan(
                        "시골 쥐가 도시를 궁금해하자 도시 쥐가 함께 가자고 초대해요.",
                        ("country_mouse", "city_mouse"),
                        "도시 초대",
                    ),
                    _plan(
                        "시골 쥐가 여행을 받아들이고 필요한 것을 챙기려 해요.",
                        ("country_mouse", "city_mouse"),
                        "여행 준비",
                    ),
                ),
            ),
            _beat(
                "journey-to-city",
                "두 친구가 들판과 돌길을 지나 낯선 도시 집에 도착해요.",
                "도시 집 안의 큰 소리 속에서 서로를 놓치면 쓸 둘만의 짧은 신호",
                ("SIGNAL", "PATH_MARKER"),
                (
                    _plan(
                        "시골 쥐가 고른 물건을 챙기고 도시 쥐와 들판을 떠나요.",
                        ("country_mouse", "city_mouse"),
                        "여행 출발",
                    ),
                    _plan(
                        "두 친구가 조용한 흙길에서 소리가 큰 돌길로 들어서요.",
                        ("country_mouse", "city_mouse"),
                        "달라진 길",
                    ),
                    _plan(
                        "사람들의 발소리에 시골 쥐가 놀라자 도시 쥐가 안전한 길을 알려 줘요.",
                        ("country_mouse", "city_mouse"),
                        "도시의 큰 소리",
                        "안내",
                    ),
                    _plan(
                        "둘이 서로를 확인하며 큰 도시 집의 작은 구멍에 도착해요.",
                        ("country_mouse", "city_mouse"),
                        "도시 집 도착",
                    ),
                ),
            ),
            _beat(
                "grand-feast",
                "직전 답의 신호를 약속한 두 쥐가 화려한 식탁을 구경하다 큰 발소리를 들어요.",
                "쿵쿵 발소리가 나면 두 쥐가 쏙 숨을 식탁 근처 장소",
                ("SAFE_PLACE", "ACTION_ORDER"),
                (
                    _plan(
                        "두 쥐가 직전 답의 신호를 주고받아 본 뒤 높은 식탁의 여러 음식을 구경해요.",
                        ("country_mouse", "city_mouse"),
                        "직전 답의 둘만의 신호",
                        "화려한 식탁",
                    ),
                    _plan(
                        "두 친구가 맛을 보려는 순간 멀리서 무거운 발소리가 들려요.",
                        ("country_mouse", "city_mouse", "house_cat"),
                        "다가오는 발소리",
                    ),
                    _plan(
                        "집고양이가 방으로 들어와 천천히 식탁 아래를 지나가요.",
                        ("country_mouse", "city_mouse", "house_cat"),
                        "집고양이",
                    ),
                    _plan(
                        "두 쥐가 다치지 않으려면 조용하고 안전한 곳을 빨리 골라야 해요.",
                        ("country_mouse", "city_mouse", "house_cat"),
                        "안전한 숨기",
                    ),
                ),
            ),
            _beat(
                "quiet-choice",
                "두 친구가 안전하게 숨은 뒤 시골 쥐가 자신에게 맞는 행복을 선택해요.",
                "시골로 돌아가며 도시 쥐에게 남길 솔직하고 다정한 한마디",
                ("DIALOGUE", "FAREWELL"),
                (
                    _plan(
                        "두 쥐가 직전 선택에 따라 안전한 틈으로 몸을 숨겨요.",
                        ("country_mouse", "city_mouse", "house_cat"),
                        "안전한 선택 실행",
                    ),
                    _plan(
                        "집고양이가 지나간 뒤에도 시골 쥐의 가슴은 빠르게 뛰어요.",
                        ("country_mouse", "city_mouse"),
                        "시골 쥐의 긴장",
                    ),
                    _plan(
                        "시골 쥐는 적은 음식이라도 편안한 집에서 먹고 싶다고 말해요.",
                        ("country_mouse", "city_mouse"),
                        "편안함의 선택",
                    ),
                    _plan(
                        "도시 쥐가 친구의 선택을 존중하며 함께 돌아갈 길을 찾아요.",
                        ("country_mouse", "city_mouse"),
                        "선택 존중",
                    ),
                ),
            ),
            _beat(
                "friends-in-two-homes",
                "두 쥐가 시골로 돌아와 서로 다른 생활을 존중하며 우정을 이어 가요.",
                None,
                (),
                (
                    _plan(
                        "시골 쥐가 직전 답의 솔직하고 다정한 한마디를 도시 쥐에게 전해요.",
                        ("country_mouse", "city_mouse"),
                        "직전 답의 작별 한마디",
                        "작별 인사",
                    ),
                    _plan(
                        "두 친구가 익숙한 들판 길을 따라 시골 집으로 돌아와요.",
                        ("country_mouse", "city_mouse"),
                        "시골 귀환",
                    ),
                    _plan(
                        "도시 쥐도 조용한 식탁에서 편안하게 웃어요.",
                        ("country_mouse", "city_mouse"),
                        "편안한 식사",
                    ),
                    _plan(
                        "둘은 각자의 집에서 살며 안전한 날마다 서로 방문하기로 해요.",
                        ("country_mouse", "city_mouse"),
                        "다름의 존중",
                        "이어지는 우정",
                    ),
                ),
                concluding=True,
            ),
        ),
    ),
    ServiceStoryFixture(
        template_id=1006,
        version=2,
        title="금도끼 은도끼",
        source_title="전래 이야기 「금도끼 은도끼」",
        source_note=SOURCE_NOTE,
        description="나무꾼이 잃어버린 쇠도끼를 정직하게 찾아요.",
        context=(
            "산속 연못에서 벌어지는 전래 이야기예요. 나무꾼은 금이나 은을 "
            "탐내지 않으며, 산신령은 정답을 강요하지 않고 세 번의 확인으로 "
            "정직함을 알아봐요."
        ),
        initial_summary="이야기는 나무꾼이 연못가에서 일을 시작하기 전부터 시작해요.",
        characters=(
            _character(
                "woodcutter",
                "나무꾼",
                "주인공",
                "성실하다",
                "자기 물건을 정확히 안다",
                "남의 물건을 탐내지 않는다",
            ),
            _character(
                "mountain_spirit",
                "산신령",
                "조력자",
                "차분하다",
                "공정하다",
                "정직한 사람을 돕는다",
            ),
        ),
        beats=(
            _beat(
                "axe-falls",
                "나무꾼의 쇠도끼가 연못에 빠지고 스스로 찾으려 해요.",
                "연못에 들어가지 않고 도끼를 찾을 기발한 도구나 방법",
                ("SEARCH_METHOD", "SAFE_ACTION"),
                (
                    _plan(
                        "나무꾼이 연못가에서 마른 나무를 정리하며 성실히 일해요.",
                        ("woodcutter",),
                        "연못가",
                        "쇠도끼",
                    ),
                    _plan(
                        "손잡이가 미끄러져 쇠도끼가 연못 가운데로 빠져요.",
                        ("woodcutter",),
                        "연못에 빠진 쇠도끼",
                    ),
                    _plan(
                        "나무꾼이 물에 함부로 들어가지 않고 가장자리에서 도끼를 찾아봐요.",
                        ("woodcutter",),
                        "안전한 찾기",
                    ),
                    _plan(
                        "도끼가 보이지 않자 나무꾼이 도움을 구할 방법을 생각해요.",
                        ("woodcutter",),
                        "도움 필요",
                    ),
                ),
            ),
            _beat(
                "spirit-appears",
                "나무꾼의 도움 요청을 듣고 산신령이 나타나 잃어버린 도끼의 모습을 물어요.",
                "나무꾼이 산신령에게 잃어버린 쇠도끼의 생김새를 알려 줄 첫마디",
                ("DIALOGUE", "DESCRIPTION"),
                (
                    _plan(
                        "나무꾼이 직전 방법을 시도한 뒤 연못을 향해 차분히 도움을 청해요.",
                        ("woodcutter",),
                        "도움 요청",
                    ),
                    _plan(
                        "연못에 잔잔한 물결이 생기고 산신령이 모습을 보여요.",
                        ("woodcutter", "mountain_spirit"),
                        "산신령의 등장",
                    ),
                    _plan(
                        "산신령이 잃어버린 도끼의 모습을 자세히 물어요.",
                        ("woodcutter", "mountain_spirit"),
                        "쇠도끼 설명",
                    ),
                    _plan(
                        "산신령이 나무꾼에게 쇠도끼의 생김새를 자세히 말해 달라며 기다려요.",
                        ("woodcutter", "mountain_spirit"),
                        "쇠도끼 설명을 기다리는 산신령",
                    ),
                ),
            ),
            _beat(
                "golden-test",
                "직전 답의 설명을 들은 산신령이 금도끼를 보여 주고 나무꾼의 대답을 기다려요.",
                "반짝이는 금도끼를 본 나무꾼이 바로 할 첫마디",
                ("DIALOGUE", "HONEST_ANSWER"),
                (
                    _plan(
                        "산신령이 직전 답의 설명을 들은 뒤 "
                        "반짝이는 금도끼를 들고 연못 위로 올라와요.",
                        ("woodcutter", "mountain_spirit"),
                        "직전 답의 쇠도끼 설명",
                        "금도끼",
                    ),
                    _plan(
                        "산신령이 금도끼가 나무꾼의 것인지 차분히 물어요.",
                        ("woodcutter", "mountain_spirit"),
                        "첫 번째 질문",
                    ),
                    _plan(
                        "나무꾼은 아름다운 금도끼를 보지만 자기 것이 아님을 알아요.",
                        ("woodcutter", "mountain_spirit"),
                        "유혹",
                        "정확한 기억",
                    ),
                    _plan(
                        "나무꾼이 욕심내지 않고 금도끼를 바라보며 솔직한 대답을 하려 해요.",
                        ("woodcutter", "mountain_spirit"),
                        "첫 번째 대답 직전",
                    ),
                ),
            ),
            _beat(
                "silver-test",
                "직전 답으로 금도끼를 거절하자 산신령이 은도끼를 보여 주고 다시 대답을 기다려요.",
                "은도끼를 본 나무꾼이 산신령에게 건넬 솔직한 대답",
                ("DIALOGUE", "HONEST_ANSWER"),
                (
                    _plan(
                        "나무꾼이 직전 답으로 금도끼를 거절하자 "
                        "산신령이 이번에는 은도끼를 가져와요.",
                        ("woodcutter", "mountain_spirit"),
                        "직전 답의 정직한 대답",
                        "은도끼",
                    ),
                    _plan(
                        "은도끼도 나무꾼의 낡은 쇠도끼보다 훨씬 반짝여요.",
                        ("woodcutter", "mountain_spirit"),
                        "두 번째 유혹",
                    ),
                    _plan(
                        "나무꾼이 손잡이와 날을 살펴보고 자기 것이 아님을 확인해요.",
                        ("woodcutter", "mountain_spirit"),
                        "도끼 확인",
                    ),
                    _plan(
                        "나무꾼이 은도끼를 바라보며 산신령에게 솔직한 대답을 하려 해요.",
                        ("woodcutter", "mountain_spirit"),
                        "두 번째 대답 직전",
                    ),
                ),
            ),
            _beat(
                "iron-found",
                "직전 답으로 은도끼를 거절하자 산신령이 낡은 쇠도끼를 보여 주고 나무꾼이 알아봐요.",
                "낡은 쇠도끼를 알아본 나무꾼이 기뻐서 외칠 말",
                ("DIALOGUE", "THANKS"),
                (
                    _plan(
                        "나무꾼이 직전 답으로 은도끼를 거절하자 "
                        "산신령이 익숙한 나무 손잡이의 쇠도끼를 가져와요.",
                        ("woodcutter", "mountain_spirit"),
                        "직전 답의 정직한 대답",
                        "쇠도끼",
                    ),
                    _plan(
                        "나무꾼이 손잡이의 작은 흠집을 보고 자기 도끼임을 알아봐요.",
                        ("woodcutter", "mountain_spirit"),
                        "익숙한 흔적",
                    ),
                    _plan(
                        "나무꾼이 바로 자기 쇠도끼임을 알아보고 기쁜 대답을 하려 해요.",
                        ("woodcutter", "mountain_spirit"),
                        "기쁜 대답 직전",
                    ),
                    _plan(
                        "산신령이 나무꾼이 쇠도끼를 정확히 알아본 것을 보고 대답을 기다려요.",
                        ("woodcutter", "mountain_spirit"),
                        "쇠도끼를 돌려주기 직전",
                    ),
                ),
            ),
            _beat(
                "honest-reward",
                "직전 답의 기쁜 말을 들은 산신령이 쇠도끼와 정직함의 선물을 건네요.",
                "선물받은 도끼로 이웃을 깜짝 도울 일",
                ("GIFT_USE", "HELP_METHOD"),
                (
                    _plan(
                        "나무꾼이 직전 답의 기쁜 말을 외치자 "
                        "산신령이 쇠도끼를 돌려주고 정직함을 칭찬해요.",
                        ("woodcutter", "mountain_spirit"),
                        "직전 답의 기쁜 말",
                        "쇠도끼 되찾기",
                        "감사",
                    ),
                    _plan(
                        "산신령이 정직한 마음을 기리며 금도끼와 은도끼도 선물해요.",
                        ("woodcutter", "mountain_spirit"),
                        "정직함의 선물",
                    ),
                    _plan(
                        "나무꾼은 갑작스러운 선물을 혼자만 쓰지 않겠다고 생각해요.",
                        ("woodcutter",),
                        "선물의 바른 쓰임",
                    ),
                    _plan(
                        "나무꾼이 마을 사람들에게 도움이 될 나눔 방법을 고르려 해요.",
                        ("woodcutter",),
                        "이웃과 나누기",
                    ),
                ),
            ),
            _beat(
                "truth-returns-home",
                "나무꾼이 선물을 바르게 쓰고 정직한 선택을 마을에 전해요.",
                None,
                (),
                (
                    _plan(
                        "나무꾼이 직전 선택대로 선물의 일부를 이웃에게 나눠요.",
                        ("woodcutter",),
                        "선물 나누기",
                    ),
                    _plan(
                        "나무꾼은 되찾은 쇠도끼로 필요한 만큼만 성실히 일해요.",
                        ("woodcutter",),
                        "쇠도끼로 일하기",
                    ),
                    _plan(
                        "마을 사람들은 금보다 정직한 마음이 더 귀하다는 이야기를 들어요.",
                        ("woodcutter",),
                        "정직의 가치",
                    ),
                    _plan(
                        "나무꾼은 자기 물건과 곧은 마음을 오래 소중히 지켜요.",
                        ("woodcutter",),
                        "정직한 결말",
                    ),
                ),
                concluding=True,
            ),
        ),
    ),
    ServiceStoryFixture(
        template_id=1007,
        version=2,
        title="빨간 모자와 숲길 약속",
        source_title="유럽 민담 「빨간 모자」의 안전한 재구성",
        source_note=SOURCE_NOTE,
        description="빨간 모자가 안전 약속과 관찰력으로 할머니 집에 가요.",
        context=(
            "빨간 모자가 약속을 지키며 걷는 숲길 모험이에요. 삼키기, 사냥, 부상은 "
            "없고, 늑대의 장난은 비밀 신호와 관찰력으로 막아요. 늑대는 끝에 "
            "잘못을 인정하고 관계의 경계를 배워요."
        ),
        initial_summary="이야기는 빨간 모자가 바구니를 준비하기 전부터 시작해요.",
        characters=(
            _character(
                "red_hood",
                "빨간 모자",
                "주인공",
                "친절하다",
                "호기심이 많다",
                "안전 약속을 기억하고 스스로 확인한다",
            ),
            _character(
                "grandmother",
                "할머니",
                "조력자",
                "현명하다",
                "침착하다",
                "비밀 신호를 기억한다",
            ),
            _character(
                "wolf",
                "늑대",
                "갈등 인물",
                "영리하지만 성급하다",
                "누구도 다치게 하지 않는다",
                "마지막에는 장난을 사과한다",
            ),
            _character(
                "forest_keeper",
                "숲지기",
                "조력자",
                "길을 잘 안다",
                "도움을 요청받으면 안전하게 돕는다",
            ),
        ),
        beats=(
            _beat(
                "basket-and-rules",
                "빨간 모자가 기본 바구니를 챙기고 할머니 집까지 지킬 약속을 확인해요.",
                "바구니에 더 넣을 작은 선물",
                ("BASKET_ITEM",),
                (
                    _plan(
                        "빨간 모자가 아픈 할머니에게 가져갈 빵과 차를 바구니에 담아요.",
                        ("red_hood",),
                        "할머니의 바구니",
                    ),
                    _plan(
                        "빨간 모자가 정해진 길을 걷고 낯선 이를 따라가지 않기로 해요.",
                        ("red_hood",),
                        "숲길 안전 약속",
                    ),
                    _plan(
                        "할머니와 빨간 모자만 아는 문 두드림 신호를 다시 확인해요.",
                        ("red_hood", "grandmother"),
                        "비밀 신호",
                    ),
                    _plan(
                        "빨간 모자가 빵과 차 옆의 빈자리에 어떤 작은 선물을 더 넣을지 생각해요.",
                        ("red_hood",),
                        "선물을 고르기 직전",
                    ),
                ),
            ),
            _beat(
                "wolf-meeting",
                "직전 선택을 바구니에 넣어 출발하고 빨간 모자가 늑대와 안전하게 대화해요.",
                "늑대의 엉뚱한 질문을 넘길 빨간 모자의 재치 있는 한마디",
                ("SAFE_DIALOGUE", "PATH_MARKER"),
                (
                    _plan(
                        "빨간 모자가 직전 답의 선물을 바구니에 넣고 밝은 숲길로 출발해요.",
                        ("red_hood",),
                        "직전 답의 선물",
                        "할머니 집으로 출발",
                    ),
                    _plan(
                        "늑대가 꽃밭 옆에서 나타나 어디로 가는지 물어요.",
                        ("red_hood", "wolf"),
                        "늑대와 만남",
                    ),
                    _plan(
                        "빨간 모자는 할머니 집의 위치를 말하지 않고 안전한 거리를 지켜요.",
                        ("red_hood", "wolf"),
                        "개인정보 지키기",
                    ),
                    _plan(
                        "늑대가 지름길을 안다며 권하지만 빨간 모자는 길 표지를 확인해요.",
                        ("red_hood", "wolf"),
                        "수상한 지름길",
                        "길 표지",
                    ),
                ),
            ),
            _beat(
                "wolf-at-door",
                "빨간 모자의 직전 답을 들은 늑대가 먼저 할머니 집에 가지만 비밀 신호를 몰라요.",
                "문밖 손님이 진짜인지 알아볼 둘만의 비밀 암호나 질문",
                ("SAFETY_CHECK", "SECRET_SIGNAL"),
                (
                    _plan(
                        "빨간 모자가 직전 답의 재치 있는 말을 건네고 "
                        "정해진 길을 지키자 늑대가 다른 길로 달려가요.",
                        ("red_hood", "wolf"),
                        "직전 답의 재치 있는 한마디",
                        "늑대의 앞선 도착",
                    ),
                    _plan(
                        "늑대가 할머니 집 문을 빨간 모자인 척 두드려요.",
                        ("grandmother", "wolf"),
                        "늑대의 흉내",
                    ),
                    _plan(
                        "할머니는 비밀 신호가 다르다는 것을 알아채고 문을 열지 않아요.",
                        ("grandmother", "wolf"),
                        "비밀 신호 확인",
                    ),
                    _plan(
                        "할머니가 문 안에서 안전하게 상대를 확인할 질문을 생각해요.",
                        ("grandmother", "wolf"),
                        "안전 확인",
                    ),
                ),
            ),
            _beat(
                "tracks-and-warning",
                "빨간 모자가 낯선 발자국을 발견하고 숲지기에게 도움을 청하기 직전까지 가요.",
                "수상한 발자국을 본 빨간 모자가 숲지기에게 가장 먼저 알려 줄 단서",
                ("HELP_REQUEST", "OBSERVATION"),
                (
                    _plan(
                        "할머니가 고른 확인 질문을 하자 늑대가 답하지 못하고 집 옆에 숨어요.",
                        ("grandmother", "wolf"),
                        "확인 질문의 효과",
                    ),
                    _plan(
                        "빨간 모자가 집 근처에서 자기 발자국보다 큰 흔적을 발견해요.",
                        ("red_hood", "wolf"),
                        "큰 발자국",
                    ),
                    _plan(
                        "빨간 모자는 혼자 문으로 가지 않고 가까운 숲지기 초소로 향해요.",
                        ("red_hood", "forest_keeper"),
                        "안전한 판단",
                    ),
                    _plan(
                        "빨간 모자가 숲지기 초소에 도착해 "
                        "여러 단서 중 무엇부터 말할지 숨을 골라요.",
                        ("red_hood", "forest_keeper"),
                        "단서를 말하기 직전",
                    ),
                ),
            ),
            _beat(
                "safe-reunion",
                "직전 답의 단서를 들은 숲지기가 빨간 모자와 함께 할머니를 안전하게 만나요.",
                "숨어 있는 늑대의 장난을 멈추게 할 셋의 합동 한마디나 작전",
                ("BOUNDARY_PHRASE", "TEAM_ACTION"),
                (
                    _plan(
                        "빨간 모자가 직전 답의 단서를 먼저 말하자 "
                        "숲지기가 함께 할머니 집을 살펴요.",
                        ("red_hood", "forest_keeper"),
                        "직전 답의 첫 단서",
                        "숲지기의 도움",
                    ),
                    _plan(
                        "빨간 모자가 정확한 비밀 신호를 보내고 할머니가 문을 열어요.",
                        ("red_hood", "grandmother", "forest_keeper"),
                        "안전한 재회",
                    ),
                    _plan(
                        "세 사람은 집 옆에 숨은 늑대의 꼬리를 발견해요.",
                        ("red_hood", "grandmother", "forest_keeper", "wolf"),
                        "숨은 늑대 발견",
                    ),
                    _plan(
                        "세 사람은 늑대의 장난을 멈추게 할 한마디나 작전을 "
                        "정하려고 서로 눈을 맞춰요.",
                        ("red_hood", "grandmother", "forest_keeper", "wolf"),
                        "합동 대응 직전",
                    ),
                ),
            ),
            _beat(
                "wolf-apology",
                "직전 답의 합동 대응에 늑대가 장난을 인정하고 사과한 뒤 고칠 일을 물어요.",
                "늑대가 장난을 되돌리려고 가장 먼저 할 일",
                ("APOLOGY", "REPAIR_ACTION"),
                (
                    _plan(
                        "세 사람이 직전 답의 한마디나 작전을 실행하자 "
                        "늑대가 숨어 있던 곳에서 나와요.",
                        ("red_hood", "grandmother", "forest_keeper", "wolf"),
                        "직전 답의 합동 대응",
                        "늑대의 등장",
                    ),
                    _plan(
                        "늑대는 남을 속인 장난이 모두를 불안하게 했다는 것을 들어요.",
                        ("grandmother", "wolf"),
                        "행동의 결과",
                    ),
                    _plan(
                        "늑대가 잘못을 인정하고 다시는 문 신호를 흉내 내지 않겠다고 해요.",
                        ("red_hood", "grandmother", "wolf"),
                        "늑대의 사과",
                    ),
                    _plan(
                        "늑대가 흩어진 바구니 물건을 바라보며 잘못을 되돌릴 첫 일을 묻고 기다려요.",
                        ("red_hood", "grandmother", "wolf"),
                        "고칠 일을 고르기 직전",
                    ),
                ),
            ),
            _beat(
                "grandmother-tea",
                "늑대가 직전 답의 일로 장난을 고치고 모두 선물을 나누며 안전 약속을 기억해요.",
                None,
                (),
                (
                    _plan(
                        "늑대가 직전 답의 일을 먼저 해 바구니를 바로잡자 "
                        "할머니가 지켜야 할 경계를 알려 줘요.",
                        ("red_hood", "grandmother", "wolf"),
                        "직전 답의 고치는 일",
                        "안전한 경계",
                    ),
                    _plan(
                        "빨간 모자가 처음 고른 선물을 바구니에서 꺼내 할머니에게 드려요.",
                        ("red_hood", "grandmother"),
                        "선물 전달",
                    ),
                    _plan(
                        "숲지기와 늑대도 문밖 탁자에서 차와 빵을 함께 나눠요.",
                        ("red_hood", "grandmother", "forest_keeper", "wolf"),
                        "안전한 나눔",
                    ),
                    _plan(
                        "빨간 모자는 관찰하고 도움을 청한 덕분에 약속을 지켰다고 느껴요.",
                        ("red_hood", "grandmother"),
                        "안전 약속의 가치",
                    ),
                ),
                concluding=True,
            ),
        ),
    ),
    ServiceStoryFixture(
        template_id=1008,
        version=2,
        title="브레멘 음악대",
        source_title="그림 형제 동화 「브레멘 음악대」",
        source_note=SOURCE_NOTE,
        description="서로 다른 네 동물이 힘을 합쳐 음악가의 꿈을 이루어요.",
        context=(
            "네 동물이 무대를 찾아가는 신나는 음악 여행이에요. 네 동물의 서로 다른 "
            "목소리가 합주에서 꼭 필요한 재능이 되고, 숲속 집의 소란스러운 "
            "사람들은 음악에 놀라 달아날 뿐 다치지 않아요."
        ),
        initial_summary="이야기는 당나귀가 먼 곳의 음악을 듣기 전부터 시작해요.",
        characters=(
            _character(
                "donkey",
                "당나귀",
                "주인공",
                "힘이 좋다",
                "낮고 든든한 소리를 낸다",
                "친구들을 차례로 초대한다",
            ),
            _character(
                "dog",
                "개",
                "친구",
                "충직하다",
                "박자를 잘 맞춘다",
            ),
            _character(
                "cat",
                "고양이",
                "친구",
                "민첩하다",
                "맑고 높은 소리를 낸다",
            ),
            _character(
                "rooster",
                "수탉",
                "친구",
                "목소리가 크다",
                "아침을 알리는 소리를 낸다",
            ),
        ),
        beats=(
            _beat(
                "donkey-dreams",
                "당나귀가 음악가의 꿈을 품고 브레멘으로 떠나요.",
                "당나귀가 연주하고 싶은 소리나 악기",
                ("INSTRUMENT", "MUSIC_ROLE"),
                (
                    _plan(
                        "당나귀가 일을 마친 뒤 멀리서 들리는 흥겨운 음악을 들어요.",
                        ("donkey",),
                        "멀리서 들리는 음악",
                    ),
                    _plan(
                        "당나귀가 자기의 낮고 힘찬 목소리를 음악에 써 보고 싶어 해요.",
                        ("donkey",),
                        "당나귀의 음악 꿈",
                    ),
                    _plan(
                        "당나귀가 음악가들이 모인다는 브레멘 이야기를 떠올려요.",
                        ("donkey",),
                        "브레멘",
                    ),
                    _plan(
                        "당나귀가 작은 짐을 챙겨 브레멘으로 여행을 시작해요.",
                        ("donkey",),
                        "여행 출발",
                    ),
                ),
            ),
            _beat(
                "dog-joins",
                "당나귀의 음악 선택을 반영하고 길에서 만난 개가 박자 담당으로 합류해요.",
                "개가 합주해 보고 싶게 만드는 당나귀의 신나는 초대 한마디",
                ("DIALOGUE", "MUSIC_ROLE"),
                (
                    _plan(
                        "당나귀가 고른 소리를 연습하며 브레멘 길을 걸어요.",
                        ("donkey",),
                        "선택한 음악",
                    ),
                    _plan(
                        "길가에서 지친 개가 발로 일정한 박자를 두드리는 모습을 봐요.",
                        ("donkey", "dog"),
                        "개의 박자",
                    ),
                    _plan(
                        "당나귀가 개의 박자가 자기 소리와 잘 맞는다고 알려 줘요.",
                        ("donkey", "dog"),
                        "어울리는 소리",
                    ),
                    _plan(
                        "개가 초대를 기다리며 브레멘 길을 바라봐요.",
                        ("donkey", "dog"),
                        "음악대 초대",
                    ),
                ),
            ),
            _beat(
                "cat-joins",
                "개가 음악대에 합류하고 두 친구가 고양이의 높은 소리를 발견해요.",
                "고양이가 합주에서 맡을 소리",
                ("MUSIC_ROLE", "MELODY"),
                (
                    _plan(
                        "당나귀가 직전 초대의 말을 건네고 개가 음악대에 들어와요.",
                        ("donkey", "dog"),
                        "개의 합류",
                    ),
                    _plan(
                        "두 친구가 박자를 맞추며 걷다가 높은 담 위의 고양이를 만나요.",
                        ("donkey", "dog", "cat"),
                        "고양이와 만남",
                    ),
                    _plan(
                        "고양이가 맑고 긴 목소리로 두 친구의 리듬에 화음을 넣어요.",
                        ("donkey", "dog", "cat"),
                        "고양이의 화음",
                    ),
                    _plan(
                        "세 친구가 고양이에게 어울릴 음악 역할을 함께 생각해요.",
                        ("donkey", "dog", "cat"),
                        "고양이의 역할",
                    ),
                ),
            ),
            _beat(
                "rooster-joins",
                "고양이가 합류하고 수탉의 큰 목소리가 음악대의 마지막 소리가 돼요.",
                "네 동물 음악대의 이름",
                ("TEAM_NAME", "START_SIGNAL"),
                (
                    _plan(
                        "고양이가 고른 역할로 노래하며 당나귀와 개를 따라가요.",
                        ("donkey", "dog", "cat"),
                        "고양이의 합류",
                    ),
                    _plan(
                        "세 친구가 지붕 위에서 큰 목소리로 노래하는 수탉을 발견해요.",
                        ("donkey", "dog", "cat", "rooster"),
                        "수탉과 만남",
                    ),
                    _plan(
                        "수탉의 소리가 멀리까지 퍼지자 모두가 공연의 시작 소리로 좋다고 해요.",
                        ("donkey", "dog", "cat", "rooster"),
                        "공연 시작 소리",
                    ),
                    _plan(
                        "네 친구가 한 팀이 되어 함께 부를 음악대 이름을 정하려 해요.",
                        ("donkey", "dog", "cat", "rooster"),
                        "음악대 결성",
                    ),
                ),
            ),
            _beat(
                "forest-rehearsal",
                "이름을 얻은 음악대가 숲에서 연습하며 엇갈리는 박자를 맞춰요.",
                "네 소리를 차례로 더할 연주 순서",
                ("PERFORMANCE_ORDER", "SONG_STYLE"),
                (
                    _plan(
                        "네 친구가 고른 음악대 이름을 외치며 숲길로 들어가요.",
                        ("donkey", "dog", "cat", "rooster"),
                        "음악대 이름",
                    ),
                    _plan(
                        "처음 함께 연주하자 큰 소리와 높은 소리가 서로 엉켜요.",
                        ("donkey", "dog", "cat", "rooster"),
                        "엇갈린 합주",
                    ),
                    _plan(
                        "개가 박자를 세고 당나귀와 고양이와 수탉이 차례를 기다려요.",
                        ("donkey", "dog", "cat", "rooster"),
                        "박자 맞추기",
                    ),
                    _plan(
                        "네 친구가 가장 멋진 연주 순서를 고르기 위해 다시 준비해요.",
                        ("donkey", "dog", "cat", "rooster"),
                        "연주 순서",
                    ),
                ),
            ),
            _beat(
                "lighted-house",
                "합주가 좋아지고 네 친구가 숲속의 밝은 집과 수상한 소란을 발견해요.",
                "불 켜진 집 안의 소란을 들키지 않고 살펴볼 음악대 작전",
                ("SAFE_CHECK", "TEAM_POSITION"),
                (
                    _plan(
                        "네 동물이 직전 순서대로 소리를 더해 처음으로 멋진 합주를 완성해요.",
                        ("donkey", "dog", "cat", "rooster"),
                        "성공한 합주",
                    ),
                    _plan(
                        "밤이 되자 나무 사이에서 밝은 창문과 시끄러운 웃음소리가 보여요.",
                        ("donkey", "dog", "cat", "rooster"),
                        "숲속의 밝은 집",
                    ),
                    _plan(
                        "네 친구가 창문 가까이 다가가 욕심 많은 사람들이 음식을 차지한 것을 봐요.",
                        ("donkey", "dog", "cat", "rooster"),
                        "집 안의 소란",
                    ),
                    _plan(
                        "동물들은 다치지 않고 집을 살필 안전한 위치와 순서를 정하려 해요.",
                        ("donkey", "dog", "cat", "rooster"),
                        "안전한 정찰",
                    ),
                ),
            ),
            _beat(
                "surprise-concert",
                "네 친구가 창밖에서 멋진 합주를 해 소란스러운 사람들이 놀라 달아나요.",
                "집을 되찾은 음악대가 가장 먼저 꾸미거나 연주할 것",
                ("TEAM_ACTION", "HOUSE_USE"),
                (
                    _plan(
                        "네 친구가 직전 선택대로 안전한 자리를 잡고 창밖에 모여요.",
                        ("donkey", "dog", "cat", "rooster"),
                        "안전한 자리",
                    ),
                    _plan(
                        "당나귀부터 수탉까지 연습한 순서로 힘찬 음악을 시작해요.",
                        ("donkey", "dog", "cat", "rooster"),
                        "창밖 합주",
                    ),
                    _plan(
                        "집 안 사람들은 처음 듣는 멋진 큰 소리에 놀라 빈손으로 달아나요.",
                        ("donkey", "dog", "cat", "rooster"),
                        "소란스러운 사람들의 퇴장",
                    ),
                    _plan(
                        "네 친구가 조용해진 집을 확인하고 새로운 보금자리를 상상해요.",
                        ("donkey", "dog", "cat", "rooster"),
                        "새 보금자리",
                    ),
                ),
            ),
            _beat(
                "music-home",
                "네 친구가 숲속 집을 음악의 집으로 꾸미고 첫 공연을 열어요.",
                None,
                (),
                (
                    _plan(
                        "네 친구가 직전 선택대로 집을 청소하거나 안전을 먼저 확인해요.",
                        ("donkey", "dog", "cat", "rooster"),
                        "새 집 정리",
                    ),
                    _plan(
                        "당나귀와 개와 고양이와 수탉이 각자의 연습 자리를 만들어요.",
                        ("donkey", "dog", "cat", "rooster"),
                        "음악 연습실",
                    ),
                    _plan(
                        "네 친구의 첫 공연 소리가 숲길 멀리까지 즐겁게 퍼져요.",
                        ("donkey", "dog", "cat", "rooster"),
                        "첫 공연",
                    ),
                    _plan(
                        "브레멘으로 가던 네 동물은 서로와 함께 있는 곳이 음악의 집임을 알아요.",
                        ("donkey", "dog", "cat", "rooster"),
                        "음악대의 집",
                        "함께 이룬 꿈",
                    ),
                ),
                concluding=True,
            ),
        ),
    ),
    ServiceStoryFixture(
        template_id=1009,
        version=2,
        title="세 마리 아기 돼지",
        source_title="영국 전래 이야기 「세 마리 아기 돼지」의 안전한 재구성",
        source_note=SOURCE_NOTE,
        description="세 돼지가 서로 다른 집을 짓고 협력의 힘을 배워요.",
        context=(
            "세 돼지가 서로 다른 집을 짓는 이야기예요. 짚집, 나무집, 벽돌집의 "
            "핵심 구조는 지키되 누구도 다치거나 잡아먹히지 않아요. 늑대는 "
            "끝에 남의 집을 무너뜨리면 안 된다는 경계를 배워요."
        ),
        initial_summary="이야기는 세 돼지가 집 지을 빈터를 고르기 전부터 시작해요.",
        characters=(
            _character(
                "pig_one",
                "첫째 돼지",
                "주인공",
                "상상력이 좋다",
                "빨리 완성하는 것을 좋아한다",
            ),
            _character(
                "pig_two",
                "둘째 돼지",
                "주인공",
                "손재주가 좋다",
                "친구를 잘 돕는다",
            ),
            _character(
                "pig_three",
                "셋째 돼지",
                "주인공",
                "꼼꼼하다",
                "오래 걸려도 튼튼하게 만든다",
            ),
            _character(
                "wolf",
                "늑대",
                "갈등 인물",
                "바람을 세게 분다",
                "누구도 다치게 하지 않는다",
                "마지막에는 남의 경계를 존중한다",
            ),
        ),
        beats=(
            _beat(
                "choose-building-sites",
                "세 돼지가 가까운 빈터에 각자의 집을 짓기로 해요.",
                "세 집을 이어 줄 공통 표지나 장식",
                ("DECORATION", "PATH_MARKER"),
                (
                    _plan(
                        "세 돼지가 햇빛이 잘 드는 넓은 빈터를 발견해요.",
                        ("pig_one", "pig_two", "pig_three"),
                        "집 지을 빈터",
                    ),
                    _plan(
                        "첫째는 짚을, 둘째는 나무를, 셋째는 벽돌을 고르기로 해요.",
                        ("pig_one", "pig_two", "pig_three"),
                        "세 가지 재료",
                    ),
                    _plan(
                        "세 돼지가 집은 달라도 가까이 지어 서로 돕기로 약속해요.",
                        ("pig_one", "pig_two", "pig_three"),
                        "가까운 세 집",
                    ),
                    _plan(
                        "세 형제가 길을 잃지 않도록 같은 표지를 달고 싶어 해요.",
                        ("pig_one", "pig_two", "pig_three"),
                        "공통 표지",
                    ),
                ),
            ),
            _beat(
                "three-houses",
                "장식 선택을 반영하고 세 돼지가 서로 다른 속도로 집을 완성해요.",
                "먼저 끝낸 돼지들이 셋째 집에 보탤 도구나 도움",
                ("HELP_METHOD", "BUILDING_TOOL"),
                (
                    _plan(
                        "세 돼지가 고른 표지를 각자의 집터 앞에 나란히 세워요.",
                        ("pig_one", "pig_two", "pig_three"),
                        "공통 표지 반영",
                    ),
                    _plan(
                        "첫째가 가벼운 짚을 묶어 가장 먼저 포근한 집을 완성해요.",
                        ("pig_one",),
                        "짚집",
                    ),
                    _plan(
                        "둘째가 나무판을 맞춰 단단한 문이 있는 집을 완성해요.",
                        ("pig_two",),
                        "나무집",
                    ),
                    _plan(
                        "셋째가 벽돌을 쌓는 동안 두 형제가 어떻게 도울지 생각해요.",
                        ("pig_one", "pig_two", "pig_three"),
                        "벽돌집 짓기",
                        "서로 돕기",
                    ),
                ),
            ),
            _beat(
                "wolf-and-straw",
                "세 집이 완성된 뒤 늑대의 바람에 짚집이 흔들리고 첫째가 안전하게 피해야 해요.",
                "첫째가 둘째에게 보낼 빠른 도움 신호",
                ("SIGNAL", "SAFE_ACTION"),
                (
                    _plan(
                        "첫째와 둘째가 직전 방법으로 벽돌을 옮겨 셋째의 집도 완성해요.",
                        ("pig_one", "pig_two", "pig_three"),
                        "세 집 완성",
                    ),
                    _plan(
                        "숲에서 온 늑대가 빈터의 세 집을 보고 큰 바람을 자랑해요.",
                        ("pig_one", "wolf"),
                        "늑대의 등장",
                    ),
                    _plan(
                        "늑대가 세게 불자 가벼운 짚집의 벽이 흔들리기 시작해요.",
                        ("pig_one", "wolf"),
                        "흔들리는 짚집",
                    ),
                    _plan(
                        "첫째는 다치지 않았지만 집 밖으로 나가 둘째에게 신호를 보내야 해요.",
                        ("pig_one", "pig_two", "wolf"),
                        "안전한 대피",
                        "도움 신호",
                    ),
                ),
            ),
            _beat(
                "wolf-and-wood",
                "첫째가 나무집으로 피하고 늑대가 다시 바람을 불어 둘째 집도 흔들려요.",
                "늑대의 바람을 피해 벽돌집까지 갈 기발한 이동 작전",
                ("HELPFUL_ITEM", "SAFE_ROUTE"),
                (
                    _plan(
                        "첫째가 직전 신호를 보내고 둘째가 문을 열어 함께 나무집에 들어가요.",
                        ("pig_one", "pig_two", "wolf"),
                        "첫째의 안전",
                    ),
                    _plan(
                        "늑대가 나무집에도 바람을 불자 창문과 지붕이 크게 흔들려요.",
                        ("pig_one", "pig_two", "wolf"),
                        "흔들리는 나무집",
                    ),
                    _plan(
                        "두 돼지가 셋째의 벽돌집으로 이어진 표지 길을 발견해요.",
                        ("pig_one", "pig_two", "pig_three"),
                        "벽돌집으로 가는 길",
                    ),
                    _plan(
                        "두 돼지가 늑대의 바람을 피해 안전하게 이동할 방법을 정해야 해요.",
                        ("pig_one", "pig_two", "wolf"),
                        "두 번째 대피",
                    ),
                ),
            ),
            _beat(
                "brick-house-holds",
                "세 돼지가 벽돌집에 모이고 늑대의 바람에도 집이 튼튼히 버텨요.",
                "벽돌집 창문에서 세 돼지가 늑대에게 함께 외칠 한마디",
                ("BOUNDARY_PHRASE", "TEAM_ACTION"),
                (
                    _plan(
                        "첫째와 둘째가 직전 방법으로 벽돌집에 무사히 도착해요.",
                        ("pig_one", "pig_two", "pig_three", "wolf"),
                        "벽돌집 도착",
                    ),
                    _plan(
                        "늑대가 가장 큰 숨으로 불어도 벽돌집은 단단히 서 있어요.",
                        ("pig_one", "pig_two", "pig_three", "wolf"),
                        "튼튼한 벽돌집",
                    ),
                    _plan(
                        "세 돼지가 각자 문과 창문과 지붕을 살피며 함께 집을 지켜요.",
                        ("pig_one", "pig_two", "pig_three"),
                        "협력",
                    ),
                    _plan(
                        "세 돼지가 늑대에게 남의 집을 흔드는 일을 멈추라고 말하려 해요.",
                        ("pig_one", "pig_two", "pig_three", "wolf"),
                        "경계 세우기",
                    ),
                ),
            ),
            _beat(
                "wolf-learns",
                "늑대가 행동의 결과를 알고 사과하며 무너진 집을 다시 짓는 데 힘을 보태요.",
                "늑대의 센 바람으로 집을 고칠 때 할 수 있는 특별한 일",
                ("REPAIR_ACTION", "APOLOGY"),
                (
                    _plan(
                        "세 돼지가 직전의 단호한 말을 전하자 늑대가 바람을 멈춰요.",
                        ("pig_one", "pig_two", "pig_three", "wolf"),
                        "늑대의 멈춤",
                    ),
                    _plan(
                        "늑대가 흔들린 짚집과 나무집을 보고 자기 행동의 결과를 알아요.",
                        ("pig_one", "pig_two", "pig_three", "wolf"),
                        "행동의 결과",
                    ),
                    _plan(
                        "늑대가 세 돼지에게 사과하고 집을 고치는 일을 돕겠다고 해요.",
                        ("pig_one", "pig_two", "pig_three", "wolf"),
                        "늑대의 사과",
                    ),
                    _plan(
                        "세 돼지가 늑대의 힘을 안전하게 쓸 수 있는 일을 함께 골라요.",
                        ("pig_one", "pig_two", "pig_three", "wolf"),
                        "힘의 바른 쓰임",
                    ),
                ),
            ),
            _beat(
                "strong-neighborhood",
                "세 돼지와 늑대가 안전 규칙을 지키며 빈터를 튼튼한 마을로 만들어요.",
                None,
                (),
                (
                    _plan(
                        "늑대가 직전 선택대로 무거운 재료를 옮기거나 먼지를 불어내요.",
                        ("pig_one", "pig_two", "pig_three", "wolf"),
                        "힘으로 돕기",
                    ),
                    _plan(
                        "첫째와 둘째가 배운 방법으로 짚집과 나무집을 더 튼튼하게 고쳐요.",
                        ("pig_one", "pig_two", "pig_three"),
                        "집 보강하기",
                    ),
                    _plan(
                        "세 집 앞에는 처음 고른 같은 표지가 다시 나란히 서요.",
                        ("pig_one", "pig_two", "pig_three"),
                        "공통 표지",
                    ),
                    _plan(
                        "세 돼지와 늑대는 힘과 재료를 안전하게 쓰는 이웃이 돼요.",
                        ("pig_one", "pig_two", "pig_three", "wolf"),
                        "튼튼한 마을",
                        "안전한 이웃",
                    ),
                ),
                concluding=True,
            ),
        ),
    ),
    ServiceStoryFixture(
        template_id=1010,
        version=2,
        title="미운 오리 새끼",
        source_title="안데르센 동화 「미운 오리 새끼」의 비폭력 재구성",
        source_note=SOURCE_NOTE,
        description="회색 새끼가 긴 여행 끝에 자기 모습을 이해하고 친구를 만나요.",
        context=(
            "회색 아기 새가 자기 모습을 찾아가는 성장 이야기예요. 주인공은 실제로 어린 백조이며 "
            "외모를 조롱하는 대사는 쓰지 않아요. 낯선 시선과 외로움, 계절의 "
            "어려움을 지나 자기 모습을 받아들이는 과정을 보여 줘요."
        ),
        initial_summary="이야기는 연못가 둥지의 마지막 큰 알이 흔들리기 전부터 시작해요.",
        characters=(
            _character(
                "young_swan",
                "회색 새끼",
                "주인공",
                "몸빛이 회색이다",
                "마음이 다정하다",
                "자라서 백조가 된다",
            ),
            _character(
                "mother_duck",
                "엄마 오리",
                "조력자",
                "새끼들을 보호한다",
                "수영을 차분히 가르친다",
            ),
            _character(
                "wild_goose",
                "기러기",
                "친구",
                "여행길을 잘 안다",
                "낯선 친구를 따뜻하게 돕는다",
            ),
            _character(
                "swans",
                "백조들",
                "조력자",
                "차분하다",
                "회색 새끼를 반갑게 맞는다",
            ),
        ),
        beats=(
            _beat(
                "last-egg",
                "마지막 큰 알에서 회색 새끼가 태어나고 엄마 오리가 따뜻하게 맞아요.",
                "알에서 나온 회색 새끼에게 엄마 오리가 건넬 첫마디",
                ("WELCOME_PHRASE", "DIALOGUE"),
                (
                    _plan(
                        "연못가 둥지에서 오리 알들이 차례로 흔들리고 새끼들이 태어나요.",
                        ("mother_duck",),
                        "연못가 둥지",
                        "알의 부화",
                    ),
                    _plan(
                        "마지막 큰 알이 늦게 갈라지고 몸집이 큰 회색 새끼가 나와요.",
                        ("young_swan", "mother_duck"),
                        "회색 새끼의 탄생",
                    ),
                    _plan(
                        "회색 새끼가 낯선 바깥을 바라보며 엄마 오리 곁으로 다가가요.",
                        ("young_swan", "mother_duck"),
                        "첫 만남",
                    ),
                    _plan(
                        "엄마 오리가 새끼를 안심시킬 따뜻한 환영 말을 생각해요.",
                        ("young_swan", "mother_duck"),
                        "환영",
                    ),
                ),
            ),
            _beat(
                "first-swim",
                "환영을 들은 회색 새끼가 엄마 오리와 첫 수영에 도전해요.",
                "회색 새끼가 물에서 자기만의 균형을 잡을 방법",
                ("MOVEMENT_STYLE", "SELF_TALK"),
                (
                    _plan(
                        "엄마 오리가 직전 환영 말을 건네고 회색 새끼가 용기를 내요.",
                        ("young_swan", "mother_duck"),
                        "환영 말 반영",
                    ),
                    _plan(
                        "엄마 오리가 연못 가장자리에서 발을 움직이는 법을 보여 줘요.",
                        ("young_swan", "mother_duck"),
                        "첫 수영 수업",
                    ),
                    _plan(
                        "회색 새끼가 몸이 커서 처음에는 물 위에서 조금 흔들려요.",
                        ("young_swan",),
                        "흔들리는 몸",
                    ),
                    _plan(
                        "회색 새끼가 자기에게 맞는 헤엄 방법을 찾아 다시 나아가려 해요.",
                        ("young_swan", "mother_duck"),
                        "자기만의 헤엄",
                    ),
                ),
            ),
            _beat(
                "different-reflection",
                "회색 새끼가 다른 모습을 알아차리지만 자신의 좋은 점도 발견해요.",
                "회색 새끼의 긴 목과 큰 발에 숨은 장점",
                ("CHEER_PHRASE", "STRENGTH_DETAIL"),
                (
                    _plan(
                        "회색 새끼가 직전 방법으로 물의 균형을 잡고 넓은 곳까지 헤엄쳐요.",
                        ("young_swan", "mother_duck"),
                        "첫 수영 성공",
                    ),
                    _plan(
                        "잔잔한 물에 비친 회색 새끼의 목과 날개가 다른 새끼보다 길어 보여요.",
                        ("young_swan",),
                        "다른 모습",
                    ),
                    _plan(
                        "낯선 새들이 바라보자 회색 새끼가 자기 모습이 이상한지 걱정해요.",
                        ("young_swan",),
                        "낯선 시선",
                        "걱정",
                    ),
                    _plan(
                        "엄마 오리가 다른 모습에도 좋은 점이 있다고 알려 주려 해요.",
                        ("young_swan", "mother_duck"),
                        "다름의 의미",
                    ),
                ),
            ),
            _beat(
                "new-journey",
                "회색 새끼가 자기에게 맞는 곳을 알아보기 위해 안전한 여행을 시작해요.",
                "여행길에 챙길 작은 물건이나 기억",
                ("TRAVEL_ITEM", "MEMORY"),
                (
                    _plan(
                        "엄마 오리가 직전의 말을 건네고 회색 새끼의 긴 목과 힘찬 헤엄을 칭찬해요.",
                        ("young_swan", "mother_duck"),
                        "자기 장점 발견",
                    ),
                    _plan(
                        "회색 새끼가 더 넓은 연못의 새들을 만나 자기 모습을 알아보고 싶어 해요.",
                        ("young_swan",),
                        "새 여행의 이유",
                    ),
                    _plan(
                        "엄마 오리가 안전한 물길과 돌아올 수 있는 길표지를 알려 줘요.",
                        ("young_swan", "mother_duck"),
                        "안전한 여행길",
                    ),
                    _plan(
                        "회색 새끼가 여행에 힘이 될 것을 하나 챙기려 해요.",
                        ("young_swan", "mother_duck"),
                        "여행 준비",
                    ),
                ),
            ),
            _beat(
                "goose-friend",
                "회색 새끼가 기러기를 만나 계절과 길을 배우며 친구가 돼요.",
                "길을 잘 아는 기러기에게 회색 새끼가 가장 먼저 물어볼 것",
                ("DIALOGUE", "HELP_REQUEST"),
                (
                    _plan(
                        "회색 새끼가 직전 선택을 챙기고 엄마 오리에게 인사한 뒤 물길을 떠나요.",
                        ("young_swan", "mother_duck"),
                        "여행 출발",
                    ),
                    _plan(
                        "넓은 갈대밭에서 길을 찾던 회색 새끼가 쉬고 있는 기러기를 만나요.",
                        ("young_swan", "wild_goose"),
                        "기러기와 만남",
                    ),
                    _plan(
                        "기러기가 곧 찬 계절이 오니 따뜻한 갈대숲을 찾으라고 알려 줘요.",
                        ("young_swan", "wild_goose"),
                        "겨울 소식",
                    ),
                    _plan(
                        "회색 새끼가 안전한 길을 알기 위해 기러기에게 도움을 청하려 해요.",
                        ("young_swan", "wild_goose"),
                        "길 묻기",
                    ),
                ),
            ),
            _beat(
                "winter-shelter",
                "기러기의 안내로 회색 새끼가 찬 계절을 안전하게 보내요.",
                "겨울 갈대숲을 포근한 잠자리로 만들 물건이나 행동",
                ("SAFE_ACTION", "COMFORT_ITEM"),
                (
                    _plan(
                        "기러기가 직전 질문에 답하며 바람이 약한 갈대숲 길을 알려 줘요.",
                        ("young_swan", "wild_goose"),
                        "기러기의 안내",
                    ),
                    _plan(
                        "찬 바람이 불자 회색 새끼가 두꺼운 갈대 사이에 안전한 자리를 찾아요.",
                        ("young_swan",),
                        "겨울 갈대숲",
                    ),
                    _plan(
                        "회색 새끼가 혼자라고 느끼지만 엄마 오리와 기러기의 말을 기억해요.",
                        ("young_swan",),
                        "외로움",
                        "따뜻한 기억",
                    ),
                    _plan(
                        "회색 새끼가 추운 날을 견딜 자기만의 편안한 방법을 고르려 해요.",
                        ("young_swan",),
                        "겨울나기",
                    ),
                ),
            ),
            _beat(
                "spring-reflection",
                "봄이 오고 자란 회색 새끼가 물에 비친 백조의 모습을 발견해요.",
                "처음 만난 백조들에게 건넬 인사",
                ("GREETING", "DIALOGUE"),
                (
                    _plan(
                        "회색 새끼가 직전 방법으로 몸을 지키고 겨울을 무사히 보내요.",
                        ("young_swan",),
                        "겨울나기 성공",
                    ),
                    _plan(
                        "봄빛이 비치자 길어진 날개와 하얘진 깃털이 물에 보여요.",
                        ("young_swan",),
                        "봄",
                        "달라진 모습",
                    ),
                    _plan(
                        "회색 새끼가 물속의 모습이 아름다운 백조와 같다는 것을 알아요.",
                        ("young_swan", "swans"),
                        "백조의 모습",
                        "자기 발견",
                    ),
                    _plan(
                        "가까이 온 백조들이 반갑게 바라보자 회색 새끼가 인사하려 해요.",
                        ("young_swan", "swans"),
                        "백조들과 첫 만남",
                    ),
                ),
            ),
            _beat(
                "swan-welcome",
                "백조들이 주인공을 환영하고 주인공은 자기 모습과 여정을 받아들여요.",
                None,
                (),
                (
                    _plan(
                        "회색 새끼가 직전 인사를 건네고 백조들이 따뜻하게 답해요.",
                        ("young_swan", "swans"),
                        "백조들의 환영",
                    ),
                    _plan(
                        "백조들은 주인공도 같은 백조이며 긴 여행을 잘 견뎠다고 알려 줘요.",
                        ("young_swan", "swans"),
                        "정체성",
                        "여정의 인정",
                    ),
                    _plan(
                        "이제 자란 백조가 엄마 오리와 기러기에게 기쁜 소식을 전해요.",
                        ("young_swan", "mother_duck", "wild_goose"),
                        "친구들과 재회",
                    ),
                    _plan(
                        "어린 백조는 자기만의 모습과 마음을 사랑하며 새 친구들과 헤엄쳐요.",
                        ("young_swan", "swans"),
                        "자기 수용",
                        "새로운 우정",
                    ),
                ),
                concluding=True,
            ),
        ),
    ),
)


def get_story_fixture(template_id: int) -> ServiceStoryFixture:
    for story in STORY_CATALOG:
        if story.template_id == template_id:
            return story
    raise KeyError(f"Unknown service story template: {template_id}")


__all__ = [
    "SOURCE_NOTE",
    "STORY_CATALOG",
    "ServiceStoryFixture",
    "StoryBeatFixture",
    "StoryCharacterFixture",
    "StoryPagePlanFixture",
    "get_story_fixture",
]
