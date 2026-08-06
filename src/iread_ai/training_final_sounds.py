from __future__ import annotations

from .personalization.hangul import decompose_text

REPRESENTATIVE_FINAL_SOUNDS = ("ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅇ")

_SIMPLE_FINAL_SOUND = {
    "ㄱ": "ㄱ",
    "ㄲ": "ㄱ",
    "ㅋ": "ㄱ",
    "ㄴ": "ㄴ",
    "ㄷ": "ㄷ",
    "ㅅ": "ㄷ",
    "ㅆ": "ㄷ",
    "ㅈ": "ㄷ",
    "ㅊ": "ㄷ",
    "ㅌ": "ㄷ",
    "ㅎ": "ㄷ",
    "ㄹ": "ㄹ",
    "ㅁ": "ㅁ",
    "ㅂ": "ㅂ",
    "ㅍ": "ㅂ",
    "ㅇ": "ㅇ",
}

_COMPLEX_FINAL_SOUND = {
    "ㄳ": "ㄱ",
    "ㄵ": "ㄴ",
    "ㄶ": "ㄴ",
    "ㄺ": "ㄱ",
    "ㄻ": "ㅁ",
    "ㄽ": "ㄹ",
    "ㄾ": "ㄹ",
    "ㄿ": "ㅂ",
    "ㅀ": "ㄹ",
    "ㅄ": "ㅂ",
}


def representative_final_sound(
    text: str,
    *,
    pronunciation: str | None = None,
) -> str | None:
    pronounced = decompose_text(pronunciation or "")
    written = decompose_text(text)
    syllables = pronounced or written
    if not syllables:
        return None
    coda = syllables[-1].coda
    if not coda:
        return None
    if coda in _SIMPLE_FINAL_SOUND:
        return _SIMPLE_FINAL_SOUND[coda]
    return _COMPLEX_FINAL_SOUND.get(coda)


__all__ = ["REPRESENTATIVE_FINAL_SOUNDS", "representative_final_sound"]
