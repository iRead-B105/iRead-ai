from iread_ai.training_feature_compatibility import feature_is_compatible


def test_basic_syllable_build_rejects_coda_targets() -> None:
    assert feature_is_compatible("BASIC_SYLLABLE_BUILD", "SYLLABLE.CV")
    assert not feature_is_compatible("BASIC_SYLLABLE_BUILD", "SYLLABLE.CVC")
    assert not feature_is_compatible(
        "BASIC_SYLLABLE_BUILD", "GRAPHEME.CODA.SIMPLE.ㄱ"
    )


def test_final_syllable_build_rejects_complex_coda_targets() -> None:
    assert feature_is_compatible("FINAL_SYLLABLE_BUILD", "SYLLABLE.CVC")
    assert feature_is_compatible(
        "FINAL_SYLLABLE_BUILD", "GRAPHEME.CODA.SIMPLE.ㄱ"
    )
    assert not feature_is_compatible(
        "FINAL_SYLLABLE_BUILD", "GRAPHEME.CODA.COMPLEX.ㄳ"
    )


def test_double_final_build_rejects_simple_coda_targets() -> None:
    assert feature_is_compatible("DOUBLE_FINAL_BUILD", "SYLLABLE.COMPLEX_CODA")
    assert feature_is_compatible(
        "DOUBLE_FINAL_BUILD", "GRAPHEME.CODA.COMPLEX.ㄳ"
    )
    assert not feature_is_compatible(
        "DOUBLE_FINAL_BUILD", "GRAPHEME.CODA.SIMPLE.ㄱ"
    )
