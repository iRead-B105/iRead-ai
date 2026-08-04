from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_backend_profile_review_app_renders_profile_and_two_api_actions() -> None:
    app_path = Path(__file__).resolve().parents[2] / "backend_profile_review_app.py"

    app = AppTest.from_file(str(app_path), default_timeout=15).run()

    assert not app.exception
    assert app.title[0].value == "백엔드 프로필 AI 연동 검토"
    labels = {button.label for button in app.button}
    assert "입력 검증" in labels
    assert "교수자 분석" in labels
    assert "커리큘럼 추천" in labels
    assert "둘 다 실행" in labels


def test_backend_profile_review_app_validates_anonymous_backend_sample() -> None:
    app_path = Path(__file__).resolve().parents[2] / "backend_profile_review_app.py"
    app = AppTest.from_file(str(app_path), default_timeout=15).run()

    validate_button = next(button for button in app.button if button.label == "입력 검증")
    app = validate_button.click().run()

    assert not app.exception
    assert any("두 AI 요청 계약이 모두 유효" in message.value for message in app.success)
