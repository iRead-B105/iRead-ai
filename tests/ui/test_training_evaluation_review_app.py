from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_training_evaluation_review_app_renders_sample_and_recording_modes() -> None:
    app_path = Path(__file__).resolve().parents[2] / "training_evaluation_review_app.py"

    app = AppTest.from_file(str(app_path), default_timeout=15).run()

    assert not app.exception
    assert app.title[0].value == "훈련 평가 검토"
    assert any(button.label == "샘플 평가 실행" for button in app.button)
