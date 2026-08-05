from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_training_playground_renders_three_step_flow() -> None:
    app_path = Path(__file__).resolve().parents[2] / "training_playground_app.py"

    app = AppTest.from_file(str(app_path), default_timeout=20).run()

    assert not app.exception
    assert app.title[0].value == "iRead 훈련 통합 테스트"
    assert [tab.label for tab in app.tabs] == [
        "1. 커리큘럼 추천",
        "2. 교안 생성",
        "3. 훈련 체험",
    ]
    labels = {button.label for button in app.button}
    assert "커리큘럼 생성" in labels
    assert "새 테스트 시작" in labels
