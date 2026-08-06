from pydantic import SecretStr

from iread_ai.config import Settings
from iread_ai.main import create_app
from iread_ai.personalization.visual_scene import (
    MockVisualScenePlanner,
    OpenAIVisualScenePlanner,
)


def test_legacy_story_disables_visual_scene_llm() -> None:
    app = create_app(
        settings=Settings(
            app_env="test",
            story_provider="openai",
            openai_api_key=SecretStr("test-openai-key"),
            story_image_provider="disabled",
        )
    )

    legacy_chapter_service = app.state.legacy_story_service._chapter_service
    assert isinstance(
        legacy_chapter_service._visual_scene_planner,
        MockVisualScenePlanner,
    )
    assert isinstance(
        app.state.story_chapter_service._visual_scene_planner,
        OpenAIVisualScenePlanner,
    )


def test_chapter_quality_retry_defaults_to_one_retry() -> None:
    assert Settings.model_fields["chapter_quality_retry_count"].default == 1
