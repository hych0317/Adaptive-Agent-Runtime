from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from adaptive_agent_runtime.llm import (
    LLMTargetSelection,
    ReasoningEffort,
    TOMLProviderConfigRepository,
)


class TOMLProviderConfigRepositoryTests(unittest.TestCase):
    def test_loads_credentials_and_target_selections(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "llm.local.toml"
            path.write_text(
                """schema_version = 1
[providers.deepseek]
api_key = "secret"

[selections.deepseek-research]
model = "deepseek-reasoner"
reasoning_effort = "high"
""",
                encoding="utf-8",
            )
            repository = TOMLProviderConfigRepository(path)

            keys = repository.load_api_keys()
            selections = repository.load_target_selections()

            self.assertEqual(keys["deepseek"].get_secret_value(), "secret")
            self.assertEqual(
                selections["deepseek-research"],
                LLMTargetSelection(
                    model_id="deepseek-reasoner",
                    reasoning_effort=ReasoningEffort.HIGH,
                ),
            )

    def test_saves_each_setting_without_erasing_the_other(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "llm.local.toml"
            repository = TOMLProviderConfigRepository(path)

            repository.save_api_key("deepseek", "secret")
            repository.save_target_selection(
                "deepseek-research",
                LLMTargetSelection(
                    model_id="deepseek-chat",
                    reasoning_effort=ReasoningEffort.MAX,
                ),
            )
            repository.save_target_selection(
                "research/codex",
                LLMTargetSelection(model_id="gpt-test"),
            )
            repository.save_api_key("openai", "second-secret")

            reloaded = TOMLProviderConfigRepository(path)
            self.assertEqual(set(reloaded.load_api_keys()), {"deepseek", "openai"})
            self.assertEqual(
                reloaded.load_target_selections()[
                    "deepseek-research"
                ].reasoning_effort,
                ReasoningEffort.MAX,
            )
            self.assertEqual(
                reloaded.load_target_selections()["research/codex"].model_id,
                "gpt-test",
            )
            self.assertEqual(list(path.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
