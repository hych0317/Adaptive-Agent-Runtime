from __future__ import annotations

import inspect
import unittest

from applications.research_agent.strategies import ReportStrategy, ResearchWorkspace


class ReportPublicationBoundaryTests(unittest.TestCase):
    def test_report_strategy_has_no_workspace_output_or_raw_artifact_write_path(self) -> None:
        source = inspect.getsource(ReportStrategy)
        self.assertNotIn("set_output(", source)
        self.assertNotIn("workspace_artifact_store", source)
        self.assertNotIn("workspace_artifact_committer", source)
        self.assertIn("_decision_handler.commit", source)
        self.assertIn("_decision_handler.resume_existing", source)

    def test_workspace_rejects_report_without_commit_receipt(self) -> None:
        from applications.research_agent.tasks import build_research_task, REPORT_GENERATION

        definition = build_research_task("Acme")
        workspace = ResearchWorkspace(definition)
        with self.assertRaisesRegex(ValueError, "governed commit receipt"):
            workspace.set_output(
                definition.node(REPORT_GENERATION),
                {"markdown": "uncommitted report"},
            )


if __name__ == "__main__":
    unittest.main()
