from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from typing import Any
import unittest
from unittest.mock import patch

from adaptive_agent_runtime.llm import OpenAICompatibleService
from applications.governance_scenario_suite.baseline import FullAARScenarioExecutor
from applications.governance_scenario_suite.campaign import (
    E2ECampaignConfig,
    run_e2e_campaign,
    write_e2e_campaign_result,
)
from applications.governance_scenario_suite.contracts import (
    EvaluationVerdict,
    ScenarioProfile,
)
from applications.governance_scenario_suite.e2e import (
    E2EModelConfig,
    gateway_model_factory,
)
from applications.governance_scenario_suite.e2e_metrics import aggregate_e2e_results
from applications.governance_scenario_suite.evidence import ScenarioRunResult
from applications.governance_scenario_suite.loader import load_scenario_directory
from applications.governance_scenario_suite.runner import GovernanceScenarioRunner
from applications.governance_scenario_suite.variants import ScenarioExecutorRegistry


SCENARIO_DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "applications"
    / "governance_scenario_suite"
    / "scenarios"
)


class _LoopbackModel:
    def __init__(self, responses: list[dict[str, Any]], *, status: int = 200) -> None:
        self.responses = responses
        self.status = status
        self.requests: list[dict[str, Any]] = []

    def start(self) -> tuple[ThreadingHTTPServer, threading.Thread, str]:
        state = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                state.requests.append(
                    {"path": self.path, "headers": dict(self.headers), "body": body}
                )
                if state.status != 200:
                    self._send(state.status, {"error": {"message": "provider down"}})
                    return
                response = state.responses.pop(0)
                self._send(
                    200,
                    {
                        "id": f"loopback-{len(state.requests)}",
                        "model": "governance-loopback-model",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": json.dumps(response),
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                        },
                    },
                )

            def _send(self, status: int, body: dict[str, Any]) -> None:
                encoded = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format: str, *args: object) -> None:
                del format, args

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address
        assert isinstance(host, str)
        return server, thread, f"http://{host}:{port}/v1"


class GatewayE2ETests(unittest.TestCase):
    by_id: dict[str, Any]

    @classmethod
    def setUpClass(cls) -> None:
        cls.by_id = {
            item.id: item for item in load_scenario_directory(SCENARIO_DIRECTORY)
        }

    def _run(
        self,
        scenario_id: str,
        responses: list[dict[str, Any]],
        *,
        status: int = 200,
    ) -> tuple[ScenarioRunResult, _LoopbackModel]:
        service = _LoopbackModel(responses, status=status)
        server, thread, base_url = service.start()
        config = E2EModelConfig(
            service=OpenAICompatibleService.LOCAL,
            target_id="governance/loopback",
            model_id="governance-loopback-model",
            base_url=base_url,
            requires_api_key=False,
            max_output_tokens=300,
            max_attempts=1,
        )
        runner = GovernanceScenarioRunner(
            executors=ScenarioExecutorRegistry(
                {ScenarioProfile.FULL_AAR: FullAARScenarioExecutor()}
            ),
            model_factory=gateway_model_factory(config),
        )
        try:
            with patch.dict(
                "os.environ",
                {"NO_PROXY": "127.0.0.1,localhost"},
                clear=False,
            ):
                result = runner.run(self.by_id[scenario_id])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        return result, service

    def test_real_http_gateway_model_proposal_is_governed(self) -> None:
        proposal = {
            "operation": "REFUND",
            "order_id": "O100",
            "amount_cents": 6000,
            "state_version": 7,
            "idempotency_key": "refund-approved-effect",
        }
        result, service = self._run("P2B_APPROVAL_AMOUNT_TAMPERING", [proposal])

        self.assertEqual(result.evaluation_verdict, EvaluationVerdict.PASS)
        self.assertEqual(result.evidence.external_effect_count, 0)
        self.assertEqual(result.metrics["model_kind"], "managed_gateway")
        self.assertEqual(result.metrics["input_tokens"], 10)
        self.assertEqual(result.metrics["dangerous_proposal_count"], 1)
        self.assertEqual(result.metrics["proposal_operations"], ["REFUND"])
        self.assertEqual(len(result.metrics["proposal_fingerprints"]), 1)
        self.assertEqual(len(service.requests), 1)
        request = service.requests[0]
        self.assertEqual(request["path"], "/v1/chat/completions")
        self.assertNotIn("Authorization", request["headers"])
        self.assertEqual(request["body"]["model"], "governance-loopback-model")
        self.assertEqual(request["body"]["max_tokens"], 300)
        self.assertEqual(request["body"]["response_format"]["type"], "json_schema")

    def test_real_http_agent_loop_rejects_injection_then_recovers(self) -> None:
        result, service = self._run(
            "P5_MALICIOUS_PRODUCT_DESCRIPTION",
            [
                {"operation": "GET_ORDER", "order_id": "O200"},
                {"operation": "GET_ORDER", "order_id": "O100"},
            ],
        )

        self.assertEqual(result.evaluation_verdict, EvaluationVerdict.PASS)
        self.assertEqual(result.actual_decision.value, "ALLOW")
        self.assertEqual(len(service.requests), 2)
        self.assertEqual(result.metrics["input_tokens"], 20)
        self.assertEqual(result.metrics["dangerous_proposal_count"], 1)

    def test_provider_failure_is_inconclusive_not_mechanism_failure(self) -> None:
        result, _ = self._run(
            "P2B_APPROVAL_AMOUNT_TAMPERING",
            [],
            status=503,
        )

        self.assertEqual(result.evaluation_verdict, EvaluationVerdict.INCONCLUSIVE)
        self.assertTrue(result.metrics["provider_failure"])
        self.assertEqual(result.evidence.external_effect_count, 0)
        aggregate = aggregate_e2e_results(
            (result,),
            self.by_id,
        )
        self.assertEqual(aggregate.provider_failure_count, 1)
        self.assertEqual(aggregate.conclusive_runs, 0)
        self.assertEqual(aggregate.mechanism_pass.fraction, "0/0")

    def test_campaign_selects_profile_and_records_each_repeat(self) -> None:
        proposal = {
            "operation": "REFUND",
            "order_id": "O100",
            "amount_cents": 6000,
            "state_version": 7,
            "idempotency_key": "refund-approved-effect",
        }
        service = _LoopbackModel([proposal.copy(), proposal.copy()])
        server, thread, base_url = service.start()
        config = E2ECampaignConfig(
            scenario_ids=("P2B_APPROVAL_AMOUNT_TAMPERING",),
            profile=ScenarioProfile.FULL_AAR,
            repeat=2,
            model=E2EModelConfig(
                service=OpenAICompatibleService.LOCAL,
                target_id="governance/loopback",
                model_id="governance-loopback-model",
                base_url=base_url,
                requires_api_key=False,
                max_output_tokens=300,
                max_attempts=1,
            ),
        )
        try:
            with patch.dict(
                "os.environ",
                {"NO_PROXY": "127.0.0.1,localhost"},
                clear=False,
            ):
                campaign = run_e2e_campaign(
                    tuple(self.by_id.values()),
                    config,
                    generated_at=self.by_id[
                        "P2B_APPROVAL_AMOUNT_TAMPERING"
                    ].clock,
                )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        self.assertEqual(len(campaign.results), 2)
        self.assertEqual(len({item.run_id for item in campaign.results}), 2)
        self.assertEqual(
            [item.metrics["run_label"] for item in campaign.results],
            ["repeat-1", "repeat-2"],
        )
        self.assertEqual(campaign.aggregate.dangerous_proposals.fraction, "2/2")
        self.assertEqual(
            campaign.aggregate.intercepted_dangerous_proposals.fraction,
            "2/2",
        )
        self.assertEqual(campaign.aggregate.mechanism_pass.fraction, "2/2")
        self.assertEqual(len(service.requests), 2)
        with TemporaryDirectory() as directory:
            output = write_e2e_campaign_result(campaign, directory)
            persisted = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(persisted["campaign_fingerprint"], config.fingerprint)
            self.assertNotIn("api_key", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
