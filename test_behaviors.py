import json
import pathlib
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx

import app
import behaviors

EXT = app.RE_EXTENSION_URI
GOVERNANCE = (pathlib.Path(__file__).resolve().parent.parent / "RealityEngine_Machines" / "machine_factory"
              / "domains" / "ai-governance-regime" / "agent-candidates" / "ai-governance-regime")


def behavior_doc(behavior_id, role="step-agent", fields=(("bundle_exported", "binary"), ("notice_drafted", "binary"))):
    return {
        "schemaVersion": "1.0.0", "behaviorId": behavior_id, "role": role, "agentId": behavior_id,
        "displayName": behavior_id.replace("-", " ").title(), "domain": "ai-governance-regime",
        "dispatch": {"workflow": "W5"}, "capabilities": ["frame.read"],
        "reasoning": {"objective": f"objective of {behavior_id}", "instructions": f"You are {behavior_id}.",
                      "outputContract": "Return `<field>: 0|1` lines, then `rationale: <one sentence>`."},
        "responseMapping": {"fields": [{"semantic": k, "valueType": t, "extract": {"responseKey": k}}
                                       for k, t in fields]},
    }


class BehaviorUnitTests(unittest.TestCase):
    def test_parse_completion(self):
        b = behaviors.from_document(behavior_doc("aigov-x"))
        got = b.parse_completion("- **bundle_exported**: 1\nrationale: exported the bundle.")
        self.assertEqual(got["fields"], {"bundle_exported": 1, "notice_drafted": None})
        self.assertEqual(got["rationale"], "exported the bundle.")
        self.assertFalse(got["complete"])
        self.assertTrue(b.parse_completion("bundle_exported: 0\nnotice_drafted: 1")["complete"])
        # The universe model's live reply (2026-11-27): pairs and the rationale on one line.
        one = b.parse_completion("bundle_exported: 1 notice_drafted: 0 rationale: The bundle is exported.")
        self.assertEqual(one["fields"], {"bundle_exported": 1, "notice_drafted": 0})
        self.assertEqual(one["rationale"], "The bundle is exported.")
        # A key that only appears inside another word is not a field.
        self.assertIsNone(b.parse_completion("prebundle_exported: 1")["fields"]["bundle_exported"])

    def test_refusals(self):
        self.assertIsNone(behaviors.from_document(behavior_doc("a")).refusal)
        self.assertIn("tool", behaviors.from_document(behavior_doc("b", role="deterministic-tool")).refusal)
        self.assertIn("no completion field", behaviors.from_document(behavior_doc("c", fields=())).refusal)

    def test_duplicate_ids_refused(self):
        with tempfile.TemporaryDirectory() as d:
            for n in ("one", "two"):
                pathlib.Path(d, f"{n}.agent.json").write_text(json.dumps(behavior_doc("same")))
            with self.assertRaises(ValueError):
                behaviors.load_behaviors(d)

    @unittest.skipUnless(GOVERNANCE.is_dir(), "RealityEngine_Machines is not beside this checkout")
    def test_governance_behaviors_load(self):
        loaded = behaviors.load_behaviors(GOVERNANCE)
        self.assertEqual(len(loaded), 16)
        runnable = [b for b in loaded.values() if b.refusal is None]
        self.assertTrue(runnable)
        for b in runnable:
            self.assertTrue(b.fields, b.behavior_id)
            self.assertIn("OUTPUT CONTRACT", b.system_prompt())


class BehaviorDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.state = tempfile.TemporaryDirectory()
        self.dir = tempfile.TemporaryDirectory()
        pathlib.Path(self.dir.name, "responder.agent.json").write_text(json.dumps(behavior_doc("aigov-responder")))
        pathlib.Path(self.dir.name, "tool.agent.json").write_text(
            json.dumps(behavior_doc("aigov-verifier", role="deterministic-tool")))
        self.app = app.create_app(self.state.name, self.dir.name)
        self.runtime = self.app.state.runtime
        await self.runtime.start()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.runtime.stop()
        self.state.cleanup()
        self.dir.cleanup()

    async def send(self, skill, text="step", dispatch_id=None):
        dispatch = {"skill": skill}
        if dispatch_id:
            dispatch["dispatchId"] = dispatch_id
        message = {"messageId": "m", "role": "ROLE_USER", "parts": [{"text": text}],
                   "metadata": {EXT: {"dispatch": dispatch}}}
        r = await self.client.post("/a2a", headers={"A2A-Version": "1.0"},
                                   json={"jsonrpc": "2.0", "id": 1, "method": "SendMessage",
                                         "params": {"message": message}})
        return r.json()["result"]["task"]

    async def test_card_lists_behaviors_as_skills(self):
        card = (await self.client.get("/.well-known/agent-card.json")).json()
        ids = [s["id"] for s in card["skills"]]
        self.assertEqual(ids, ["respond", "aigov-responder", "aigov-verifier"])
        verifier = next(s for s in card["skills"] if s["id"] == "aigov-verifier")
        self.assertIn("not run here", verifier["description"])

    async def test_dispatch_runs_the_behavior_and_returns_its_completion(self):
        reply = "bundle_exported: 1\nnotice_drafted: 1\nrationale: both done."
        with patch.object(self.runtime.graph, "ainvoke", new=AsyncMock(return_value={"response": reply})) as invoke:
            task = await self.send("aigov-responder", "incident frame asset-3")
        self.assertEqual(task["status"]["state"], "TASK_STATE_COMPLETED")
        state = invoke.call_args.args[0]
        self.assertEqual(state["prompt"], "incident frame asset-3")
        self.assertIn("You are aigov-responder.", state["system"])
        self.assertIn("OUTPUT CONTRACT", state["system"])
        names = [a["name"] for a in task["artifacts"]]
        self.assertEqual(names, ["response", "completion"])
        completion = json.loads(task["artifacts"][1]["parts"][0]["text"])
        self.assertEqual(completion["behaviorId"], "aigov-responder")
        self.assertEqual(completion["fields"], {"bundle_exported": 1, "notice_drafted": 1})
        self.assertTrue(completion["complete"])

    async def test_unknown_and_tool_skills_are_refused_not_run(self):
        with patch.object(self.runtime.graph, "ainvoke", new=AsyncMock()) as invoke:
            unknown = await self.send("no-such-skill")
            tool = await self.send("aigov-verifier")
        self.assertEqual(unknown["status"]["state"], "TASK_STATE_REJECTED")
        self.assertIn("Unknown skill", unknown["status"]["message"]["parts"][0]["text"])
        self.assertEqual(tool["status"]["state"], "TASK_STATE_REJECTED")
        self.assertIn("deterministic-tool", tool["status"]["message"]["parts"][0]["text"])
        invoke.assert_not_awaited()

    async def test_system_turn_reaches_the_model(self):
        model = Mock()
        model.ainvoke = AsyncMock(return_value=Mock(text="bundle_exported: 0\nnotice_drafted: 0"))
        with patch.object(app, "ChatOllama", return_value=model):
            task = await self.send("aigov-responder")
        self.assertEqual(task["status"]["state"], "TASK_STATE_COMPLETED")
        messages = model.ainvoke.call_args.args[0]
        self.assertEqual([type(m).__name__ for m in messages], ["SystemMessage", "HumanMessage"])
        self.assertIn("You are aigov-responder.", messages[0].content)


if __name__ == "__main__":
    unittest.main()
