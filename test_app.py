import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx

import app

EXT = app.RE_EXTENSION_URI


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.state = tempfile.TemporaryDirectory()
        await self.start()

    async def start(self):
        self.app = app.create_app(self.state.name)
        self.runtime = self.app.state.runtime
        await self.runtime.start()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test")

    async def stop(self):
        await self.client.aclose()
        await self.runtime.stop()

    async def asyncTearDown(self):
        await self.stop()
        self.state.cleanup()

    async def send(self, parts, metadata=None, context_id=None, **configuration):
        message = {"messageId": "test", "role": "ROLE_USER", "parts": parts}
        if metadata is not None:
            message["metadata"] = metadata
        if context_id is not None:
            message["contextId"] = context_id
        return await self.client.post(
            "/a2a",
            headers={"A2A-Version": "1.0"},
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "SendMessage",
                "params": {"message": message, "configuration": configuration},
            },
        )

    async def get_task(self, task_id):
        response = await self.client.post(
            "/a2a", headers={"A2A-Version": "1.0"},
            json={"jsonrpc": "2.0", "id": 2, "method": "GetTask", "params": {"id": task_id}},
        )
        return response.json()["result"]

    def stub_graph(self, **kwargs):
        return patch.object(self.runtime.graph, "ainvoke", new=AsyncMock(**kwargs))

    async def test_card_and_completed_task(self):
        card = (await self.client.get("/.well-known/agent-card.json")).json()
        self.assertEqual(card["supportedInterfaces"][0]["protocolVersion"], "1.0")
        with self.stub_graph(return_value={"response": "answer"}) as invoke:
            response = await self.send([{"text": "question"}], context_id="ctx-1")
        task = response.json()["result"]["task"]
        self.assertEqual(task["status"]["state"], "TASK_STATE_COMPLETED")
        self.assertEqual(task["artifacts"][0]["parts"][0]["text"], "answer")
        invoke.assert_awaited_once_with({"prompt": "question"}, {"configurable": {"thread_id": "ctx-1"}})
        self.assertEqual((await self.get_task(task["id"]))["status"]["state"], "TASK_STATE_COMPLETED")

    async def test_failure_and_unsupported_input(self):
        with self.stub_graph(side_effect=RuntimeError("model unavailable")):
            task = (await self.send([{"text": "question"}])).json()["result"]["task"]
        self.assertEqual(task["status"]["state"], "TASK_STATE_FAILED")
        task = (await self.send([{"url": "https://example.com/file"}])).json()["result"]["task"]
        self.assertEqual(task["status"]["state"], "TASK_STATE_REJECTED")

    async def test_graph_invokes_ollama(self):
        model = Mock()
        model.ainvoke = AsyncMock(return_value=Mock(text="model response"))
        with patch.object(app, "ChatOllama", return_value=model):
            result = await app.graph.ainvoke({"prompt": "hello"})
        self.assertEqual(result["response"], "model response")
        model.ainvoke.assert_awaited_once_with("hello")

    def test_defaults_to_universe_model_without_format(self):
        with patch.dict(app.os.environ, {}, clear=False) as env:
            env.pop("OLLAMA_MODEL", None)
            with patch.object(app, "ChatOllama") as chat:
                app.chat_model()
        kwargs = chat.call_args.kwargs
        self.assertEqual(kwargs["model"], "nemotron-3-nano:4b-mlx")
        self.assertNotIn("format", kwargs)

    def test_model_override_from_environment(self):
        with patch.dict(app.os.environ, {"OLLAMA_MODEL": "other-model"}):
            with patch.object(app, "ChatOllama") as chat:
                app.chat_model()
        self.assertEqual(chat.call_args.kwargs["model"], "other-model")
        self.assertNotIn("format", chat.call_args.kwargs)

    async def test_unsupported_version(self):
        response = await self.client.post(
            "/a2a", headers={"A2A-Version": "9.0"},
            json={"jsonrpc": "2.0", "id": 1, "method": "SendMessage", "params": {}},
        )
        self.assertIn("error", response.json())

    async def test_frame_step_keys_thread_by_asset_and_continuation(self):
        metadata = {EXT: {"frame": {"assetId": "asset-7", "continuation": {"continuationId": "k2"}}}}
        with self.stub_graph(return_value={"response": "ok"}) as invoke:
            await self.send([{"text": "step"}], metadata=metadata, context_id="minted-elsewhere")
        self.assertEqual(invoke.call_args.args[1], {"configurable": {"thread_id": "asset-7/k2"}})

    async def test_duplicate_dispatch_is_rejected_not_rerun(self):
        metadata = {EXT: {"dispatch": {"dispatchId": "d-1"}}}
        with self.stub_graph(return_value={"response": "ok"}) as invoke:
            first = (await self.send([{"text": "q"}], metadata=metadata)).json()["result"]["task"]
            second = (await self.send([{"text": "q"}], metadata=metadata)).json()["result"]["task"]
        self.assertEqual(first["status"]["state"], "TASK_STATE_COMPLETED")
        self.assertEqual(second["status"]["state"], "TASK_STATE_REJECTED")
        self.assertIn(first["id"], second["status"]["message"]["parts"][0]["text"])
        invoke.assert_awaited_once()

    async def test_tasks_checkpoints_and_dispatches_survive_restart(self):
        model = Mock()
        model.ainvoke = AsyncMock(return_value=Mock(text="kept"))
        metadata = {EXT: {"dispatch": {"dispatchId": "d-9"},
                          "frame": {"assetId": "asset-1", "continuation": {"continuationId": "k0"}}}}
        with patch.object(app, "ChatOllama", return_value=model):
            task = (await self.send([{"text": "before restart"}], metadata=metadata)).json()["result"]["task"]
        self.assertEqual(task["status"]["state"], "TASK_STATE_COMPLETED")

        await self.stop()
        await self.start()

        self.assertEqual((await self.get_task(task["id"]))["status"]["state"], "TASK_STATE_COMPLETED")
        state = await self.runtime.graph.aget_state({"configurable": {"thread_id": "asset-1/k0"}})
        self.assertEqual(state.values["response"], "kept")
        with self.stub_graph(return_value={"response": "x"}) as invoke:
            again = (await self.send([{"text": "q"}], metadata=metadata)).json()["result"]["task"]
        self.assertEqual(again["status"]["state"], "TASK_STATE_REJECTED")
        invoke.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
