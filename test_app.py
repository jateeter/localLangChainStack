import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx

import app


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app.app), base_url="http://test"
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def send(self, parts, **configuration):
        return await self.client.post(
            "/a2a",
            headers={"A2A-Version": "1.0"},
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "SendMessage",
                "params": {
                    "message": {"messageId": "test", "role": "ROLE_USER", "parts": parts},
                    "configuration": configuration,
                },
            },
        )

    async def test_card_and_completed_task(self):
        card = (await self.client.get("/.well-known/agent-card.json")).json()
        self.assertEqual(card["supportedInterfaces"][0]["protocolVersion"], "1.0")
        with patch.object(app.graph, "ainvoke", new=AsyncMock(return_value={"response": "answer"})) as invoke:
            response = await self.send([{"text": "question"}])
        task = response.json()["result"]["task"]
        self.assertEqual(task["status"]["state"], "TASK_STATE_COMPLETED")
        self.assertEqual(task["artifacts"][0]["parts"][0]["text"], "answer")
        invoke.assert_awaited_once_with({"prompt": "question"})
        response = await self.client.post(
            "/a2a", headers={"A2A-Version": "1.0"},
            json={"jsonrpc": "2.0", "id": 2, "method": "GetTask", "params": {"id": task["id"]}},
        )
        self.assertEqual(response.json()["result"]["status"]["state"], "TASK_STATE_COMPLETED")

    async def test_failure_and_unsupported_input(self):
        with patch.object(app.graph, "ainvoke", new=AsyncMock(side_effect=RuntimeError("model unavailable"))):
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


if __name__ == "__main__":
    unittest.main()
