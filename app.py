import asyncio
import logging
import os
from typing import TypedDict

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers.default_request_handler_v2 import DefaultRequestHandlerV2
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types.a2a_pb2 import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Part,
    Task,
    TaskState,
    TaskStatus,
)
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    prompt: str
    response: str


async def respond(state: GraphState) -> dict:
    model = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
    )
    result = await model.ainvoke(state["prompt"])
    return {"response": result.text}


builder = StateGraph(GraphState)
builder.add_node("respond", respond)
builder.add_edge(START, "respond")
builder.add_edge("respond", END)
graph = builder.compile()


class GraphExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id, context_id = context.task_id, context.context_id
        updater = TaskUpdater(event_queue, task_id, context_id)
        if not context.current_task:
            await event_queue.enqueue_event(
                Task(
                    id=task_id,
                    context_id=context_id,
                    status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                    history=[context.message],
                )
            )
        message = context.message
        if not message or not message.parts or any(
            part.WhichOneof("content") != "text" for part in message.parts
        ) or not context.get_user_input().strip():
            await updater.reject(updater.new_agent_message([Part(text="Only non-empty text messages are supported.")]))
            return
        await updater.start_work()
        try:
            result = await asyncio.wait_for(
                graph.ainvoke({"prompt": context.get_user_input()}),
                timeout=float(os.getenv("MODEL_TIMEOUT_SECONDS", "120")),
            )
            await updater.add_artifact([Part(text=result["response"])], name="response")
            await updater.complete()
        except Exception:
            logger.exception("Agent execution failed")
            await updater.failed(updater.new_agent_message([Part(text="Agent execution failed.")]))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await TaskUpdater(event_queue, context.task_id, context.context_id).cancel()


public_url = os.getenv("A2A_PUBLIC_URL", "http://127.0.0.1:8090").rstrip("/")
card = AgentCard(
    name="Local LangGraph Agent",
    description="A local LangGraph agent backed by an Ollama chat model.",
    version="0.1.0",
    supported_interfaces=[
        AgentInterface(url=f"{public_url}/a2a", protocol_binding="JSONRPC", protocol_version="1.0")
    ],
    capabilities=AgentCapabilities(streaming=True),
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    skills=[AgentSkill(id="respond", name="Respond", description="Respond to a text message.", tags=["chat"])],
)
handler = DefaultRequestHandlerV2(
    agent_executor=GraphExecutor(), task_store=InMemoryTaskStore(), agent_card=card
)


async def health(request):
    return JSONResponse({"status": "ok"})


app = Starlette(routes=[
    Route("/healthz", health),
    *create_agent_card_routes(card),
    *create_jsonrpc_routes(handler, "/a2a"),
])
