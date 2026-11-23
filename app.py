import asyncio
import contextlib
import logging
import os
import sqlite3
from pathlib import Path
from typing import TypedDict

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers.default_request_handler_v2 import DefaultRequestHandlerV2
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.tasks import DatabaseTaskStore, TaskUpdater
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
from google.protobuf.json_format import MessageToDict
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from sqlalchemy.ext.asyncio import create_async_engine
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    prompt: str
    response: str


# The RealityEngine universe model: the MLX build of Nemotron 3 Nano 4B
# (RealityEngine_CI CLAUDE.md). The plain ":4b" tag is the GGUF build on
# llama.cpp and is not a substitute.
UNIVERSE_MODEL = "nemotron-3-nano:4b-mlx"


def chat_model() -> ChatOllama:
    # Never pass `format` (JSON mode or a schema): Ollama's MLX runner refuses
    # structured output for the nemotron-3-nano family (RealityEngine_CI
    # docs/OLLAMA_INTEGRATION.md). Expected keys belong in the prompt.
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", UNIVERSE_MODEL),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
    )


async def respond(state: GraphState) -> dict:
    model = chat_model()
    result = await model.ainvoke(state["prompt"])
    return {"response": result.text}


builder = StateGraph(GraphState)
builder.add_node("respond", respond)
builder.add_edge(START, "respond")
builder.add_edge("respond", END)
# Compiled without a checkpointer for direct use; Runtime.start() compiles the
# served graph with the persistent one.
graph = builder.compile()

# The RealityEngine A2A extension (RealityEngine_CI docs/A2A_FRAMEWORK_SPEC.md
# section 6, still a draft: CI#848). Only the members this service needs to key
# its state are read here. Carrying the extension exactly is #6.
# The extension's identifier is the URL of its published specification, pinned
# to the version tag (spec section 6.1, owner 2026-11-23). It is a constant,
# compared as a string, never fetched.
RE_EXTENSION_URI = "https://github.com/jateeter/RealityEngine_CI/blob/re-a2a-v0.1/docs/A2A_FRAMEWORK_SPEC.md"


def re_extension(context: RequestContext) -> dict:
    """The RealityEngine extension object from the message metadata, or {}."""
    message = context.message
    if message is None or not message.HasField("metadata"):
        return {}
    value = MessageToDict(message.metadata).get(RE_EXTENSION_URI)
    return value if isinstance(value, dict) else {}


def thread_id_for(context: RequestContext) -> str:
    """The LangGraph thread for a request (the resource model, #2).

    The thread is the dispatch's context (spec section 4.5, owner 2026-11-23):
    the workflow's context scope (frame, continuation, span or dispatch) sets
    its lifetime, so one thread serves one step, travels with a continuation,
    or spans frames, as configured. The id comes from the extension's
    `context.contextId`, never from a context id this server minted. Without
    the extension (a manual request), the A2A context id is used.
    """
    context_id = (re_extension(context).get("context") or {}).get("contextId")
    return context_id or context.context_id


def dispatch_id_for(context: RequestContext) -> str | None:
    return (re_extension(context).get("dispatch") or {}).get("dispatchId")


class Runtime:
    """Persistent state: the A2A task store, the LangGraph checkpointer and the
    dispatch index. Everything lives under RE_LANGSTACK_STATE_DIR and survives a
    restart (#4)."""

    def __init__(self, state_dir: str | os.PathLike):
        # Nothing touches the filesystem until start(): importing this module
        # must not need the state directory.
        self.state_dir = Path(state_dir)
        self.engine = create_async_engine(f"sqlite+aiosqlite:///{self.state_dir / 'tasks.sqlite'}")
        self.task_store = DatabaseTaskStore(self.engine)
        self.graph = None
        self._stack: contextlib.AsyncExitStack | None = None
        self._index: sqlite3.Connection | None = None

    async def start(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._index = sqlite3.connect(self.state_dir / "dispatches.sqlite", check_same_thread=False)
        self._index.execute(
            "CREATE TABLE IF NOT EXISTS dispatches (dispatch_id TEXT PRIMARY KEY, task_id TEXT NOT NULL)"
        )
        self._index.commit()
        self._stack = contextlib.AsyncExitStack()
        saver = await self._stack.enter_async_context(
            AsyncSqliteSaver.from_conn_string(str(self.state_dir / "checkpoints.sqlite"))
        )
        self.graph = builder.compile(checkpointer=saver)
        await self.task_store.initialize()

    async def stop(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
        self.graph = None
        if self._index is not None:
            self._index.close()
            self._index = None
        await self.engine.dispose()

    def claim_dispatch(self, dispatch_id: str, task_id: str) -> str | None:
        """Record dispatch_id -> task_id once. Returns the task that already holds
        the dispatch, or None when this task now holds it."""
        try:
            with self._index:
                self._index.execute(
                    "INSERT INTO dispatches (dispatch_id, task_id) VALUES (?, ?)", (dispatch_id, task_id)
                )
            return None
        except sqlite3.IntegrityError:
            row = self._index.execute(
                "SELECT task_id FROM dispatches WHERE dispatch_id = ?", (dispatch_id,)
            ).fetchone()
            return row[0] if row else None


class GraphExecutor(AgentExecutor):
    def __init__(self, runtime: Runtime):
        self.runtime = runtime

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
        dispatch_id = dispatch_id_for(context)
        if dispatch_id:
            holder = self.runtime.claim_dispatch(dispatch_id, task_id)
            if holder is not None and holder != task_id:
                # Once-only by dispatchId: a second request is never re-run.
                await updater.reject(
                    updater.new_agent_message([Part(text=f"Duplicate dispatch {dispatch_id}: task {holder}.")])
                )
                return
        await updater.start_work()
        try:
            result = await asyncio.wait_for(
                self.runtime.graph.ainvoke(
                    {"prompt": context.get_user_input()},
                    {"configurable": {"thread_id": thread_id_for(context)}},
                ),
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
async def health(request):
    return JSONResponse({"status": "ok"})


def create_app(state_dir: str | os.PathLike) -> Starlette:
    runtime = Runtime(state_dir)
    handler = DefaultRequestHandlerV2(
        agent_executor=GraphExecutor(runtime), task_store=runtime.task_store, agent_card=card
    )

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        await runtime.start()
        try:
            yield
        finally:
            await runtime.stop()

    application = Starlette(
        routes=[
            Route("/healthz", health),
            *create_agent_card_routes(card),
            *create_jsonrpc_routes(handler, "/a2a"),
        ],
        lifespan=lifespan,
    )
    application.state.runtime = runtime
    return application


app = create_app(os.getenv("RE_LANGSTACK_STATE_DIR", "/data"))
