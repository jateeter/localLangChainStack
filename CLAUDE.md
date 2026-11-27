# localLangChainStack Guidance

Last reviewed: 2026-11-23

See `/Users/johnt/workspace/GitHub/CLAUDE.md` for the integrated application map. Update both this file and the root map when this stack's role, its A2A surface, its model or transport wiring, or its suite integration changes.

## Role

This repo is **the A2A replacement path for `localOpenClawStack`** in `release-v0.0.3` (owner, 2026-11-23, decision A13). It hosts the agents RealityEngine engines dispatch to:

- the per-clone workflow agents;
- the corpus's `agent:*` bindings.

They are LangGraph agents, with LangChain in the flow, served over **A2A v1.0**.

**The serving path is decided (A14, #2):** open-source `langgraph` (MIT), served with the official A2A Python SDK (`a2a-sdk`, Apache-2.0). There is **no LangGraph Agent Server** (`langgraph-api`, Elastic-2.0, licence key, online check). The Agent Server's **resource model** is the design, built on open-source primitives:

| RealityEngine | Resource | Built from | Issue |
|---|---|---|---|
| a template machine's agent | graph | `langgraph` `StateGraph` | — |
| a clone's agent: created per clone and step, versioned, deleted on recycle | assistant (graph + config) | a registry in this service | #3 |
| frame and continuation (ingest context) | thread, created from our continuation id, checkpointed per step | `langgraph` checkpointer | #4 |
| step (dispatch context) | run | the A2A executor | #4 |
| park, then approve or rework | interrupt, then resume with `Command` | `langgraph` | #4 |
| frame isolation, tools per agent | auth scoping by the lifecycle's step capability | in front of the A2A handler | #5 |

## Authority

The masters are in RealityEngine_CI. Follow them, not this file, when they differ:

- **The plan and decisions:** `RealityEngine_CI/docs/OWASP_ACS_ROADMAP.md` § A2A, decisions A9–A14 and tasks K1–K14.
- **The A2A framework specification** (K1) and **the RealityEngine A2A extension for continuations and asset references** (K3): being written. Until they exist, this stack claims no extension (#6).
- **Continuations:** `RealityEngine_CI/docs/CONTINUATIONS_APPROVAL_CHAIN.md`, and localAssetStorageStack's lifecycle API.
- **Transport:** `RealityEngine_CI/docs/TRANSPORT_SECURITY.md` and `config/transport-profile.json`.
- **Model:** `RealityEngine_CI/docs/OLLAMA_INTEGRATION.md`.
- **Suite integration:** RealityEngine_CI#847, covering the focus set, build control, regression and release.

## Codebase Map

- `app.py`: the LangGraph graph, the A2A executor, the Agent Card, and the Starlette app (`/healthz`, `/.well-known/agent-card.json`, `/a2a` JSON-RPC).
- `behaviors.py`: runtime-neutral agent behaviors (RealityEngine_Machines `schemas/agent-behavior.schema.json`) rendered for A2A, at the template level of #3. Each is a card skill; `dispatch.skill` selects it; the reply's completion fields are parsed into a `completion` artifact. `deterministic-tool` behaviors are refused.
- `test_behaviors.py`: behavior tests, including loading the 16 ai-governance-regime behaviors when RealityEngine_Machines is beside this checkout.
- `test_app.py`: unit tests. Inference is stubbed, and no Ollama is needed.
- `Dockerfile`, `docker-compose.yml`, `.env.example`: the local container.

## Key Commands

```bash
python -m unittest -v test_app test_behaviors   # with requirements.txt (and httpx) installed
docker compose config
docker compose up --build -d
curl -s http://127.0.0.1:8090/.well-known/agent-card.json
docker compose down
```

A local green says this checkout's tests pass. Integrated verification (the `acs-a2a` regression stage, 3-of-3 with followers agreeing, and the conformance kit) runs from RealityEngine_CI.

## Rules That Bite

- **The lifecycle service is the authority for continuations** (localAssetStorageStack D10). Threads, checkpoints and interrupts are the agent's working state. A graph never chooses the next workflow step: only the PE's ISRE folding does.
- **Continuations and asset references pass exactly as given.** They are never minted, rewritten or dropped. Documents are read from the pod at their pinned revision, never from the store.
- **Replies return only over A2A,** through the dispatching engine's A2A handler and governance port. Never use a webhook or a direct call to a PE.
- **The universe model:** `nemotron-3-nano:4b-mlx`, never the plain `:4b` (GGUF). Never send `format` (JSON mode or a schema), because the MLX runner refuses it for this family.
- **Maintained LangChain packages only:** `langchain-core`, `langchain-ollama`, `langgraph`, `langgraph-prebuilt` and the like. Never `langchain-community` or another sunset or unmaintained package, never `langgraph-api`. A deprecation warning from a LangChain package is a defect.
- **No data leaves the install.** LangSmith tracing and LangChain cloud telemetry stay off. The only outbound connection is to the configured Ollama.
- **Transport:** TLS 1.3 only. mTLS to RealityEngine entities, TLS-only to external parties, and Ollama through the suite's TLS front (#8). Today the listener is plain HTTP on loopback, an open gap, not the design.
- **Tools are bound per assistant, never server-wide.** That is the deficiency OpenClaw couldn't close (RealityEngine_CI#714).

## Standing rules

The standing rules are in `../RealityEngine_CI/docs/ENGINEERING_CONTRACT.md`, which is authoritative. They are not listed or summarised here, so this file never falls behind the contract. Read it before changing anything here.
