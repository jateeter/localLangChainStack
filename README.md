# localLangChainStack

A local Docker Compose deployment of a LangGraph agent using LangChain's Ollama
integration. It exposes an A2A 1.0 JSON-RPC agent for RealityEngine dispatchers.
The deployment uses an existing Ollama server, as localOpenClawStack does.

## Start

Requirements: Docker Engine, Docker Compose v2, and an Ollama server with a
chat model installed. The default is the RealityEngine universe model,
`nemotron-3-nano:4b-mlx` (`ollama pull nemotron-3-nano:4b-mlx`; Ollama 0.40.13
or later). It is the MLX build: the plain `:4b` tag is GGUF on llama.cpp and is
not a substitute. The agent never requests structured output (`format`), which
Ollama's MLX runner refuses for this model family.

```sh
cp .env.example .env
docker compose up --build -d
curl http://127.0.0.1:8090/healthz
curl http://127.0.0.1:8090/.well-known/agent-card.json
```

By default the container connects to `http://host.docker.internal:11434`.
Ollama must listen on an interface reachable from the container; configure
`OLLAMA_BASE_URL` for another server. Set `OLLAMA_MODEL` to an installed model
and `MODEL_TIMEOUT_SECONDS` to bound each invocation. `LANGSTACK_PORT` changes
the host port. Set `A2A_PUBLIC_URL` to the base URL reachable by the A2A client
(including the chosen port); the Agent Card advertises `/a2a` at that URL.
`host.docker.internal` inside another container refers to its host, not this
service; containers on the same Compose network can use the service name.

```sh
docker compose logs -f langgraph-agent
docker compose down
```

## A2A dispatch

Discover `/.well-known/agent-card.json` and send A2A 1.0 JSON-RPC requests to
the advertised interface. For example:

```sh
curl http://127.0.0.1:8090/a2a \
  -H 'Content-Type: application/json' -H 'A2A-Version: 1.0' \
  -d '{"jsonrpc":"2.0","id":1,"method":"SendMessage","params":{"message":{"messageId":"example-1","role":"ROLE_USER","parts":[{"text":"Hello"}]}}}'
```

The agent accepts non-empty text parts, runs a LangGraph model node and returns
a completed task with a text artifact, or a failed task if inference fails.
JSON can be sent as text. Unsupported file/data parts are rejected rather than
silently dropped. A2A streaming and task polling are supported by the SDK;
tasks are held in memory and are lost on restart. The health endpoint indicates
that the server is running, not that the model is available.

Configure the RealityEngine A2A dispatcher to discover this card and consume
the task result. The suite's A2A dispatch/ingress contract, source mapping and
RealityEngine continuation/asset extension are still being specified
([RealityEngine_CI#841](https://github.com/jateeter/RealityEngine_CI/issues/841));
this service does not implement the legacy ACP completion callback or claim
those unfinalized extensions. It does not expose a RealityEngine ingestion
endpoint. PE-specific configuration and end-to-end ingestion must follow the
final suite contract.

## Security

The agent is unauthenticated HTTP and the published port is loopback-only, for
local development. Anyone who can reach it can invoke the model. It is not a
production or cross-host deployment: use an authenticated TLS 1.3/mTLS
boundary as required by the RealityEngine transport profile before connecting
across trust boundaries. Model input and output are untrusted; this agent has
no tools or access to PE resources. No credentials are needed in `.env`.

## Tests

With the dependencies from `requirements.txt` installed, run
`python -m unittest -v test_app`. Tests stub inference and do not require Ollama.
