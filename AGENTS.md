# Agent Guidance: localLangChainStack

Read `CLAUDE.md` for the codebase map, the resource model and the rules.

## Role

This repo is the A2A replacement path for `localOpenClawStack` in RealityEngine `release-v0.0.3`. It serves LangGraph agents, with LangChain in the flow, over A2A v1.0, using open-source `langgraph` and the official A2A Python SDK. There is no LangGraph Agent Server (#2).

## Authority

The masters are in RealityEngine_CI. Read current versions before changing observable behavior:

- [OWASP_ACS_ROADMAP.md § A2A](https://github.com/jateeter/RealityEngine_CI/blob/main/docs/OWASP_ACS_ROADMAP.md): decisions A9–A14, tasks K1–K14.
- [CONTINUATIONS_APPROVAL_CHAIN.md](https://github.com/jateeter/RealityEngine_CI/blob/main/docs/CONTINUATIONS_APPROVAL_CHAIN.md): frames and continuations.
- [TRANSPORT_SECURITY.md](https://github.com/jateeter/RealityEngine_CI/blob/main/docs/TRANSPORT_SECURITY.md): the TLS profile and the audience rule.
- [ENGINEERING_CONTRACT.md](https://github.com/jateeter/RealityEngine_CI/blob/main/docs/ENGINEERING_CONTRACT.md): the standing rules.

## Development Rules

- Pass continuations and asset references exactly as given. The lifecycle service stays the authority, and a graph never picks the next workflow step.
- Default to the universe model `nemotron-3-nano:4b-mlx`, and never send `format`.
- Use only maintained LangChain packages: no `langchain-community`, no `langgraph-api`. No LangSmith or cloud telemetry.
- Replies go back over A2A only, never by webhook or a direct PE call.
- Bind tools per assistant, never server-wide.
- Never commit credentials, `.env`, runtime state or logs.

## Verification

```bash
python -m unittest -v test_app
docker compose config
```

Report a check that could not run as unavailable, never as passing. Parity and conformance claims come only from RealityEngine_CI's lanes.
