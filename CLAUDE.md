# Inference Server — Architecture & Design

## Architecture

```
React Frontend (port 3000)
    ↓ GraphQL mutation / query (HTTP)
Rust API Server (port 8080)   ← axum + async-graphql
    ↓ HTTP POST /infer  (internal Docker network)
Python Model Worker (port 8001, internal only)
    ↓  [optional] DuckDuckGo web search → context injection
    ↓ HuggingFace transformers
GPU / CPU inference
```

## Design Decisions

### Why Rust for the API layer?
The Rust server does **not** load the model. It handles:
- GraphQL schema, request validation, auth
- Queuing, rate limiting, streaming coordination
- Connection multiplexing to the Python worker

Axum + async-graphql give near-zero overhead. Rust's ownership model makes
concurrent request handling safe without a GIL.

### Why Python for the model worker?
PyTorch and HuggingFace transformers are Python-native. Running LLaMA inference
from Rust (via tch-rs or ONNX) eats a weekend on compatibility issues alone.
The worker is **internal-only** — never exposed outside the Docker network.

### Why GraphQL?
- `mutation infer(prompt, webSearch)` for inference requests
- `query health` for readiness checks
- Subscriptions wired for streaming (SSE / WebSocket) when you need it
- Self-documenting schema; introspection works out of the box

### Web Search (RAG)
When `webSearch: true` is passed, the Python worker:
1. Queries DuckDuckGo for the user's prompt (top 4 results)
2. Injects results as a system message before the model prompt
3. Instructs the model to answer from the live results and ignore its training cutoff

Library: `ddgs` (the renamed `duckduckgo-search`). No API key required.

## Project Structure

```
inference-server/
├── CLAUDE.md
├── README.md
├── .env.example              ← copy to .env, add HF_TOKEN
├── .gitignore
├── docker-compose.yml
├── scripts/
│   ├── start.sh              # build + start (detached)
│   ├── stop.sh               # stop and remove containers
│   ├── restart.sh            # stop → rebuild → start
│   └── install-nvidia.sh     # NVIDIA driver + container toolkit
├── rust-server/
│   ├── Dockerfile            # multi-stage: rust:1-slim → debian:bookworm-slim
│   ├── Cargo.toml            # axum, async-graphql, reqwest (rustls), tower-http
│   └── src/main.rs           # GraphQL schema: infer(prompt, webSearch?) + health
├── python-worker/
│   ├── Dockerfile            # python:3.11-slim + torch cu121
│   ├── requirements.txt      # fastapi, transformers, peft, bitsandbytes, ddgs
│   └── worker.py             # model load, DDG search, /infer endpoint
└── frontend/
    ├── Dockerfile
    ├── package.json          # React, Vite, Apollo Client, GraphiQL v3
    ├── vite.config.ts
    ├── tsconfig.json
    ├── index.html
    └── src/
        ├── main.tsx          # Apollo provider → VITE_GRAPHQL_URL
        └── App.tsx           # Tab 1: chat + web search toggle | Tab 2: GraphiQL
```

## Services

| Service | Port | Tech | Responsibility |
|---------|------|------|----------------|
| `rust-server` | 8080 (public) | Axum + async-graphql | GraphQL API, routing |
| `python-worker` | 8001 (internal) | FastAPI + transformers + ddgs | Model inference + web search |
| `frontend` | 3000 (public) | React + Vite + Apollo + GraphiQL | Chat UI + schema explorer |

## Quick Start

```bash
cp .env.example .env          # add HF_TOKEN
./scripts/start.sh
```

URLs once running:
- **Frontend + Chat** → http://localhost:3000
- **GraphQL Explorer** → http://localhost:3000 (second tab)
- **GraphQL endpoint** → http://localhost:8080/graphql

## Environment Variables

| Variable | Default | Notes |
|----------|---------|-------|
| `HF_TOKEN` | — | Required for gated models (Llama etc.) |
| `MODEL_ID` | `meta-llama/Llama-3.2-3B-Instruct` | Any HF causal LM |
| `LOAD_IN_4BIT` | `false` | Enable for 7B+ models on ≤12 GB VRAM |

## GPU Support

The `python-worker` service has the NVIDIA `deploy` block active in
`docker-compose.yml`. Requires the NVIDIA Container Toolkit on the host:

```bash
./scripts/install-nvidia.sh   # then reboot
```

## Frontend Tabs

**Chat** — bubble-style messages. Web search toggle in the input bar (🌐 Off/On).
When on, the user message is labelled `🌐 web search` and the loading indicator
shows `🌐 searching…`.

**GraphQL Explorer** — embedded GraphiQL v3. Schema introspection, query/mutation
builder, variables panel, history, docs sidebar. Points at `rust-server:8080`.

## Extending

- **Streaming**: Add a `Subscription` in `rust-server/src/main.rs`, switch Apollo
  Client to a WebSocket link in `frontend/src/main.tsx`
- **Queuing**: Add Redis to compose; Rust enqueues, Python dequeues
- **Multiple models**: Add a `model: String` arg to the GraphQL mutation, route
  in `worker.py`
- **Auth**: Add a Tower middleware layer in Rust before the GraphQL handler
- **LoRA fine-tuning**: `peft` is already installed in the worker image:

```python
from peft import get_peft_model, LoraConfig
config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, config)
```
