# Inference Server — Architecture & Design

## Architecture

```
React Frontend (port 3000)
    ↓ GraphQL mutation / query  (HTTP  → /graphql)
    ↓ GraphQL subscription      (WS   → /graphql/ws)
Rust API Server (port 8080)   ← axum + async-graphql
    ↓ HTTP POST /infer          (internal Docker network)
    ↓ HTTP POST /stream   →  SSE stream
    ↓ HTTP GET  /adapters, /model-info, /fine-tune/*
Python Model Worker (port 8001, internal only)
    ↓  [optional] DuckDuckGo web search → context injection
    ↓  HuggingFace transformers (base model or LoRA adapter)
GPU / CPU inference
```

## Design Decisions

### Why Rust for the API layer?
The Rust server does **not** load the model. It handles:
- GraphQL schema, request validation
- Streaming coordination: consumes Python's SSE stream and relays tokens over WebSocket to the browser
- Connection multiplexing to the Python worker

Axum + async-graphql give near-zero overhead. Rust's ownership model makes concurrent request handling safe without a GIL.

### Why Python for the model worker?
PyTorch and HuggingFace transformers are Python-native. The worker is **internal-only** — never exposed outside the Docker network.

### Why GraphQL?
- `subscription inferStream(prompt, webSearch?)` for token-by-token streaming
- `mutation infer(prompt, webSearch?)` for non-streaming (kept for GraphQL Explorer)
- `mutation startFineTune(...)` / `mutation loadAdapter(...)` / `mutation useBaseModel` for adapter management
- `query fineTuneStatus` / `query adapters` / `query modelInfo` for observability
- Self-documenting schema; introspection works out of the box in the Explorer tab

### Streaming pipeline
```
model.generate() → TextIteratorStreamer (background thread)
    → FastAPI SSE  (/stream)
    → reqwest bytes_stream in Rust
    → SSE frame parser (finds \n\n boundaries)
    → async-graphql Subscription (yield TokenChunk)
    → Apollo WebSocket link
    → React token-by-token state update
```

### Web Search (RAG)
When `webSearch: true` is passed, the Python worker:
1. Queries DuckDuckGo for the user's prompt (top 4 results)
2. Injects results as a system message before the model prompt
3. Instructs the model to answer from live results and ignore its training cutoff

Library: `ddgs` (the renamed `duckduckgo-search`). No API key required.

### LoRA Fine-tuning (QLoRA)
Fine-tuning runs entirely in the Python worker via a background thread. Memory strategy for 11 GB VRAM:

1. **Offload** the float16 inference model (~6 GB) to CPU
2. **Load** a fresh 4-bit (QLoRA) training copy of the same model (~1.5 GB GPU)
3. **Train** with `paged_adamw_8bit`, `batch_size=1`, `gradient_accumulation=8`, gradient checkpointing
4. **Save** LoRA adapter weights to the `adapters` volume (`/adapters/<name>/`)
5. **Delete** the training model, move inference model back to GPU

Inference is blocked (model on CPU) only during training — chat resumes automatically when done.

Loading an adapter (`PeftModel.from_pretrained(base_model, path)`) adds ~100–200 MB VRAM. Switching between adapters or back to base model happens at runtime with no restart.

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
│   ├── Cargo.toml            # axum, async-graphql, reqwest (rustls), async-stream, futures-util
│   └── src/main.rs           # GraphQL schema — queries, mutations, subscription
├── python-worker/
│   ├── Dockerfile            # python:3.11-slim + torch cu121
│   ├── requirements.txt      # fastapi, transformers, peft, bitsandbytes, datasets, ddgs
│   └── worker.py             # model load, DDG search, inference, streaming SSE, fine-tuning
└── frontend/
    ├── Dockerfile
    ├── package.json          # React, Vite, Apollo Client, graphql-ws, GraphiQL v3
    ├── vite.config.ts
    ├── tsconfig.json
    ├── index.html
    └── src/
        ├── main.tsx          # Apollo provider — split HTTP/WS link
        └── App.tsx           # Tab 1: streaming chat | Tab 2: fine-tune | Tab 3: GraphiQL
```

## Services

| Service | Port | Tech | Responsibility |
|---------|------|------|----------------|
| `rust-server` | 8080 (public) | Axum + async-graphql | GraphQL API, SSE→WS relay |
| `python-worker` | 8001 (internal) | FastAPI + transformers + peft + ddgs | Inference, streaming, fine-tuning, web search |
| `frontend` | 3000 (public) | React + Vite + Apollo + GraphiQL | Chat UI, fine-tune UI, schema explorer |

## Docker Volumes

| Volume | Mount | Purpose |
|--------|-------|---------|
| `hf-cache` | `/root/.cache/huggingface` | Downloaded model weights — persists across restarts |
| `adapters` | `/adapters` | Trained LoRA adapters — one subdirectory per adapter |

## Environment Variables

| Variable | Default | Notes |
|----------|---------|-------|
| `HF_TOKEN` | — | Required for gated models (Llama etc.) |
| `MODEL_ID` | `meta-llama/Llama-3.2-3B-Instruct` | Any HF causal LM |
| `LOAD_IN_4BIT` | `false` | Enable for 7B+ models on ≤12 GB VRAM |
| `ADAPTER_DIR` | `/adapters` | Where trained LoRA adapters are saved |
| `WORKER_URL` | `http://python-worker:8001` | Rust → Python internal URL |
| `VITE_GRAPHQL_URL` | `http://localhost:8080/graphql` | Frontend HTTP endpoint |
| `VITE_GRAPHQL_WS_URL` | `ws://localhost:8080/graphql/ws` | Frontend WebSocket endpoint |

## GraphQL Schema Summary

**Queries**
- `health` → `String` — liveness check
- `modelInfo` → `ModelInfo` — model ID, device, VRAM, active adapter, 4-bit flag
- `adapters` → `AdaptersInfo` — `{ available: [String], active: String }` — list of saved adapters + which is loaded
- `fineTuneStatus` → `FineTuneStatus` — `{ running, step, totalSteps, loss, error, completedAdapter }`

**Mutations**
- `infer(prompt, webSearch?)` → `InferenceResult` — non-streaming inference
- `startFineTune(adapterName?, datasetName?, numSamples?, numEpochs?, learningRate?, loraR?, loraAlpha?)` → `{ ok, error }`
- `loadAdapter(adapterName)` → `{ ok, error, activeAdapter }` — load a saved LoRA adapter
- `useBaseModel` → `{ ok, activeAdapter }` — unload adapter, revert to base model

**Subscriptions**
- `inferStream(prompt, webSearch?)` → `TokenChunk` — `{ token: String, done: Boolean }` — token-by-token streaming

## GPU Support

The `python-worker` service has the NVIDIA `deploy` block active in `docker-compose.yml`. Requires the NVIDIA Container Toolkit on the host:

```bash
./scripts/install-nvidia.sh   # then reboot
```

## Frontend Tabs

**Chat** — bubble-style streaming messages. Tokens appear as they arrive over WebSocket. Web search toggle (🌐 Off/On) in the input bar. The header always shows the active model (base or LoRA adapter name).

**Fine-tune** — adapter management and training:
- *Models* section: lists base model + all saved adapters with Load/Unload buttons. Active model is highlighted.
- *Train new LoRA adapter* form: configure adapter name, dataset, samples, epochs, learning rate, LoRA r/alpha.
- *Training progress* bar: polls every 2 s during training, shows step/total/loss.

**GraphQL Explorer** — embedded GraphiQL v3. Schema introspection, query/mutation/subscription builder, variables panel, history, docs sidebar.

## Extending

- **Auth**: Add a Tower middleware layer in Rust before the GraphQL handler
- **Queuing**: Add Redis to compose; Rust enqueues, Python dequeues
- **Multiple models**: Add a `model: String` arg to the GraphQL mutation, maintain a model registry dict in `worker.py`
- **Merging adapters**: Use `peft`'s `merge_and_unload()` to bake a LoRA adapter permanently into the base model weights
- **Custom datasets**: Any HuggingFace dataset ID works in the fine-tune form as long as it has `instruction`/`input`/`output` columns (Alpaca format). Adapt `_format_alpaca()` in `worker.py` for other schemas.
