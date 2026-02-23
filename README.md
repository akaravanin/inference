# Inference Server

A self-hosted LLM inference stack with a React chat UI, Rust GraphQL API, and Python model worker.

```
React Frontend (port 3000)
    ↓ GraphQL query / mutation (HTTP)
    ↓ GraphQL subscription (WebSocket)
Rust API Server (port 8080)   — axum + async-graphql
    ↓ HTTP + SSE (internal Docker network)
Python Model Worker           — FastAPI + HuggingFace transformers
    ↓
GPU inference (CUDA)
```

**Features**
- Streaming chat UI — tokens appear as they are generated (GraphQL subscriptions over WebSocket)
- Optional live web search (DuckDuckGo RAG) injected as context before inference
- LoRA / QLoRA fine-tuning via the Fine-tune tab — train, manage, and switch adapters without restarting
- GraphQL Explorer tab (GraphiQL v3) for schema introspection and ad-hoc queries
- 4-bit quantisation support via bitsandbytes (`LOAD_IN_4BIT=true`)
- Model weights and trained adapters cached in named Docker volumes — no re-download on restart

---

## Prerequisites

- Docker + Docker Compose v2 (`docker compose version`)
- NVIDIA GPU with CUDA support (tested on RTX 3070, 11.57 GB VRAM)
- NVIDIA Container Toolkit ([install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- HuggingFace account with access to [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

---

## Quick Start

### 1. Install NVIDIA drivers (first time only, requires reboot)

```bash
./scripts/install-nvidia.sh
sudo reboot
```

Verify after reboot:
```bash
nvidia-smi
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set HF_TOKEN to your HuggingFace access token
```

Accept the model license at [huggingface.co/meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) if you haven't already.

### 3. Start

```bash
./scripts/start.sh
```

First run downloads ~6 GB of model weights — watch progress with:
```bash
docker logs -f inference-server-python-worker-1
```

**URLs once running:**

| Service | URL |
|---------|-----|
| Chat UI | http://localhost:3000 |
| Fine-tune UI | http://localhost:3000 (second tab) |
| GraphQL Explorer | http://localhost:3000 (third tab) |
| GraphQL endpoint | http://localhost:8080/graphql |
| GraphQL WebSocket | ws://localhost:8080/graphql/ws |

---

## Scripts

```bash
./scripts/start.sh      # build + start all containers (detached)
./scripts/stop.sh       # stop and remove containers
./scripts/restart.sh    # full rebuild + restart
```

---

## Configuration

All config lives in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | HuggingFace access token (required for gated models) |
| `MODEL_ID` | `meta-llama/Llama-3.2-3B-Instruct` | Any HF causal LM |
| `LOAD_IN_4BIT` | `false` | Enable 4-bit quantisation (needed for 7B+ on ≤12 GB VRAM) |
| `ADAPTER_DIR` | `/adapters` | Mount point for saved LoRA adapters (Docker volume) |

### Switching models

```bash
# Llama 3.1 8B in 4-bit (~4.5 GB VRAM)
MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
LOAD_IN_4BIT=true
```

---

## LoRA Fine-tuning

Open the **Fine-tune** tab in the UI. You can:

1. **Train** a new LoRA adapter on any HuggingFace dataset (defaults to `tatsu-lab/alpaca`, 500 samples, 1 epoch — completes in ~5 minutes)
2. **Load** any saved adapter to serve inference with it
3. **Switch** back to the base model at any time without restarting

### Memory strategy

The base model (~6 GB float16) is offloaded to CPU during training, freeing VRAM for a fresh 4-bit QLoRA training copy (~1.5 GB). After training completes, the base model is restored to GPU automatically.

Typical training peak VRAM: ~5–6 GB (for 3B model, 500 samples).

### Adapters are persisted

Trained adapters are saved to the `adapters` Docker volume at `/adapters/<name>/`. They survive container restarts and are listed automatically in the Fine-tune tab.

---

## Architecture

See [CLAUDE.md](CLAUDE.md) for full design decisions and extension guide.

---

## Project Structure

```
├── rust-server/          Axum + async-graphql API + WebSocket subscriptions
├── python-worker/        FastAPI + transformers inference + LoRA fine-tuning
├── frontend/             React + Vite + Apollo Client chat + fine-tune UI
├── scripts/              start / stop / restart / install-nvidia
├── .env.example          Environment variable template
├── docker-compose.yml
├── CLAUDE.md             Architecture & design decisions
└── README.md
```

---

## License

MIT
