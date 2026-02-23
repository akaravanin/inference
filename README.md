# Inference Server

A self-hosted LLM inference stack with a React chat UI, Rust GraphQL API, and Python model worker.

```
React Frontend (port 3000)
    ↓ GraphQL (HTTP)
Rust API Server (port 8080)   — axum + async-graphql
    ↓ HTTP (internal)
Python Model Worker           — HuggingFace transformers
    ↓
GPU inference (CUDA)
```

**Features**
- Chat UI with optional live web search (DuckDuckGo RAG)
- GraphQL Explorer tab (GraphiQL v3) for schema introspection
- 4-bit quantisation support via bitsandbytes
- LoRA / QLoRA ready (peft included)
- Model weights cached in a named Docker volume — no re-download on restart

---

## Prerequisites

- Docker + Docker Compose v2 (`docker compose version`)
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit ([install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- HuggingFace account with access to your chosen model

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
# Edit .env and add your HuggingFace token
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
| GraphQL Explorer | http://localhost:3000 (second tab) |
| GraphQL endpoint | http://localhost:8080/graphql |

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

### Switching models

```bash
# Llama 3.1 8B in 4-bit (~4.5 GB VRAM)
MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
LOAD_IN_4BIT=true
```

---

## Architecture

See [CLAUDE.md](CLAUDE.md) for full design decisions and extension guide.

---

## Project Structure

```
├── rust-server/          Axum + async-graphql API
├── python-worker/        FastAPI + transformers inference worker
├── frontend/             React + Vite + Apollo Client chat UI
├── scripts/              start / stop / restart / install-nvidia
├── .env.example          Environment variable template
├── docker-compose.yml
├── CLAUDE.md             Architecture & design decisions
└── README.md
```

---

## Future: LoRA Fine-tuning

The worker already includes `peft` and `bitsandbytes`. To fine-tune:

```python
from peft import get_peft_model, LoraConfig

config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, config)
```

---

## License

MIT
