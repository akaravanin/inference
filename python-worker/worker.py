import json
import os
import threading
import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset
from ddgs import DDGS

app = FastAPI()

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_ID     = os.getenv("MODEL_ID",     "meta-llama/Llama-3.2-3B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN",     None)
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "false").lower() == "true"
ADAPTER_DIR  = os.getenv("ADAPTER_DIR",  "/adapters")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model loading (once at startup) ──────────────────────────────────────────

print(f"Loading {MODEL_ID} on {DEVICE} (4-bit={LOAD_IN_4BIT})...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if LOAD_IN_4BIT and DEVICE == "cuda":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN,
    )
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        token=HF_TOKEN,
    )

base_model.eval()
if DEVICE == "cuda":
    print(f"Model ready. VRAM used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
else:
    print("Model ready. Device: cpu")

# ── Active model (base or adapter-loaded) ────────────────────────────────────

# inference_model points to whichever model is currently serving requests
inference_model = base_model
active_adapter: str | None = None   # None means base model is active
model_lock = threading.Lock()

# ── Fine-tune state ───────────────────────────────────────────────────────────

finetune_status = {
    "running": False,
    "step": 0,
    "total_steps": 0,
    "loss": None,
    "error": None,
    "completed_adapter": None,   # name of the last successfully trained adapter
}
finetune_lock = threading.Lock()

# ── Web search ────────────────────────────────────────────────────────────────

def search_web(query: str, max_results: int = 4) -> str:
    print(f"[search] querying: {query!r}")
    try:
        results = DDGS().text(query, max_results=max_results)
        if not results:
            print("[search] no results returned")
            return ""
        print(f"[search] got {len(results)} results")
        lines = []
        for r in results:
            lines.append(f"Title: {r['title']}\nSummary: {r['body']}\nURL: {r['href']}")
        return "\n\n".join(lines)
    except Exception as e:
        print(f"[search] error: {e}")
        return ""

def build_messages(prompt: str, web_search: bool) -> list:
    if web_search:
        search_results = search_web(prompt)
        if search_results:
            print(f"[infer] injecting search context ({len(search_results)} chars)")
            return [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. "
                        "The user's question has been looked up on the web RIGHT NOW and the results are below. "
                        "Your knowledge cutoff is outdated — IGNORE IT and answer ONLY using the search results provided. "
                        "Do not say you don't know or reference your training cutoff.\n\n"
                        "=== LIVE WEB SEARCH RESULTS ===\n"
                        f"{search_results}\n"
                        "=== END OF SEARCH RESULTS ===\n\n"
                        "Answer the user's question based on the above results. Cite the URL sources."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
        print("[infer] search returned nothing, falling back to plain prompt")
    return [{"role": "user", "content": prompt}]

# ── Schema ────────────────────────────────────────────────────────────────────

class InferRequest(BaseModel):
    prompt: str
    web_search: bool = False
    max_new_tokens: int = 512
    temperature: float = 0.7

class InferResponse(BaseModel):
    text: str

class FineTuneRequest(BaseModel):
    adapter_name: str = "my-lora"
    dataset_name: str = "tatsu-lab/alpaca"
    num_samples: int = 500
    num_epochs: int = 1
    learning_rate: float = 2e-4
    lora_r: int = 8
    lora_alpha: int = 16

class LoadAdapterRequest(BaseModel):
    adapter_name: str

# ── LoRA fine-tuning helpers ──────────────────────────────────────────────────

class _ProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        with finetune_lock:
            finetune_status["step"] = state.global_step
            if logs and "loss" in logs:
                finetune_status["loss"] = round(logs["loss"], 4)


def _format_alpaca(sample) -> str:
    if sample.get("input", "").strip():
        return (
            f"### Instruction:\n{sample['instruction']}\n\n"
            f"### Input:\n{sample['input']}\n\n"
            f"### Response:\n{sample['output']}"
        )
    return (
        f"### Instruction:\n{sample['instruction']}\n\n"
        f"### Response:\n{sample['output']}"
    )


def _tokenize(sample, max_length: int = 512):
    text = _format_alpaca(sample)
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    enc["labels"] = enc["input_ids"].copy()
    return enc


def run_finetune(req: FineTuneRequest):
    """Runs in a background thread.

    Memory strategy (11 GB GPU):
      - The float16 inference model (~6 GB) is moved to CPU while training so the
        GPU is free for a fresh 4-bit (QLoRA) training copy (~1.5 GB) plus
        gradients, optimizer states, and activations (~3-4 GB).
      - After training the temporary model is deleted and the inference model is
        moved back to GPU, so chat resumes automatically.
    """
    global inference_model, active_adapter

    adapter_path = os.path.join(ADAPTER_DIR, req.adapter_name)
    train_model = None

    try:
        # ── Reset status ───────────────────────────────────────────────────
        with finetune_lock:
            finetune_status["step"] = 0
            finetune_status["loss"] = None
            finetune_status["error"] = None

        # ── Move inference model to CPU to free VRAM ───────────────────────
        if DEVICE == "cuda":
            print("[finetune] moving inference model to CPU to free VRAM...")
            with model_lock:
                base_model.cpu()
            torch.cuda.empty_cache()
            free_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
            print(f"[finetune] VRAM free after offload: {free_gb:.1f} GB")

        # ── Load dataset ───────────────────────────────────────────────────
        print(f"[finetune] loading dataset {req.dataset_name!r} ({req.num_samples} samples)...")
        ds = load_dataset(req.dataset_name, split=f"train[:{req.num_samples}]")
        ds = ds.map(_tokenize, remove_columns=ds.column_names)

        # batch_size=1, grad_accum=8 → effective batch 8 with minimal VRAM
        effective_batch = 8
        total_steps = max(1, (len(ds) // effective_batch) * req.num_epochs)
        with finetune_lock:
            finetune_status["total_steps"] = total_steps

        # ── Load fresh 4-bit model for QLoRA ──────────────────────────────
        print("[finetune] loading 4-bit training model (QLoRA)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        train_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto" if DEVICE == "cuda" else None,
            token=HF_TOKEN,
        )
        train_model = prepare_model_for_kbit_training(
            train_model, use_gradient_checkpointing=True
        )

        # ── Attach LoRA adapters ───────────────────────────────────────────
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=req.lora_r,
            lora_alpha=req.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )
        peft_model = get_peft_model(train_model, lora_cfg)
        peft_model.print_trainable_parameters()

        # ── Training ───────────────────────────────────────────────────────
        training_args = TrainingArguments(
            output_dir=adapter_path,
            num_train_epochs=req.num_epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=effective_batch,
            learning_rate=req.learning_rate,
            fp16=(DEVICE == "cuda"),
            logging_steps=5,
            save_strategy="no",
            report_to="none",
            optim="paged_adamw_8bit",   # memory-efficient optimizer
        )

        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=ds,
            data_collator=DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
            callbacks=[_ProgressCallback()],
        )

        print(f"[finetune] training for {req.num_epochs} epoch(s)...")
        trainer.train()

        # ── Save adapter ───────────────────────────────────────────────────
        os.makedirs(adapter_path, exist_ok=True)
        peft_model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        print(f"[finetune] adapter saved to {adapter_path}")

        with finetune_lock:
            finetune_status["completed_adapter"] = req.adapter_name
            finetune_status["running"] = False

    except Exception as e:
        print(f"[finetune] error: {e}")
        with finetune_lock:
            finetune_status["error"] = str(e)
            finetune_status["running"] = False

    finally:
        # ── Always: free training model, move inference model back to GPU ──
        if train_model is not None:
            del peft_model, train_model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            print("[finetune] moving inference model back to GPU...")
            with model_lock:
                base_model.cuda()
            print(f"[finetune] VRAM after restore: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    vram = f"{torch.cuda.memory_allocated() / 1e9:.1f} GB" if DEVICE == "cuda" else "n/a"
    return {"status": "ok", "device": DEVICE, "model": MODEL_ID, "vram_used": vram, "active_adapter": active_adapter}


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest) -> InferResponse:
    messages = build_messages(req.prompt, req.web_search)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    with model_lock:
        inputs = tokenizer([text], return_tensors="pt").to(inference_model.device)
        with torch.no_grad():
            outputs = inference_model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                do_sample=req.temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    return InferResponse(text=tokenizer.decode(generated_ids, skip_special_tokens=True))


@app.post("/stream")
def stream(req: InferRequest) -> StreamingResponse:
    messages = build_messages(req.prompt, req.web_search)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    with model_lock:
        inputs = tokenizer([text], return_tensors="pt").to(inference_model.device)
        current_model = inference_model  # capture ref before releasing lock

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    thread = threading.Thread(
        target=current_model.generate,
        kwargs={
            **inputs,
            "max_new_tokens": req.max_new_tokens,
            "temperature": req.temperature,
            "do_sample": req.temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
            "streamer": streamer,
        },
    )
    thread.start()

    def sse():
        for token in streamer:
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


@app.post("/fine-tune")
def start_finetune(req: FineTuneRequest):
    with finetune_lock:
        if finetune_status["running"]:
            return {"ok": False, "error": "Fine-tuning already in progress"}
        finetune_status["running"] = True
        finetune_status["step"] = 0
        finetune_status["total_steps"] = 0
        finetune_status["loss"] = None
        finetune_status["error"] = None

    t = threading.Thread(target=run_finetune, args=(req,), daemon=True)
    t.start()
    return {"ok": True}


@app.get("/fine-tune/status")
def get_finetune_status():
    with finetune_lock:
        return dict(finetune_status)


@app.get("/adapters")
def list_adapters():
    """Return all saved adapters on disk plus which one is currently active."""
    available = []
    if os.path.isdir(ADAPTER_DIR):
        for entry in sorted(os.scandir(ADAPTER_DIR), key=lambda e: e.name):
            # A valid saved adapter always has adapter_config.json
            if entry.is_dir() and os.path.exists(os.path.join(entry.path, "adapter_config.json")):
                available.append(entry.name)
    return {"available": available, "active": active_adapter}


@app.get("/model-info")
def model_info():
    vram = f"{torch.cuda.memory_allocated() / 1e9:.1f} GB" if DEVICE == "cuda" else "n/a"
    return {
        "model_id": MODEL_ID,
        "device": DEVICE,
        "vram_used": vram,
        "active_adapter": active_adapter,
        "load_in_4bit": LOAD_IN_4BIT,
    }


@app.post("/fine-tune/load")
def load_adapter(req: LoadAdapterRequest):
    global inference_model, active_adapter
    adapter_path = os.path.join(ADAPTER_DIR, req.adapter_name)
    if not os.path.isdir(adapter_path):
        return {"ok": False, "error": f"Adapter '{req.adapter_name}' not found at {adapter_path}"}
    try:
        with model_lock:
            # Discard any previously loaded adapter before loading a new one
            if inference_model is not base_model:
                del inference_model
                torch.cuda.empty_cache()
            print(f"[adapter] loading {req.adapter_name} from {adapter_path}")
            loaded = PeftModel.from_pretrained(base_model, adapter_path)
            loaded.eval()
            inference_model = loaded
            active_adapter = req.adapter_name
        print(f"[adapter] now serving with adapter: {req.adapter_name}")
        return {"ok": True, "active_adapter": active_adapter}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/fine-tune/unload")
def unload_adapter():
    global inference_model, active_adapter
    with model_lock:
        inference_model = base_model
        active_adapter = None
    print("[adapter] reverted to base model")
    return {"ok": True, "active_adapter": None}
