import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from ddgs import DDGS

app = FastAPI()

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_ID     = os.getenv("MODEL_ID",     "meta-llama/Llama-3.2-3B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN",     None)
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "false").lower() == "true"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model loading (once at startup) ──────────────────────────────────────────

print(f"Loading {MODEL_ID} on {DEVICE} (4-bit={LOAD_IN_4BIT})...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)

if LOAD_IN_4BIT and DEVICE == "cuda":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        token=HF_TOKEN,
    )

model.eval()
if DEVICE == "cuda":
    print(f"Model ready. VRAM used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
else:
    print("Model ready. Device: cpu")

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

# ── Schema ────────────────────────────────────────────────────────────────────

class InferRequest(BaseModel):
    prompt: str
    web_search: bool = False
    max_new_tokens: int = 512
    temperature: float = 0.7

class InferResponse(BaseModel):
    text: str

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    vram = f"{torch.cuda.memory_allocated() / 1e9:.1f} GB" if DEVICE == "cuda" else "n/a"
    return {"status": "ok", "device": DEVICE, "model": MODEL_ID, "vram_used": vram}


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest) -> InferResponse:
    if req.web_search:
        search_results = search_web(req.prompt)
        if search_results:
            print(f"[infer] injecting search context ({len(search_results)} chars)")
            messages = [
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
                {"role": "user", "content": req.prompt},
            ]
        else:
            print("[infer] search returned nothing, falling back to plain prompt")
            messages = [{"role": "user", "content": req.prompt}]
    else:
        messages = [{"role": "user", "content": req.prompt}]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            do_sample=req.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    result = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return InferResponse(text=result)
