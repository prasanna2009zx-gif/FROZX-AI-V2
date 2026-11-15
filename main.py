import os
import secrets
import torch

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI(
    title="FROZX AI API",
    description="Official FROZX AI Intelligence API",
    version="2.0.0"
)

MODEL_ID = "prasanna-frozx-v2/FROZX-AI-V2"

tokenizer = None
model = None

@app.on_event("startup")
def load_model():
    global tokenizer, model

    token = os.getenv("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=token
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=token,
        torch_dtype=torch.float32,
        device_map="cpu"
    )

    model.eval()


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {
        "name": "FROZX AI",
        "status": "online",
        "version": "2.0.0"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest, x_api_key: str = Header(None)):

    expected_key = os.getenv("FROZX_API_KEY", "")

    if not expected_key or not x_api_key or not secrets.compare_digest(
        x_api_key, expected_key
    ):
        raise HTTPException(status_code=401, detail="Invalid API Key")

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model loading")

    messages = [
        {
            "role": "system",
            "content": "You are FROZX AI, a helpful Tamil and English AI assistant."
        },
        {
            "role": "user",
            "content": request.message
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7
        )

    answer = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    return {
        "success": True,
        "model": "FROZX AI V2",
        "response": answer
    }
