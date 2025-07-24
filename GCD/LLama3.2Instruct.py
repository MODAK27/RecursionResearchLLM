import sys
import os
import re
import random
import time
import torch
from math import gcd
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList, pipeline
import re
from constants import *
# ── Debugger setup ──────────────────────────────────────────────────────────
sys.path.append("/home1/dmodak/pycharm_egg_debugger/pydevd-pycharm.egg")
print("Tracing...")
import pydevd_pycharm
pydevd_pycharm.settrace(
    'localhost',
    port=12342,
    stdoutToServer=True,
    stderrToServer=True,
    suspend=False
)
print("Debugger attached!")


# ── Configuration ──────────────────────────────────────────────────────────
NUM_SAMPLES    = 1
MAX_NEW_TOKENS = 1024

# Few-shot messages for chat-style Llama-3.2-3B-Instruct
FEW_SHOT_MESSAGES = [
    {"role": "system",    "content": "You are a helpful assistant that solves math problems step by step."},
    {"role": "user",      "content": "Compute gcd(98, 56). Let's think step by step:"},
    {"role": "assistant", "content": (
        "1. 98 > 56, so 98 % 56 = 42\n"
        "2. 56 > 42, so 56 % 42 = 14\n"
        "3. 42 > 14, so 42 % 14 = 0\n"
        "Answer: 14"
    )},
    {"role": "user",      "content": "Compute gcd(22, 114). Let's think step by step:"},
    {"role": "assistant", "content": (
        "1. 114 > 22, so 114 % 22 = 4\n"
        "2. 22 > 4, so 22 % 4 = 2\n"
        "3. 4 > 2, so 4 % 2 = 0\n"
        "Answer: 2"
    )},
]

# ── Helpers ─────────────────────────────────────────────────────────────────
def sample_pairs(n):
    return [(random.randint(1, 500), random.randint(1, 500)) for _ in range(n)]

def extract_prediction(output_text):
    # ensure we have a string
    if isinstance(output_text, list):
        output_text = output_text[0]
    if isinstance(output_text, dict):
        # chat pipeline may return {'role':..., 'content':...}
        output_text = output_text.get('content') or output_text.get('generated_text') or str(output_text)
    matches = re.findall(r"\b(\d+)\b", output_text or "")
    return int(matches[-1]) if matches else None

# ── Device & dtype setup ─────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DTYPE  = torch.float16
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    DTYPE  = torch.bfloat16
else:
    DEVICE = torch.device("cpu")
    DTYPE  = torch.float32
print(f"Using device: {DEVICE}, dtype: {DTYPE}")

# ── Load tokenizer & model manually (no accelerate) ──────────────────────────
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
model     = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=DTYPE)
model.to(DEVICE)
model.eval()

# ── Create chat-style pipeline with preloaded model/tokenizer ─────────────────
pipe = pipeline(
    "text-generation",                           # use text-generation for chat-format messages
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=MAX_NEW_TOKENS,
    pad_token_id=tokenizer.eos_token_id,          # ensure proper padding
    device=0 if DEVICE.type == "cuda" else -1,
)

# ── Benchmark loop ─────────────────────────────────────────────────────────
correct = 0
start   = time.time()

test_pairs = sample_pairs(NUM_SAMPLES)
for a, b in test_pairs:
    # construct chat-style messages as a list of dicts
    messages = FEW_SHOT_MESSAGES + [
        {"role": "user", "content": f"Compute gcd({a}, {b}). Let's think step by step:"}
    ]
    # pass list of messages directly to text-generation pipeline
    response = pipe(
        messages,
        do_sample=False,
    )
    # extract generated_text from response
    text = response[0]["generated_text"]
    pred = extract_prediction(text)
    if pred == gcd(a, b):
        correct += 1

elapsed = time.time() - start
print(f"Accuracy: {correct}/{len(test_pairs)} = {correct/len(test_pairs)*100:.1f}%")
print(f"Avg time per sample: {elapsed/len(test_pairs)*1000:.1f} ms")

prompt_ids = tokenizer(prompt, return_tensors="pt")
prompt_len = prompt_ids["input_ids"].shape[1]
inputs = prompt_ids.to(DEVICE)
# print("input shape", len(inputs_["input_ids"][0]))