import sys
import os
import re
import random
import time
import torch
from math import gcd
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import re

from constants import *

# ── Debugger setup ──────────────────────────────────────────────────────────
# sys.path.append("/home1/dmodak/pycharm_egg_debugger/pydevd-pycharm.egg")
# print("Tracing...")
# import pydevd_pycharm
# pydevd_pycharm.settrace(
#     'localhost',
#     port=12342,
#     stdoutToServer=True,
#     stderrToServer=True,
#     suspend=False
# )
# print("Debugger attached!")

# ── Device setup ────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_built() and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# ── Configuration ──────────────────────────────────────────────────────────
MODEL_NAMES = [
    # "gpt2",
    # "xinchen9/Mistral-7B-CoT",
    "Qwen/Qwen2-7B",
    "Qwen/Qwen2-Math-7B",
    # "Qwen/Qwen2-7B-Instruct",
    # "Qwen/Qwen2-Math-7B-Instruct",
    # "meta-llama/Llama-3.2-3B-Instruct",
    # "mistralai/Mistral-7B-Instruct-v0.3"
    # "meta-math/MetaMath-7B-V1.0",
]
NUM_SAMPLES    = 100
MAX_NEW_TOKENS = 512
# COT style was of Mistral-7B-CoT on KAIST collection
PROMPT_TEMPLATE_COT_STYLE = {
    "prompt_template": '''
            In this task, you need to compute the greatest common divisor of two numbers.\n\n"
            "Problem: Compute gcd({a}, {b}).\n"
            "A: Let's think step by step:
            ''',
    "few_shot_prompt_template": """
        Given a simple high-school level math question, you are required to solve it and provide the final answer. The final answer is always a single number.

        Problem: Compute gcd(98, 56).
        Let's think step by step:
        1. 98 > 56 ⇒ 98 % 56 = 42
        2. 56 > 42 ⇒ 56 % 42 = 14
        3. 42 > 14 ⇒ 42 % 14 = 0
        Answer: 14

        Problem: Compute gcd(22, 114).
        Let's think step by step:
        1. 114 > 22 ⇒ 114 % 22 = 4
        2. 22 > 4 ⇒ 22 % 4 = 2
        3. 4 > 2 ⇒ 4 % 2 = 0
        Answer: 2

        Now solve:
        Problem: Compute gcd({a}, {b}).
        Let's think step by step:"""
}
PROMPT_TEMPLATE_META_MATH = {
        "prompt_template": ''' 
            "Below is an instruction that describes a task. " "Write a response that appropriately completes the request.\n\n" "### Instruction:\nCompute gcd({a}, {b}).\n\n### Response: Let's think step by step."
            ''',
        "few_shot_prompt_template": ''' 
            "Below is an instruction that describes a task. " "Write a response that appropriately completes the request.\n\n" 
            "### Instruction:\nCompute gcd(98, 56).\n\n### Response: Let's think step by step. \n 1. 98 > 56 ⇒ 98 % 56 = 42. \n 2. 56 > 42 ⇒ 56 % 42 = 14. \n 3. 42 > 14 ⇒ 42 % 14 = 0. \n Answer: 14"
        
            "### Instruction:\nCompute gcd(22, 114).\n\n### Response: Let's think step by step. \n 1. 114 > 22 ⇒ 114 % 22 = 4. \n 2. 22 > 4 ⇒ 22 % 4 = 2 \n 3. 4 > 2 ⇒ 4 % 2 = 0. \n Answer: 2"
        
            "### Instruction:\nCompute gcd({a}, {b}).\n\n### Response: Let's think step by step."
            '''
    }
FEW_SHOT_MESSAGES_STYLE_2 = {
    "prompt_template": '''
        Compute gcd({a}, {b}). Let's think step by step
     ''',
    "few_shot_prompt_template" : '''
        "Find the Gcd({a}, {b}). Let's think step by step"
        "To compute the greatest common divisor (gcd) of 98 and 56, we can use the Euclidean algorithm. Here's how it works step-by-step:\n\n"
        "1. **First Division Step**: Divide the larger number by the smaller number.\n"
        "   - \\(98 \\div 56 = 1\\) remainder \\(42\\)\n\n"
        "2. **Second Division Step**: Now, take the divisor from the first step (56) and divide it by the remainder from the first step (42).\n"
        "   - \\(56 \\div 42 = 1\\) remainder \\(14\\)\n\n"
        "3. **Third Division Step**: Divide 42 by 14.\n"
        "   - \\(42 \\div 14 = 3\\) remainder \\(0\\)\n\n"
        "Since we've reached a remainder of 0, the last non-zero remainder is the greatest common divisor (gcd) of the original two numbers.\n\n"
        "Therefore, the gcd of 98 and 56 is **14**."
        
        Now, Compute gcd({a}, {b}). Let's think step by step:
        '''
}
MODEL_CONFIGS = {
    "meta-math/MetaMath-7B-V1.0": PROMPT_TEMPLATE_META_MATH,
    "xinchen9/Mistral-7B-CoT": PROMPT_TEMPLATE_COT_STYLE,
    "meta-llama/Llama-3.2-3B-Instruct": PROMPT_TEMPLATE_COT_STYLE,
    "Qwen/Qwen2-7B" : FEW_SHOT_MESSAGES_STYLE_2,
    "Qwen/Qwen2-Math-7B" : FEW_SHOT_MESSAGES_STYLE_2,
    "Qwen/Qwen2-7B-Instruct": PROMPT_TEMPLATE_COT_STYLE,
    "Qwen/Qwen2-Math-7B-Instruct": PROMPT_TEMPLATE_COT_STYLE,
}
USE_FEWSHOT = False


# ── Helpers ─────────────────────────────────────────────────────────────────

import re

class PredictionExtractor:
    def __init__(self):
        """
        1) Try to find any occurrence of:
           - answer, ans or result
           - optionally followed by : or = or is
           - optional whitespace
           - then a (possibly negative) integer
        2) If none of those are found, fall back to grabbing the last standalone integer.
        Returns an int or None.
        """
        self.key_pattern = re.compile(
            r"\b(?:answer|ans|result)\b(?:\s+is)?\s*[:=]?\s*(-?\d+)",
            re.IGNORECASE
        )
        self.fallback_pattern = re.compile(r"\b-?\d+\b")  #in text d+ means one or more digits and \b word boundary, so it won’t match digits embedded in letters (“x123y”).

        # counters
        self.count_key     = 0   # matched answer/ans/result pattern
        self.count_fallback= 0   # fell back to any integer
        self.count_none    = 0   # found nothing

    def extract_prediction(self, output_text):
        """
        Tries in order:
        1) answer/ans/result (with optional 'is', ':' or '=') → count_key
        2) any standalone integer → count_fallback
        3) nothing → count_none
        Returns int or None.
        """

        key_matches = self.key_pattern.findall(output_text)
        if key_matches:
            self.count_key += 1
            return int(key_matches[-1])

        # 2) fallback: any standalone integer
        fallback_matches = self.fallback_pattern.findall(output_text)
        if fallback_matches:
            self.count_fallback += 1
            return int(fallback_matches[-1])

        # 3) no number
        self.count_none += 1
        return None

    def summary(self):
        total = self.count_key + self.count_fallback + self.count_none
        return {
            "total":        total,
            "key_matches":  self.count_key,
            "fallback":     self.count_fallback,
            "none":         self.count_none,
        }

def sample_pairs(n):
    return [(random.randint(1, 500), random.randint(1, 500)) for _ in range(n)]


# class AnswerNumberStoppingCriteria(StoppingCriteria):
#     def __init__(self, tokenizer, prompt_len):
#         super().__init__()
#         self.tokenizer = tokenizer
#         self.prompt_len = prompt_len
#         # matches “answer:” or “result:” (case-insensitive) followed by digits
#         self.pattern = re.compile(r"(?:answer|result):\s*\d+", re.IGNORECASE)
#         self.hit_count = 0
#
#     def __call__(self, input_ids, scores, **kwargs):
#         gen_ids = input_ids[0][self.prompt_len:]
#         text    = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
#         if self.pattern.search(text):
#             self.hit_count += 1
#             return True
#         return False



# ── Benchmark loop ──────────────────────────────────────────────────────────
def benchmark_model(model_name, pairs):
    global stop_criteria
    print(f"\n=== Benchmarking {model_name} ===")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=False, trust_remote_code=True, token=HUGGINGFACEHUB_API_TOKEN
    )

    # Load the model and move it to the chosen device
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if DEVICE.type == "cuda" else None,
        use_cache=True,
        revision="main"
    ).to(DEVICE)
    model.eval()

    correct = 0
    total   = len(pairs)
    start   = time.time()

    extractor = PredictionExtractor()

    config = MODEL_CONFIGS[model_name]
    prompt_template = config["prompt_template"]
    few_shot_prompt_template = config["few_shot_prompt_template"]
    MAX_NEW_TOKENS_ALl =  -sys.maxsize - 1
    for a, b in pairs:
        if USE_FEWSHOT:
            prompt = few_shot_prompt_template.format(a=a, b=b)
        else:
            prompt = prompt_template.format(a=a, b=b)
        prompt_ids = tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_ids["input_ids"].shape[1]
        inputs = prompt_ids.to(DEVICE)
        MAX_NEW_TOKENS_ = max(prompt_len*1.5, MAX_NEW_TOKENS)
        MAX_NEW_TOKENS_ALl = max(MAX_NEW_TOKENS_,MAX_NEW_TOKENS_ALl)
        # print("input shape", len(inputs_["input_ids"][0]))
        stop_criteria = None
        # stop_criteria = StoppingCriteriaList(
        #     [AnswerNumberStoppingCriteria(tokenizer, prompt_len)]
        # )
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS_,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                # stopping_criteria=stop_criteria
            )
        text   = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        # print("---",text)
        pred   = extractor.extract_prediction(text)
        # print(pred)
        actual = gcd(a, b)
        # print(actual)
        if pred == actual:
            correct += 1

    elapsed = time.time() - start
    acc     = correct / total * 100

    # Console output
    print(f"Accuracy: {correct}/{total} = {acc:.1f}%")
    print(f"Avg time per sample: {elapsed/total*1000:.1f} ms")

    # Write only the summary to output.txt
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path   = os.path.join(script_dir, "output.txt")
    with open(log_path, "a") as f:
        f.write(f"USE_FEWSHOT : {USE_FEWSHOT} and stop_criteria : {stop_criteria} and max_tokens_length allowed : {MAX_NEW_TOKENS_ALl} Accuracy for {model_name}: {correct}/{total} = {acc:.1f}%\n")
        f.write(f"Avg time per sample: {elapsed/total*1000:.1f} ms\n")
        print("Extraction summary:", extractor.summary(), file=f)

if __name__ == "__main__":
    test_pairs = sample_pairs(NUM_SAMPLES)
    for model_name in MODEL_NAMES:
        benchmark_model(model_name, test_pairs)
