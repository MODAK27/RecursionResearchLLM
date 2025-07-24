import sys
import os
import re
import random
import time
import torch
import multiprocessing as mp
from math import gcd
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList, pipeline
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

# ── Configuration ──────────────────────────────────────────────────────────
NUM_SAMPLES    = 100
MAX_NEW_TOKENS = 1024
USE_FEWSHOT = True

FEW_SHOT_MESSAGES_STYLE_1 = [
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
FEW_SHOT_MESSAGES_STYLE_2 = [
    {"role": "system",    "content": "You are a helpful assistant that solves math problems step by step."},

    {"role": "user",      "content": "Compute gcd(98, 56)."},
    {"role": "assistant", "content": (
        "To compute the greatest common divisor (gcd) of 98 and 56, we can use the Euclidean algorithm. Here's how it works step-by-step:\n\n"
        "1. **First Division Step**: Divide the larger number by the smaller number.\n"
        "   - \\(98 \\div 56 = 1\\) remainder \\(42\\)\n\n"
        "2. **Second Division Step**: Now, take the divisor from the first step (56) and divide it by the remainder from the first step (42).\n"
        "   - \\(56 \\div 42 = 1\\) remainder \\(14\\)\n\n"
        "3. **Third Division Step**: Divide 42 by 14.\n"
        "   - \\(42 \\div 14 = 3\\) remainder \\(0\\)\n\n"
        "Since we've reached a remainder of 0, the last non-zero remainder is the greatest common divisor (gcd) of the original two numbers.\n\n"
        "Therefore, the gcd of 98 and 56 is **14**."
    )},

    {"role": "user",      "content": "Compute gcd(85, 489)."},
    {"role": "assistant", "content": (
        "To compute the greatest common divisor (gcd) of 85 and 489, we can use the Euclidean algorithm. Here's how it works step-by-step:\n\n"
        "1. **First Division Step**: Divide the larger number by the smaller number.\n"
        "   - \\(489 \\div 85 = 5\\) remainder \\(64\\)\n\n"
        "2. **Second Division Step**: Replace the larger number with the smaller number from the first step and the smaller number with the remainder from the first step, then divide again.\n"
        "   - \\(85 \\div 64 = 1\\) remainder \\(21\\)\n\n"
        "3. **Third Division Step**: Repeat the process.\n"
        "   - \\(64 \\div 21 = 3\\) remainder \\(1\\)\n\n"
        "4. **Fourth Division Step**: Continue with the division.\n"
        "   - \\(21 \\div 1 = 21\\) remainder \\(0\\)\n\n"
        "Since we've reached a remainder of 0, the divisor at this stage, which is 1, is the greatest common divisor (gcd).\n\n"
        "Therefore, **gcd(85, 489) = 1**. This means that 85 and 489 are coprime, i.e., they have no common divisors other than 1."
    )},
]

# ── Configuration ──────────────────────────────────────────────────────────
MODEL_NAMES = [
    # "gpt2",
    # "xinchen9/Mistral-7B-CoT",
    "Qwen/Qwen2-7B-Instruct",
    "Qwen/Qwen2-Math-7B-Instruct",
    # "mistralai/Mistral-7B-Instruct-v0.3"
    # "meta-llama/Llama-3.2-3B-Instruct",
    # "mistralai/Mistral-7B-Instruct-v0.3"
    # "meta-math/MetaMath-7B-V1.0",
]
class PredictionExtractor:
    def __init__(self):
        # 0) “sentence that has  (gcd) of (…) and (…) is (…)”
        self.sentence_pattern = re.compile(
            r"""(?ix)                                      # ignore case, verbose
            \b(?:gcd|greatest\s+common\s+divisor)\b        # “gcd” or the full phrase
            [^.]*?                                         # anything (except “.”) up to...
            \b(\d+)\b                                      #   group 1: first number
            [^.]*?                                         # skip to next number
            \b(\d+)\b                                      #   group 2: second number
            [^.]*?                                         # skip to next number
            \b(\d+)\b                                      #   group 3: the GCD result
            [^.]*\.                                        # rest of the sentence through the period
            """,
            flags=re.IGNORECASE | re.VERBOSE
        )

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
        self.count_sentence = 0
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
        sentence_pattern_matches = self.sentence_pattern.search(output_text)
        if sentence_pattern_matches:
            a, b, result = sentence_pattern_matches.groups()
            # print(a, b, result)
            self.count_sentence += 1
            return int(result)

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
            "sentence_match":self.count_sentence,
            "key_matches":  self.count_key,
            "fallback":     self.count_fallback,
            "none":         self.count_none,
        }


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

def run_on_gpu(device_id, model_name, test_pairs):
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    benchmark_model(model_name, test_pairs, device)


def benchmark_model(model_name, test_pairs, device):
    print(f"\n=== Benchmarking {model_name} ===")
    extractor = PredictionExtractor()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map={"": device}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    correct = 0
    MAX_NEW_TOKENS_ALl =  -sys.maxsize - 1
    total = len(test_pairs)
    for a, b in test_pairs:
        # construct chat-style messages as a list of dicts
        if USE_FEWSHOT:
            messages = FEW_SHOT_MESSAGES_STYLE_2 + [
                {"role": "user", "content": f"Compute gcd({a}, {b}). Let's think step by step:"}
            ]
        else:
            messages = [
                {"role": "user", "content": f"Compute gcd({a}, {b}). Let's think step by step:"}
            ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompt_ids = tokenizer([text], return_tensors="pt")
        prompt_len = prompt_ids["input_ids"].shape[1]
        model_inputs = prompt_ids.to(device)
        MAX_NEW_TOKENS_ = max(prompt_len * 1.5, MAX_NEW_TOKENS)
        MAX_NEW_TOKENS_ALl = max(MAX_NEW_TOKENS_, MAX_NEW_TOKENS_ALl)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS_,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print(response)

        pred = extractor.extract_prediction(response)
        # print("----------- ",pred)
        # print(gcd(a, b))
        if pred == gcd(a, b):
            correct += 1

    acc     = correct / total * 100
    print(f"Accuracy: {correct}/{len(test_pairs)} = {correct/len(test_pairs)*100:.1f}%")
    # Write only the summary to output.txt
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, "output.txt")
    with open(log_path, "a") as f:
        f.write(f"USE_FEWSHOT {USE_FEWSHOT} : and stop_criteria : {None} and max_tokens_length allowed : {MAX_NEW_TOKENS_ALl} Accuracy for {model_name}: {correct}/{total} = {acc:.1f}%\n")
        print("Extraction summary:", extractor.summary(), file=f)


if __name__ == "__main__":
    device = "cuda"  # the device to load the model onto
    test_pairs = sample_pairs(NUM_SAMPLES)
    half = len(test_pairs) // 2
    pairs_gpu0 = test_pairs[:half]
    pairs_gpu1 = test_pairs[half:]
    for model_name in MODEL_NAMES:
        mp.set_start_method("spawn")
        processes = []
        for gpu_id, pairs in enumerate([pairs_gpu0, pairs_gpu1]):
            p = mp.Process(target=run_on_gpu, args=(gpu_id, model_name, pairs))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
