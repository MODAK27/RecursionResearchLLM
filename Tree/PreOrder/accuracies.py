import sys
import os
import re
import random
import time
import torch
from math import gcd
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from Helpers.helper import TreeBuilder, NUM_SAMPLES

from constants import *

# ── Device setup ────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_built() and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# ── Configuration ──────────────────────────────────────────────────────────
USE_FEWSHOT = False
MAX_NEW_TOKENS = 2048
MODEL_NAMES = [
    # "gpt2",
    # "xinchen9/Mistral-7B-CoT",
    # "Qwen/Qwen2-7B",
    "Qwen/Qwen2-Math-7B",
    # "Qwen/Qwen2-7B-Instruct",
    # "Qwen/Qwen2-Math-7B-Instruct",
    # "meta-llama/Llama-3.2-3B-Instruct",
    # "mistralai/Mistral-7B-Instruct-v0.3"
    # "meta-math/MetaMath-7B-V1.0",
]
INSTRUCT_MODEL_NAMES = [
    "Qwen/Qwen2-7B-Instruct",
    "Qwen/Qwen2-Math-7B-Instruct",
    # "meta-llama/Llama-3.2-3B-Instruct",
    # "mistralai/Mistral-7B-Instruct-v0.3"
]
PROMPT_TEMPLATE_COT_STYLE = {
    "prompt_template": """Given a binary tree represented as a nested list, output its preorder traversal as a Python list of integers..

    """,
    "few_shot_prompt_template": """" s """
}

MODEL_CONFIGS = {
    # "meta-math/MetaMath-7B-V1.0": PROMPT_TEMPLATE_META_MATH,
    # "xinchen9/Mistral-7B-CoT": PROMPT_TEMPLATE_COT_STYLE,
    # "meta-llama/Llama-3.2-3B-Instruct": PROMPT_TEMPLATE_COT_STYLE,
    # "Qwen/Qwen2-7B" : FEW_SHOT_MESSAGES_STYLE_2,
    "Qwen/Qwen2-Math-7B": PROMPT_TEMPLATE_COT_STYLE,
    "Qwen/Qwen2-7B-Instruct": PROMPT_TEMPLATE_COT_STYLE,
    "Qwen/Qwen2-Math-7B-Instruct": PROMPT_TEMPLATE_COT_STYLE,
}


# ── OUTPUT EXTRACTOR ─────────────────────────────────────────────────────────
class PreOrderExtractor:
    """Extract the first Python‐style list of ints that appears after 'PreOrder Traversal:'."""

    def __init__(self):
        self.key_pattern = re.compile(
            r'Preorder\s+Traversal\s*:\s*`?\s*'  # “ Traversal:”
            r'(?:'  # start non-capturing group
            r'\[\s*(?P<bracket>-?\d+(?:\s*,\s*-?\d+)*)\s*\]'  # [n, n, …]
            r'|'  # OR
            r'(?P<arrow>-?\d+(?:\s*->\s*-?\d+)*)'  # n -> n -> …
            r'|'
            + r'(?P<flat>-?\d+(?:\s*,\s*-?\d+)*)(?:\.)?'  # n, n, n…  (optional “.”)
              r')',
            re.IGNORECASE
        )
        self.key_pattern2 = re.compile(
            r'(?:Preorder\s+Traversal\s*:\s*|Expected\s+output\s+is\s*)'  # match " Traversal:" or "Expected output is"
            r'[`*\s]*'  # allow backticks, asterisks, whitespace
            r'(?:'
            r'\[\s*(?P<bracket>-?\d+(?:\s*,\s*-?\d+)*)\s*\]'  # [n, n, …]
            r'|'
            r'(?P<arrow>-?\d+(?:\s*->\s*-?\d+)+)'  # n -> n -> … (require ≥1 arrow)
            r'|'
            r'(?P<flat>-?\d+(?:\s*,\s*-?\d+)*)(?:\.)?'  # n, n, n…  (optional “.”)
            r')'
            r'[\s`*]*',  # allow trailing backticks, asterisks, whitespace
            re.IGNORECASE
        )

        self.none = 0
        self.matches = 0

    def extract_prediction(self, text: str):
        all_matches = list(self.key_pattern.finditer(text))
        all_matches.extend(self.key_pattern2.finditer(text))
        if not all_matches:
            self.none += 1
            return None

        # pick the **last** match in the text
        m = all_matches[-1]
        self.matches += 1

        if m.group('bracket') is not None:
            parts = [s.strip() for s in m.group('bracket').split(',')]
        elif m.group('arrow') is not None:
            parts = [s.strip() for s in m.group('arrow').split('->')]
        else:
            raw = m.group('flat')
            parts = raw.split(',')
        # Convert to ints and return
        return [int(x) for x in parts]

    def summary(self):
        total = self.none + self.matches
        return {
            "total": total,
            "key_matches": self.matches,
            "none": self.none,
        }


def test_PreOrder_For_One_Sample():
    tree = TreeBuilder.generate_non_empty_tree(depth=3, max_val=20)
    nested = TreeBuilder.tree_to_nested_list(tree)
    ground_truth = []
    TreeBuilder.PreOrder_traversal(tree, ground_truth)
    # Show nested representation and ground truth
    print("Nested list representation of the tree:")
    print(nested)
    TreeBuilder.draw_tree(nested, save_path="Preorder/PreorderTree.png")
    print("\nGround-truth preorder traversal:")
    print(ground_truth)
    # Prepare prompt
    prompt = f"""Given a binary tree represented as a nested list, output its preorder traversal as a Python list of integers.

    Tree: {nested}

    Preorder traversal:"""
    print("\nPrompt:")
    print(prompt)
    # Simulate model output (in a real benchmark, you'd call model.generate)
    simulated_output = str(ground_truth)
    print("\nSimulated model output:")
    print(simulated_output)
    extractor = PreOrderExtractor()
    pred = extractor.extract(simulated_output)
    print("\nExtractor prediction:")
    print(pred)
    # Verify correctness
    print("\nDoes prediction match ground truth?", pred == ground_truth)


def generateTree():
    tree = TreeBuilder.generate_non_empty_tree(depth=3, max_val=20)
    nested = TreeBuilder.tree_to_nested_list(tree)
    ground_truth = []
    TreeBuilder.preorder_traversal(tree, ground_truth)
    # Show nested representation and ground truth
    # print("Nested list representation of the tree:")
    # print(nested)
    # TreeBuilder.draw_tree(nested, save_path="Preorder/PreorderTree.png")
    # print("\nGround-truth preorder traversal:")
    # print(ground_truth)
    return nested, ground_truth


# ── Benchmark loop ──────────────────────────────────────────────────────────
def benchmark_model(model_name, nested_list, groundTruth_list):
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
    total = NUM_SAMPLES
    extractor = PreOrderExtractor()

    config = MODEL_CONFIGS[model_name]
    prompt_template = config["prompt_template"]
    few_shot_prompt_template = config["few_shot_prompt_template"]
    MAX_NEW_TOKENS_ALl = -sys.maxsize - 1
    prompt_print = None
    for nested, groundTruth in zip(nested_list, groundTruth_list):
        prompt = '''Tree: {nested}.Print your final answer as PreOrder traversal:'''.format(nested=nested)
        if USE_FEWSHOT:
            prompt = few_shot_prompt_template + prompt
        else:
            prompt = prompt_template + prompt
        prompt_ids = tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_ids["input_ids"].shape[1]
        inputs = prompt_ids.to(DEVICE)
        MAX_NEW_TOKENS_ = max(prompt_len * 1.5, MAX_NEW_TOKENS)
        MAX_NEW_TOKENS_ALl = max(MAX_NEW_TOKENS_, MAX_NEW_TOKENS_ALl)
        prompt_print = prompt
        # print("input shape", len(inputs_["input_ids"][0]))
        # stop_criteria = None
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
        response = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        # print("---", response)
        pred = extractor.extract_prediction(response)
        # print(pred)
        # print(groundTruth)
        if pred == groundTruth:
            correct += 1

    acc = correct / total * 100

    # Console output
    # print(f"Accuracy: {correct}/{total} = {acc:.1f}%")
    stop_criteria = None
    # print("i printed prediction:", pred)
    # print("ground truth:", groundTruth)
    # Write only the summary to output.txt
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, "output.txt")
    with open(log_path, "a") as f:
        f.write(
            f"\nUSE_FEWSHOT : {USE_FEWSHOT} and stop_criteria : {stop_criteria} and max_tokens_length allowed : {MAX_NEW_TOKENS_ALl} Accuracy for {model_name}: {correct}/{total} = {acc:.1f}%\n")
        print("Extraction summary:", extractor.summary(), file=f)
        print(prompt_print, file=f)
        print("i printed prediction:", pred, file=f)
        print("ground truth:", groundTruth, file=f)
        print("response:", response, file=f)


def benchmark_model_instruct(model_name, nested_list, groundTruth_list):
    print(f"\n=== Benchmarking {model_name} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the model and move it to the chosen device
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    ).to("cuda")

    correct = 0
    total = NUM_SAMPLES
    extractor = PreOrderExtractor()

    config = MODEL_CONFIGS[model_name]
    prompt_template = config["prompt_template"]
    few_shot_prompt_template = config["few_shot_prompt_template"]
    MAX_NEW_TOKENS_ALl = -sys.maxsize - 1
    prompt_print = None
    for nested, groundTruth in zip(nested_list, groundTruth_list):
        prompt = "Tree: {nested}. Print your final answer as PreOrder traversal:".format(nested=nested)
        if USE_FEWSHOT:
            messages = few_shot_prompt_template + prompt
        else:
            messages = [{"role": "system", "content": "You are an expert at preorder traversals."},
                {"role": "user", "content": prompt_template + prompt},
            ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompt_ids = tokenizer([text], return_tensors="pt")
        prompt_len = prompt_ids["input_ids"].shape[1]
        model_inputs = prompt_ids.to("cuda")
        MAX_NEW_TOKENS_ = max(prompt_len * 1.5, MAX_NEW_TOKENS)
        MAX_NEW_TOKENS_ALl = max(MAX_NEW_TOKENS_, MAX_NEW_TOKENS_ALl)
        prompt_print = messages
        # print("input shape", len(inputs_["input_ids"][0]))
        # stop_criteria = None
        # stop_criteria = StoppingCriteriaList(
        #     [AnswerNumberStoppingCriteria(tokenizer, prompt_len)]
        # )
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
        # print("---", response)
        pred = extractor.extract_prediction(response)
        # print(pred)
        # print(groundTruth)
        if pred == groundTruth:
            correct += 1

    acc = correct / total * 100

    # Console output
    # print(f"Accuracy: {correct}/{total} = {acc:.1f}%")
    stop_criteria = None
    # print("printed prediction:", pred)
    # print("ground truth:", groundTruth)
    # Write only the summary to output.txt
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, "output.txt")
    with open(log_path, "a") as f:
        f.write(
            f"\nUSE_FEWSHOT : {USE_FEWSHOT} and stop_criteria : {stop_criteria} and max_tokens_length allowed : {MAX_NEW_TOKENS_ALl} Accuracy for {model_name}: {correct}/{total} = {acc:.1f}%\n")
        print("Extraction summary:", extractor.summary(), file=f)
        print(prompt_print, file=f)
        print("printed prediction:", pred, file=f)
        print("ground truth:", groundTruth, file=f)
        print("response:", response, file=f)


# ── DEMONSTRATION FOR ONE DUMMY TREE ─────────────────────────────────────────
if __name__ == "__main__":

    log_path = os.path.join(os.path.dirname(__file__), "output.txt")
    #

    # truncate the file (make it empty)
    open(log_path, "w").close()
    with open(log_path, "a") as f:
        print("\n-----------PREORDER---------:\n", file=f)

    nested_list, groundTruth_list = TreeBuilder.generateDataset("Preorder")



    for model_name in INSTRUCT_MODEL_NAMES:
        benchmark_model_instruct(model_name, nested_list, groundTruth_list)

    for model_name in MODEL_NAMES:
        benchmark_model(model_name, nested_list, groundTruth_list)
