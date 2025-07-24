import sys
import os
import re
import copy
import random
import time
from xml.etree.ElementTree import TreeBuilder
from itertools import product
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
PROMPT_TEMPLATE_NON_COT_STYLE = {
    "prompt_template": """Tree in the form of nested list: {nested}. 
        Output its inorder traversal as a list of integers excluding empty branches and without printing `None` in the final output. Print your final answer as Inorder traversal:
        """,

    "few_shot_prompt_template": """ For a binary tree represented as a nested list, output its inorder traversal as a list of integers excluding empty branches and without printing `None` in the final output.
        Example1:
        given Tree in the form of nested list: [8, [1, [7, [], []], []], [8, [5, [], []], [10, [], []]]].
        Output its inorder traversal as a list of integers excluding empty branches and without printing `None` in the final output. Print your final answer as Inorder traversal:
        Inorder traversal: [7, 1, 8, 5, 8, 10]

        Example2:
        given Tree in the form of nested list: [8, [], [7, [], [[2],[],[]]]]. 
        Output its inorder traversal as a list of integers excluding empty branches and without printing `None` in the final output. Print your final answer as Inorder traversal:
        Inorder traversal: [8,7,2]
        
        Example3:
        given Tree in the form of nested list: [12, [2], [3,[],[2]]]. 
        Output its inorder traversal as a list of integers excluding empty branches and without printing `None` in the final output. Print your final answer as Inorder traversal:
        Inorder traversal: [2, 12, 3, 2]
        
        Example4:
        given Tree in the form of nested list: [13, [16, [], [12, [], []]], [11, [13, [], []], []]]. 
        Output its inorder traversal as a list of integers excluding empty branches and without printing `None` in the final output. Print your final answer as Inorder traversal:
        Inorder traversal: [16, 12, 13, 13, 11]
        
        Example5:
        given Tree in the form of nested list: [7, [3, [1, [], []], []], [9, [], [11, [], []]]]. 
        Output its inorder traversal as a list of integers excluding empty branches and without printing `None` in the final output. Print your final answer as Inorder traversal:
        Inorder traversal: [1, 3, 7, 9, 11]
        
        Now it's your turn:
        Now, given Tree in the form of nested list: {nested}. 
        Output its inorder traversal as a list of integers excluding empty branches and without printing `None` in the final output. Print your final answer as Inorder traversal: 
        """
}

PROMPT_TEMPLATE_COT_STYLE = {
    "prompt_template": """
        Tree in the form of nested list: {nested}. 
        Output its inorder traversal as a list of integers excluding empty branches and without printing `None` in the final output. 
        Think step by step and do reasoning behind and Print your final answer as Inorder traversal:
        """
    ,

    "few_shot_prompt_template":
        """
        Example 2:
        Tree in the form of nested list: [7, [3, [1, [], []], []], [9, [], [11, [], []]]].
        Output its inorder traversal as a list of integers excluding empty branches and without printing `None` in the final output. Print your final answer as Inorder traversal:

        Let’s think step by step:
        1. See node = [7]; left = [3,[1, [], []], []]; right = [9, [], [11, [], []]]. Inorder traversal: []
        2. Traverse in left subtree:
            2.1 See node = [3]; left = [1, [], []]; right = []. Inorder traversal: []
            2.2 Traverse in left subtree:
                2.3.1. see node = [1]; left = []; right = []. Inorder traversal: []
                2.3.2 Traverse in left subtree:
                    2.3.2.1 See node = [] Inorder traversal:[]
                2.3.3 Visit node =[1]; Inorder traversal: [1]
                2.3.4 In right subtree:
                    2.3.4.1 See node = [] Inorder traversal:[1]
            2.3 Visit node = [3]; Inorder traversal: [1, 3]
            2.4 In right subtree:
                2.4.1. node = []; Inorder traversal: [1,3]
        3. Visit node = [7]; Inorder traversal: [1, 3, 7]
        4. Traverse In right subtree:
            4.1 See node = [9]; left = []; right =  [11, [], []]. Inorder traversal: [1, 3, 7]
            4.2 In left subtree:
                4.2.1 see node = []; Inorder traversal: [1, 3, 7]
            4.3 Visit node = [9]; Inorder traversal: [1, 3, 7, 9]
            4.4 In right subtree:
                4.4.1. See node = [11]; left = []; right = []. Inorder traversal: [1, 3, 7, 9]
                4.4.2 Traverse in left subtree:
                    4.4.2.1 See node = [] Inorder traversal: [1, 3, 7, 9]
                4.4.3 Visit node = [11]; Inorder traversal: [1, 3, 7, 9, 11]
                4.4.4 Traverse in right subtree:
                    4.4.4.1 See node = [] Inorder traversal: [1, 3, 7, 9, 11]

        Answer Inorder traversal: [1, 3, 7, 9, 11]
        
        Example 3:
        Tree in the form of nested list: [12, [2], [3,[],[2]]].
        Output its inorder traversal as a list of integers excluding empty branches and without printing `None` in the final output. Print your final answer as Inorder traversal:

        Let’s think step by step:
        1. See node = [12]; left = [2]; right = [3,[],[2]]]. Inorder traversal: []
        2. Traverse in left subtree:
            2.1 See node = [2];  Inorder traversal: []
            2.2 Visit node = [2]; Inorder traversal: [2]
        3. Visit node = [12]; Inorder traversal: [2, 12]
        4. Traverse In right subtree:
            4.1 See node = [3]; left = []; right =  [2]. Inorder traversal: [2, 12]
            4.2 In left subtree:
                4.2.1 see node = []; Inorder traversal: [2, 12]
            4.3 Visit node = [3]; Inorder traversal: [2, 12, 3]
            4.4 In right subtree:
                4.4.1. See node = [2]; Inorder traversal: [2, 12, 3]
                4.4.2 Visit node = [2]; Inorder traversal: [2, 12, 3, 2]

        Answer Inorder traversal: [2, 12, 3, 2]
        
        Example 4:
        Tree in the form of nested list: [13, [16, [], [12, [], []]], [11, [13, [], []], []]].
        Output its inorder traversal as a list of integers excluding empty branches and without printing `None` in the final output. Print your final answer as Inorder traversal:

        Let’s think step by step:
        1. See node = [13]; left = [16, [], [12, [], []]]; right = [11, [13, [], []], []]. Inorder traversal: []
        2. Traverse in left subtree:
            2.1 See node = [16]; left = []; right = [12, [], []]. Inorder traversal: []
            2.2 Traverse in left subtree:
                2.3.1. see node = []; Inorder traversal: []
            2.3 Visit node = [16]; Inorder traversal: [16]
            2.4 Traverse in right subtree:
                2.4.1. See node = [12]; Inorder traversal: [16]
                2.4.2  Traverse in left subtree:
                    2.4.2.1 see node = []; Inorder traversal: [16]
                2.4.3  Visit node = [12]; Inorder traversal: [16, 12]
                2.4.4  Traverse in right subtree:
                    2.4.4.1 see node = []; Inorder traversal: [16, 12]
        3. Visit node = [13]; Inorder traversal: [16, 12, 13]
        4. In right subtree:
            4.1 See node = [11]; left = [13, [], []]; right =  []. Inorder traversal: [16, 12, 13]
            4.2 In left subtree:
                4.2.1 See node = [13]; left=[]; right=[] Inorder traversal: [16, 12, 13]
                4.2.2  Traverse in left subtree:
                    4.2.2.1 see node = []; Inorder traversal: [16, 12, 13]
                4.2.3  Visit node = [13]; Inorder traversal: [16, 12, 13, 13]
                4.2.4  Traverse in right subtree:
                    4.2.4.1 see node = []; Inorder traversal: [16, 12, 13, 13]
            4.3 Visit node = [11]; Inorder traversal: [16, 12, 13, 13, 11]
            4.4 In right subtree:
                4.4.1. See node = [];  Inorder traversal: [16, 12, 13, 13, 11]

        Answer Inorder traversal: [16, 12, 13, 13, 11]
        
        Example 5:
        Tree in the form of nested list: [8, [1, [7, [], []], []], [8, [5, [], []], [10, [], []]]].
        Output its inorder traversal as a list of integers excluding empty branches and without printing `None` in the final output. Print your final answer as Inorder traversal:

        Let’s think step by step:
        1. See node = [8]; left = [1, [7, [], []], []]; right = [8, [5, [], []], [10, [], []]]]. Inorder traversal: []
        2. Traverse in left subtree:
            2.1 See node = [1]; left = [7, [], []]; right = []. Inorder traversal: []
            2.2 Traverse in left subtree:
                2.3.1. see node = [7]; left = []; right = []. Inorder traversal: []
                2.3.2 Traverse in left subtree:
                    2.3.2.1 See node = [] Inorder traversal:[]
                2.3.3 Visit node =[7]; Inorder traversal: [7]
                2.3.4 Traverse In right subtree:
                    2.3.4.1 See node = [] Inorder traversal:[7]
            2.3 Visit node = [1]; Inorder traversal: [7, 1]
            2.4 Traverse In right subtree:
                2.4.1. See node = []; Inorder traversal: [7, 1]
        3. Visit node = [8]; Inorder traversal: [7, 1, 8]
        4. In right subtree:
            4.1 See node = [8]; left = [5, [], []]; right =  [10, [], []]. Inorder traversal: [7, 1, 8]
            4.2 In left subtree:
                4.2.1 See node = [5]; left=[]; right=[] Inorder traversal: [7, 1, 8]
                4.2.2  Traverse in left subtree:
                    4.2.2.1 see node = []; Inorder traversal: [7, 1, 8]
                4.2.3  Visit node = [5]; Inorder traversal: [7, 1, 8, 5]
                4.2.4  Traverse in right subtree:
                    4.2.4.1 see node = []; Inorder traversal: [7, 1, 8, 5]
            4.3 Visit node = [8]; Inorder traversal: [7, 1, 8, 5, 8]
            4.4 In right subtree:
                4.4.1. See node = [10];  Inorder traversal: [7, 1, 8, 5, 8]
                4.4.2  Traverse in left subtree:
                    4.4.2.1 see node = []; Inorder traversal: [7, 1, 8, 5, 8]
                4.4.3  Visit node = [10]; Inorder traversal: [7, 1, 8, 5, 8, 10]
                4.4.4  Traverse in right subtree:
                    4.4.4.1 see node = []; Inorder traversal: [7, 1, 8, 5, 8, 10]

        Answer Inorder traversal: [7, 1, 8, 5, 8, 10]
        
        Now it’s your turn:
        Tree in the form of nested list: {nested}.
        Output its inorder traversal as a list of integers excluding empty branches and without printing `None` in the final output. Print your final answer as Inorder traversal:
        Let’s think step by step:
        """

}

PROMPT_TEMPLATE_NON_COT_STYLE_INSTRUCT = {
    "few_shot_prompt_template": [
    {
        "role": "system",
        "content": "You are an expert at inorder traversal."
    },
    # — Example 1 —
    {
        "role": "assistant",
        "content": """Tree in the form of nested list: [8, [1,[],[]], []]. 
         Output its inorder traversal as a list of integers excluding empty branches and without printing `None` in the final output. Print your final answer as Inorder traversal:
         Inorder traversal: [7, 1, 8, 5, 8, 10]"""
    },
    # — Example 2 —
    {
        "role": "assistant",
        "content": """Tree in the form of nested list: [8, [], [7, [], [[2],[],[]]]]. 
         Output its inorder traversal as a list of integers excluding empty branches and without printing `None` in the final output. Print your final answer as Inorder traversal:
         Inorder traversal: [8,7,2]"""
    },
    {
        "role": "assistant",
        "content": """Tree in the form of nested list: [12, [2], [3,[],[2]]]. 
         Output its inorder traversal as a list of integers excluding empty branches and without printing `None` in the final output. Print your final answer as Inorder traversal:
         Inorder traversal: [2, 12, 3, 2]"""
    },
    {
        "role": "assistant",
        "content": """Tree in the form of nested list: [13, [16, [], [12, [], []]], [11, [13, [], []], []]]. 
         Output its inorder traversal as a list of integers excluding empty branches and without printing `None` in the final output. Print your final answer as Inorder traversal:
         Inorder traversal: [16, 12, 13, 13, 11]"""
    },
    {
        "role": "assistant",
        "content": """Tree in the form of nested list: [7, [3, [1, [], []], []], [9, [], [11, [], []]]]. 
         Output its inorder traversal as a list of integers excluding empty branches and without printing `None` in the final output. Print your final answer as Inorder traversal:
         Inorder traversal: [1, 3, 7, 9, 11]"""
    },
    # — Your turn —
    {
        "role": "user",
        "content": "Tree in the form of nested list: {nested}. "
                   "Output its inorder traversal as a list of integers excluding empty branches and without printing `None` in the final output. Print your final answer as Inorder traversal:"
    }]
    ,
    "prompt_template": [
        {
            "role": "system",
            "content": "You are an expert at inorder traversal."
        },
        {
            "role": "user",
            "content": "Tree in the form of nested list: {nested}. "
                       "Output its inorder traversal as a list of integers excluding empty branches and without printing `None` in the final output. Print your final answer as Inorder traversal:"
        }
    ]
}
PROMPT_TEMPLATE_COT_STYLE_INSTRUCT = {
    "few_shot_prompt_template": [
    {
        "role": "system",
        "content": "You are an expert at inorder traversal."
    },
    # — Example 1 —
    {
        "role": "assistant",
        "content": """
        Tree in the form of nested list: [12, [2], [3,[],[2]]].
        Output its inorder traversal as a list of integers excluding empty branches and without printing `None` in the final output. Print your final answer as Inorder traversal:

        Let’s think step by step:
        1. See node = [12]; left = [2]; right = [3,[],[2]]]. Inorder traversal: []
        2. Traverse in left subtree:
            2.1 See node = [2];  Inorder traversal: []
            2.2 Visit node = [2]; Inorder traversal: [2]
        3. Visit node = [12]; Inorder traversal: [2, 12]
        4. Traverse In right subtree:
            4.1 See node = [3]; left = []; right =  [2]. Inorder traversal: [2, 12]
            4.2 In left subtree:
                4.2.1 see node = []; Inorder traversal: [2, 12]
            4.3 Visit node = [3]; Inorder traversal: [2, 12, 3]
            4.4 In right subtree:
                4.4.1. See node = [2]; Inorder traversal: [2, 12, 3]
                4.4.2 Visit node = [2]; Inorder traversal: [2, 12, 3, 2]

        Answer Inorder traversal: [2, 12, 3, 2]
        """
    },
    # — Example 2 —
    {
        "role": "assistant",
        "content": """
        Example 2:
        Tree in the form of nested list: [7, [3, [1, [], []], []], [9, [], [11, [], []]]].
        Output its inorder traversal as a list of integers excluding empty branches and without printing `None` in the final output. Print your final answer as Inorder traversal:

        Let’s think step by step:
        1. See node = [7]; left = [3,[1, [], []], []]; right = [9, [], [11, [], []]]. Inorder traversal: []
        2. Traverse in left subtree:
            2.1 See node = [3]; left = [1, [], []]; right = []. Inorder traversal: []
            2.2 Traverse in left subtree:
                2.3.1. see node = [1]; left = []; right = []. Inorder traversal: []
                2.3.2 Traverse in left subtree:
                    2.3.2.1 See node = [] Inorder traversal:[]
                2.3.3 Visit node =[1]; Inorder traversal: [1]
                2.3.4 In right subtree:
                    2.3.4.1 See node = [] Inorder traversal:[1]
            2.3 Visit node = [3]; Inorder traversal: [1, 3]
            2.4 In right subtree:
                2.4.1. node = []; Inorder traversal: [1,3]
        3. Visit node = [7]; Inorder traversal: [1, 3, 7]
        4. In right subtree:
            4.1 See node = [9]; left = []; right =  [11, [], []]. Inorder traversal: [1, 3, 7]
            4.2 In left subtree:
                4.2.1 node = []; Inorder traversal: [1, 3, 7]
            4.3 Visit node = [9]; Inorder traversal: [1, 3, 7, 9]
            4.4 In right subtree:
                4.4.1. See node = [11]; left = []; right = []. Inorder traversal: [1, 3, 7, 9]
                4.4.2 Traverse in left subtree:
                    4.4.2.1 See node = [] Inorder traversal: [1, 3, 7, 9]
                4.4.3 Visit node = [11]; Inorder traversal: [1, 3, 7, 9, 11]
                4.4.4 Traverse in right subtree:
                    4.4.4.1 See node = [] Inorder traversal: [1, 3, 7, 9, 11]

        Answer Inorder traversal: [1, 3, 7, 9, 11]
        """
    },
    {
        "role": "assistant",
        "content": """
        Tree in the form of nested list: [13, [16, [], [12, [], []]], [11, [13, [], []], []]].
        Output its inorder traversal as a list of integers excluding empty branches and without printing `None` in the final output. Print your final answer as Inorder traversal:

        Let’s think step by step:
        1. See node = [13]; left = [16, [], [12, [], []]]; right = [11, [13, [], []], []]. Inorder traversal: []
        2. Traverse in left subtree:
            2.1 See node = [16]; left = []; right = [12, [], []]. Inorder traversal: []
            2.2 Traverse in left subtree:
                2.3.1. see node = []; Inorder traversal: []
            2.3 Visit node = [16]; Inorder traversal: [16]
            2.4 Traverse in right subtree:
                2.4.1. See node = [12]; Inorder traversal: [16]
                2.4.2  Traverse in left subtree:
                    2.4.2.1 see node = []; Inorder traversal: [16]
                2.4.3  Visit node = [12]; Inorder traversal: [16, 12]
                2.4.4  Traverse in right subtree:
                    2.4.4.1 see node = []; Inorder traversal: [16, 12]
        3. Visit node = [13]; Inorder traversal: [16, 12, 13]
        4. In right subtree:
            4.1 See node = [11]; left = [13, [], []]; right =  []. Inorder traversal: [16, 12, 13]
            4.2 In left subtree:
                4.2.1 See node = [13]; left=[]; right=[] Inorder traversal: [16, 12, 13]
                4.2.2  Traverse in left subtree:
                    4.2.2.1 see node = []; Inorder traversal: [16, 12, 13]
                4.2.3  Visit node = [13]; Inorder traversal: [16, 12, 13, 13]
                4.2.4  Traverse in right subtree:
                    4.2.4.1 see node = []; Inorder traversal: [16, 12, 13, 13]
            4.3 Visit node = [11]; Inorder traversal: [16, 12, 13, 13, 11]
            4.4 In right subtree:
                4.4.1. See node = [];  Inorder traversal: [16, 12, 13, 13, 11]


        Answer Inorder traversal: [16, 12, 13, 13, 11]
        """
    },
    {
        "role": "assistant",
        "content": """
        Tree in the form of nested list: [8, [1, [7, [], []], []], [8, [5, [], []], [10, [], []]]].
        Output its inorder traversal as a list of integers excluding empty branches and without printing `None` in the final output. Print your final answer as Inorder traversal:

        Let’s think step by step:
        1. See node = [8]; left = [1, [7, [], []], []]; right = [8, [5, [], []], [10, [], []]]]. Inorder traversal: []
        2. Traverse in left subtree:
            2.1 See node = [1]; left = [7, [], []]; right = []. Inorder traversal: []
            2.2 Traverse in left subtree:
                2.3.1. see node = [7]; left = []; right = []. Inorder traversal: []
                2.3.2 Traverse in left subtree:
                    2.3.2.1 See node = [] Inorder traversal:[]
                2.3.3 Visit node =[7]; Inorder traversal: [7]
                2.3.4 Traverse In right subtree:
                    2.3.4.1 See node = [] Inorder traversal:[7]
            2.3 Visit node = [1]; Inorder traversal: [7, 1]
            2.4 Traverse In right subtree:
                2.4.1. See node = []; Inorder traversal: [7, 1]
        3. Visit node = [8]; Inorder traversal: [7, 1, 8]
        4. In right subtree:
            4.1 See node = [8]; left = [5, [], []]; right =  [10, [], []]. Inorder traversal: [7, 1, 8]
            4.2 In left subtree:
                4.2.1 See node = [5]; left=[]; right=[] Inorder traversal: [7, 1, 8]
                4.2.2  Traverse in left subtree:
                    4.2.2.1 see node = []; Inorder traversal: [7, 1, 8]
                4.2.3  Visit node = [5]; Inorder traversal: [7, 1, 8, 5]
                4.2.4  Traverse in right subtree:
                    4.2.4.1 see node = []; Inorder traversal: [7, 1, 8, 5]
            4.3 Visit node = [8]; Inorder traversal: [7, 1, 8, 5, 8]
            4.4 In right subtree:
                4.4.1. See node = [10];  Inorder traversal: [7, 1, 8, 5, 8]
                4.4.2  Traverse in left subtree:
                    4.4.2.1 see node = []; Inorder traversal: [7, 1, 8, 5, 8]
                4.4.3  Visit node = [10]; Inorder traversal: [7, 1, 8, 5, 8, 10]
                4.4.4  Traverse in right subtree:
                    4.4.4.1 see node = []; Inorder traversal: [7, 1, 8, 5, 8, 10]

        Answer Inorder traversal: [7, 1, 8, 5, 8, 10]
    """
    },
    # — Your turn —
    {
        "role": "user",
        "content": "Tree in the form of nested list: {nested}. "
                   "Output its inorder traversal as a list of integers excluding empty branches and without printing `None` in the final output. Print your final answer as Inorder traversal:"
    }]
    ,
    "prompt_template": [
        {
            "role": "system",
            "content": "You are an expert at inorder traversal."
        },
        {
            "role": "user",
            "content": "Tree in the form of nested list: {nested}. "
                       "Output its inorder traversal as a  list of integers excluding empty branches and without printing `None` in the final output. "
                       "Think step by step and do reasoning behind and Print your final answer as Inorder traversal:"
        }
    ]
}


MODEL_CONFIGS = {
    "Qwen/Qwen2-Math-7B" : { "COT": PROMPT_TEMPLATE_COT_STYLE,
                             "NO_COT": PROMPT_TEMPLATE_NON_COT_STYLE
                            },
    "Qwen/Qwen2-7B-Instruct": { "COT": PROMPT_TEMPLATE_COT_STYLE_INSTRUCT,
                                 "NO_COT": PROMPT_TEMPLATE_NON_COT_STYLE_INSTRUCT
                                },
    "Qwen/Qwen2-Math-7B-Instruct": { "COT": PROMPT_TEMPLATE_COT_STYLE_INSTRUCT,
                                     "NO_COT": PROMPT_TEMPLATE_NON_COT_STYLE_INSTRUCT
                                    }
}
# ── OUTPUT EXTRACTOR ─────────────────────────────────────────────────────────
class InorderExtractor:
    """Extract the first Python‐style list of ints that appears after 'Inorder Traversal:'."""
    def __init__(self):
        self.key_pattern =  re.compile(
            r'Inorder\s+Traversal\s*:\s*`?\s*'             # “Inorder Traversal:”
            r'(?:'                                    # start non-capturing group
            r'\[\s*(?P<bracket>-?\d+(?:\s*,\s*-?\d+)*)\s*\]'  # [n, n, …]
            r'|'                                    # OR
            r'(?P<arrow>-?\d+(?:\s*->\s*-?\d+)*)'   # n -> n -> …
            r'|'
+           r'(?P<flat>-?\d+(?:\s*,\s*-?\d+)*)(?:\.)?'         # n, n, n…  (optional “.”)
            r')',
            re.IGNORECASE
        )
        self.key_pattern2 = re.compile(
            r'(?:Inorder\s+Traversal\s*:\s*|Expected\s+output\s+is\s*)'  # match "Inorder Traversal:" or "Expected output is"
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
            "total":        total,
            "key_matches":  self.matches,
            "none":         self.none,
        }

def test_Inorder_For_One_Sample():
    tree = TreeBuilder.generate_non_empty_tree(depth=3, max_val=20)
    nested = TreeBuilder.tree_to_nested_list(tree)
    ground_truth = []
    TreeBuilder.inorder_traversal(tree, ground_truth)
    # Show nested representation and ground truth
    print("Nested list representation of the tree:")
    print(nested)
    TreeBuilder.draw_tree(nested, save_path="Inorder/InorderTree.png")
    print("\nGround-truth inorder traversal:")
    print(ground_truth)
    # Prepare prompt
    prompt = f"""Given a binary tree represented as a nested list, output its inorder traversal as a Python list of integers.
    
    Tree: {nested}
    
    Inorder traversal:"""
    print("\nPrompt:")
    print(prompt)
    # Simulate model output (in a real benchmark, you'd call model.generate)
    simulated_output = str(ground_truth)
    print("\nSimulated model output:")
    print(simulated_output)
    # Extract using InorderExtractor
    extractor = InorderExtractor()
    pred = extractor.extract(simulated_output)
    print("\nExtractor prediction:")
    print(pred)
    # Verify correctness
    print("\nDoes prediction match ground truth?", pred == ground_truth)

# ── Benchmark loop ──────────────────────────────────────────────────────────
def benchmark_model(model_name, option, nested_list, groundTruth_list):

    print(f"\n=== Benchmarking {model_name} ===")
    option_fewshot, option_cot = option
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
    total   = NUM_SAMPLES
    extractor = InorderExtractor()

    config = MODEL_CONFIGS[model_name]


    MAX_NEW_TOKENS_ALl =  -sys.maxsize - 1
    prompt_print = None
    for nested, groundTruth in zip(nested_list, groundTruth_list):
        if option_fewshot:
            few_shot_prompt_template = copy.deepcopy(config[option_cot]["few_shot_prompt_template"])
            prompt = copy.deepcopy(few_shot_prompt_template.format(nested=nested))
        else:
            prompt_template = config[option_cot]["prompt_template"]
            prompt = prompt_template.format(nested=nested)
        prompt_ids = tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_ids["input_ids"].shape[1]
        inputs = prompt_ids.to(DEVICE)
        MAX_NEW_TOKENS_ = max(prompt_len*1.5, MAX_NEW_TOKENS)
        MAX_NEW_TOKENS_ALl = max(MAX_NEW_TOKENS_,MAX_NEW_TOKENS_ALl)
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
        response   = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        # print("---",response)
        pred   = extractor.extract_prediction(response)
        # print(pred)
        # print(groundTruth)
        if pred == groundTruth:
            correct += 1

    acc     = correct / total * 100

    # Console output
    # print(f"Accuracy: {correct}/{total} = {acc:.1f}%")
    stop_criteria = None
    # print("i printed prediction:", pred)
    # print("ground truth:", groundTruth)
    # Write only the summary to output.txt
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path   = os.path.join(script_dir, "output_inorder_11.txt")
    with open(log_path, "a") as f:
        f.write(f"\nUSE_FEWSHOT : {option_fewshot} and USE_COT : {option_cot} and stop_criteria : {stop_criteria} and max_tokens_length allowed : {MAX_NEW_TOKENS_ALl} Accuracy for {model_name}: {correct}/{total} = {acc:.1f}%\n")
        print("Extraction summary:", extractor.summary(), file=f)
        print("prompt_print",prompt_print, file=f)
        print("printed prediction:", pred, file=f)
        print("ground truth:", groundTruth, file=f)
        print("response:", response, file=f)


def benchmark_model_instruct(model_name, option, nested_list, groundTruth_list):
    print(f"\n=== Benchmarking {model_name} ===")
    option_fewshot, option_cot = option
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the model and move it to the chosen device
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    ).to("cuda")


    correct = 0
    total = NUM_SAMPLES
    extractor = InorderExtractor()

    config = MODEL_CONFIGS[model_name]
    MAX_NEW_TOKENS_ALl = -sys.maxsize - 1
    prompt_print = None
    groundTruth_ = None
    for nested, groundTruth in zip(nested_list, groundTruth_list):
        if option_fewshot:
            messages = copy.deepcopy(config[option_cot]["few_shot_prompt_template"])
        else:
            messages = copy.deepcopy(config[option_cot]["prompt_template"])

        for msg in messages:
            if msg["role"] == "user":
                msg["content"] = msg["content"].format(nested=nested)
                break

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
        groundTruth_ = groundTruth

    acc = correct / total * 100

    # Console output
    # print(f"Accuracy: {correct}/{total} = {acc:.1f}%")
    stop_criteria = None
    # print("printed prediction:", pred)
    # print("ground truth:", groundTruth)
    # Write only the summary to output.txt
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, "output_inorder_11.txt")
    with open(log_path, "a") as f:
        f.write(
            f"\nUSE_FEWSHOT : {option_fewshot}  and USE_COT : {option_cot} and stop_criteria : {stop_criteria} and max_tokens_length allowed : {MAX_NEW_TOKENS_ALl} Accuracy for {model_name}: {correct}/{total} = {acc:.1f}%\n")
        print("Extraction summary:", extractor.summary(), file=f)
        print("prompt_print",prompt_print, file=f)
        print("printed prediction:", pred, file=f)
        print("ground truth:", groundTruth_, file=f)
        print("response:", response, file=f)

if __name__ == "__main__":

    log_path = os.path.join(os.path.dirname(__file__), "output_inorder_11.txt")

    # truncate the file (make it empty)
    # open(log_path, "w").close()
    with open(log_path, "a") as f:
        print("\n-----------INORDER ---------:\n", file=f)

    nested_list, groundTruth_list = TreeBuilder.generateDataset("Inorder")
    # print("---------hello", nested_list, groundTruth_list)
    # for nested, groundTruth in zip(nested_list, groundTruth_list):
    #     print(nested,"------", groundTruth)
    USE_FEWSHOT = [ True, False]
    USE_COT = ["COT", "NO_COT"]
    for option in product(USE_FEWSHOT, USE_COT):
        # for model_name in INSTRUCT_MODEL_NAMES:
        #     benchmark_model_instruct(model_name,option, nested_list, groundTruth_list)
        for model_name in MODEL_NAMES:
            benchmark_model(model_name,option, nested_list, groundTruth_list)

