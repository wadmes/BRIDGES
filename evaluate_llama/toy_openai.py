# The script to evaluate performance of LLAMA3 in function description tasks (it will report the perplexity of the model on the given dataset)

import argparse
import tqdm
from transformers import AutoImageProcessor, AutoTokenizer
from transformers import AutoModel, AutoModelForCausalLM, PreTrainedModel, BitsAndBytesConfig
import torch
import timeit
import json
import openai
import pickle as pkl
import time
import tiktoken
client = openai.OpenAI(
  api_key="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"
)

test_file = {}
# for .v file in /home/weili3/VLSI-LLM-Graph/netlist_data/arith, if the file name includes 8 or 16, load them as a string, and store into test_file
import os
for root, dirs, files in os.walk("/home/weili3/VLSI-LLM-Graph/netlist_data/arith"):
    for file in files:
        if "8" in file or "16" in file:
            # only test 16 and multiplier
            if "8" in file and "multiplier" in file:

                with open(os.path.join(root, file), 'r') as f:
                    test_file[file] = f.read()

# print(test_file)

for file in test_file.keys():
    messages = [
    {"role": "system", "content": """
    Below verilog file is either an adder, comparator, divider, or multiplier, are you able to classify which type it is and what bit it is? Type your answer directly, for example, multiplier-8bit,no analysis.
    """},
    {"role": "user", "content": f"""test_file: {test_file[file]}"""},]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=4,  # Adjust as needed
        n=1,  # Single response
        temperature=0,
        top_p=0,
        logprobs=True,
        top_logprobs=1,
        )
    reply = response.choices[0].message.content.strip()
    print(response)
    print(reply)
    print("target:", file)
