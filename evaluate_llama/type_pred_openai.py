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
system_setting = """You are a specialized Verilog code analyzer focused on classifying hardware designs into specific categories. 
    Your task is to analyze Verilog code and determine its primary design type from the following categories:

Encryption Unit: Designs implementing cryptographic algorithms, secure hash functions, or other security-related operations
Data Path Unit: Components handling data flow, multiplexers, decoders, registers, and data routing
Control Logic Unit: State machines, sequence controllers, and decision-making logic
Arithmetic Unit: Mathematical operations, ALUs, multipliers, dividers, and computational blocks
Communication Protocol Unit: Implementations of protocols like UART, I2C, SPI, or other communication interfaces
Signal Processing Unit: Filters, FFT implementations, signal conditioning, and digital signal processing
Clock Management Unit: Clock generators, PLL implementations, clock dividers, and timing control
Others: Designs that don't clearly fit into the above categories"""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="dataset path")
    # context window, int
    parser.add_argument("--context_window", type=int, default=2000, help="context window size")
    # use_data, options are rtl and netlist
    parser.add_argument("--use_data", type=str, default="rtl", help="data type")
    return parser.parse_args()


def main(args):
    dataset_path = args.dataset_path
    enc = tiktoken.get_encoding("o200k_base")
    candidates = ['Encryption Unit', 'Data Path Unit', 'Control Logic Unit', 'Arithmetic Unit', 'Communication Protocol Unit', 'Signal Processing Unit', 'Clock Management Unit', 'Others']
    candidate_tokens_list = []
    for candidate in candidates:
        candidate_tokens = enc.encode(candidate)
        candidate_tokens_list.append(candidate_tokens)

    orig_prompt = """


    """

    # load dataset, the dataset is a list of graphs, it includes graph.rtl, graph.netlist, graph.text (function description, append to the prompt)
    graph_list = torch.load(dataset_path)
    # set model to eval mode
    loss_list = []
    correct = 0
    total = 0
    total_correct = 0
    results = {"netlist_id": [], "correct": [], "label": [], "prediction": []}
    tested_rtl = {}
    with torch.no_grad():
        for graph in tqdm.tqdm(graph_list):
            correct = 0
            if len(graph.consistent_label) == 0:
                continue
            else:
                graph.consistent_label = graph.consistent_label.replace('Units', 'Unit')
            if graph.rtl_id in tested_rtl.keys():
                continue
            total += 1
            start = timeit.default_timer()
            if args.use_data == "rtl":
                data = graph.rtl[:args.context_window]
            else:
                data = graph.netlist[:args.context_window]
            messages = [
            {"role": "system", "content": system_setting},
            {"role": "user", "content": f"""Please analyze the following code and classify it into one of the specified design types. Its {args.use_data} code is {data}. Please reply with only the category name. The design type is: """}
            ]
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
            # Extract log probabilities
            # logprobs = response.choices[0].logprobs.content
            # print(logprobs)

            # # Calculate cumulative log probabilities for each candidate
            # candidate_scores = {}
            # for i, candidate in enumerate(candidates):
            #     candidate_tokens = candidate_tokens_list[i]
            #     candidate_logprob = sum(
            #         logprobs["token_logprobs"][i]
            #         for i, token in enumerate(logprobs["tokens"])
            #         if token in candidate_tokens
            #     )
            #     candidate_scores[candidate] = candidate_logprob

            # # Find the candidate with the highest probability
            # best_candidate = max(candidate_scores, key=candidate_scores.get)
            if reply in graph.consistent_label or graph.consistent_label in reply:
                correct = 1
                total_correct += 1
            
            print(f"Predicted: {reply}, True: {graph.consistent_label}, Correct: {correct}, Total correct: {total_correct}/{total}")
            results["netlist_id"].append(graph.netlist_id)
            results["correct"].append(correct)
            results["label"].append(graph.consistent_label)
            results["prediction"].append(reply)
            tested_rtl[graph.rtl_id] = 1
            
    

    print(f"Accuracy: {total_correct/total}")
    # save the results as csv

    dataset_name = args.dataset_path.split("/")[-1].split(".")[0]
    # save as csv
    with open(f"{dataset_name}_{args.use_data}.csv", "w") as f:
        f.write("netlist_id,correct,label,prediction\n")
        for i in range(len(results["netlist_id"])):
            f.write(f"{results['netlist_id'][i]},{results['correct'][i]},{results['label'][i]},{results['prediction'][i]}\n")

main(get_args())