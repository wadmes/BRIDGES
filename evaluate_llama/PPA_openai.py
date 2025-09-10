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
import re
client = openai.OpenAI(
  api_key="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"
)
system_setting = """You are a hardware expert. Provide the area approximation of a Verilog module.
                
    Example:
    ---
    **RTL code:**
    module anonymized_module_0(
        Y ,
        A1,
        A2,
        B1,
        B2
    );

        output Y ;
        input  A1;
        input  A2;
        input  B1;
        input  B2;

        // Voltage supply signals
        supply1 VPWR;
        supply0 VGND;
        supply1 VPB ;
        supply0 VNB ;

        assign Y = ((A1 & A2 & B1 & B2) | (A1 & A2 & !B1 & !B2) | (A1 & !A2 & B1 & !B2) | (!A1 & A2 & !B1 & B2));

    endmodule
    **synthesis effort, which includes 1. (generic effort}: Balances quality and runtime by controlling overall synthesis intensity,
    2. mapping effort: Influences library mapping, affecting timing, area, and power, and 3. optimization effort: Controls additional post-mapping QoR (timing or power) optimizations.):**
    high_low_medium
    **area:**
    10.5
    ---
    """

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


    # load dataset, the dataset is a list of graphs, it includes graph.rtl, graph.netlist, graph.text (function description, append to the prompt)
    graph_list = torch.load(dataset_path,weights_only=False)
    # set model to eval mode
    mape_list = []
    correct = 0
    total = 0
    total_correct = 0
    results = {"netlist_id": [], "label": [], "prediction": []}
    tested_rtl = {}
    count = 0
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
            {"role": "user", "content": f"""
                 Provide a area approximation of the following Verilog module. 
    **{args.use_data} code**
    {data}
    **synthesis effort:**
    {graph.synthesis_efforts}.
      Please reply with only the area number. The area is:
             """}
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
            
            print(f"Predicted: {reply}, True: {graph.area}")
            results["netlist_id"].append(graph.netlist_id)
            results["label"].append(graph.area)
            results["prediction"].append(reply)
            tested_rtl[graph.rtl_id] = 1
            # if reply (str) is a number, calculate mape
            p = '[\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+'

            if re.search(p, reply) is not None:
                reply_num = re.search(p, reply).group()
                print(reply_num)
                mape = abs((float(reply_num) - float(graph.area)) / float(graph.area))
                mape_list.append(mape)
            else:
                print(f"Error: {reply} is not a number")
                mape_list.append(10)
                # exit()

            
    
    # save the results as csv

    dataset_name = args.dataset_path.split("/")[-1].split(".")[0]
    # save as csv
    with open(f"{dataset_name}_{args.use_data}_area.csv", "w") as f:
        f.write("netlist_id,label,prediction\n")
        for i in range(len(results["netlist_id"])):
            f.write(f"{results['netlist_id'][i]},{results['label'][i]},{results['prediction'][i]}\n")

    # print three values, 1. avg_mape, 2. % that mape <0.01 3. % that mape <0.1 4. % that mape <0.5
    avg_mape = sum(mape_list) / len(mape_list)
    print(f"avg_mape: {avg_mape}")
    print(f"% that mape <0.01: {len([x for x in mape_list if x < 0.01]) / len(mape_list)}")
    print(f"% that mape <0.1: {len([x for x in mape_list if x < 0.1]) / len(mape_list)}")
    print(f"% that mape <0.5: {len([x for x in mape_list if x < 0.5]) / len(mape_list)}")

main(get_args())