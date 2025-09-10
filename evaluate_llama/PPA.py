# The script to evaluate performance of LLAMA3 in function description tasks (it will report the perplexity of the model on the given dataset)

import argparse
import tqdm
from transformers import AutoImageProcessor, AutoTokenizer
from transformers import AutoModel, AutoModelForCausalLM, PreTrainedModel, BitsAndBytesConfig
import torch
import timeit
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="model name")
    parser.add_argument("--dataset_path", type=str, required=True, help="dataset path")
    # context window, int
    parser.add_argument("--context_window", type=int, default=512, help="context window size")
    # use_data, options are rtl and netlist
    parser.add_argument("--use_data", type=str, default="RTL", help="data type")
    return parser.parse_args()


def main(args):
    model_name = args.model_name
    dataset_path = args.dataset_path

    # load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    orig_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a hardware expert. Provide the area approximation of a Verilog module.
                
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
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Provide a area approximation of the following Verilog module. 
    **{} code**
    {}
    **synthesis effort:**
    {}.
      Please reply with only the area number. The area is:<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
    #     **power**
    # 5.18266e-05
    # load dataset, the dataset is a list of graphs, it includes graph.rtl, graph.netlist, graph.text (function description, append to the prompt)
    graph_list = torch.load(dataset_path,weights_only=False)
    # set model to eval mode
    model.eval()
    loss_list = []
    results = {"netlist_id": [],  "label": [], "prediction": []}
    with torch.no_grad():
        for graph in tqdm.tqdm(graph_list):
            start = timeit.default_timer()
            if args.use_data == "RTL":
                data = graph.rtl[:args.context_window]
            else:
                data = graph.netlist[:args.context_window]
            # prompt is the same for all graphs
            prompt = orig_prompt.format(args.use_data, data,graph.synthesis_efforts)
            prompt_len = len(tokenizer(prompt)["input_ids"])
            # append function description to the prompt
            prompt += graph.text
            # calculate the perplexity of the model
            inputs = tokenizer(prompt, return_tensors="pt", padding="longest", truncation=True)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            # only keep tokens after the prompt and not padding tokens as the target

            inputs["labels"] = inputs["input_ids"].clone()
            inputs["labels"][:, :prompt_len] = -100
            inputs["labels"][inputs["labels"] == tokenizer.pad_token_id] = -100
            # print(f"Time to prepare data: {timeit.default_timer() - start}")
        
            outputs = model(**inputs)
            results["netlist_id"].append(graph.netlist_id)
            results["area_label"].append(graph.area)
            results["area_prediction"].append(outputs.logits[0, prompt_len:, :].argmax(dim=-1).cpu().numpy())
            # print(f"Time to calculate loss: {timeit.default_timer() - start}")
            del inputs, outputs
            torch.cuda.empty_cache()
            # add loss_list[-1] to tqdm progress bar
            # tqdm.tqdm.write(f"Perplexity: {loss_list[-1]}")


    import math
    # write the result to a csv (append mode), the csv file will have columns: model_name, dataset_path, use_data, avg_loss, perplexity
    with open("area_results.csv", "a") as f:
        f.write(f"{model_name},{dataset_path},{args.use_data},{avg_loss},{math.exp(avg_loss)}\n")

main(get_args())