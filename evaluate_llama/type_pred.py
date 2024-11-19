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
    parser.add_argument("--use_data", type=str, default="rtl", help="data type")
    return parser.parse_args()


def main(args):
    model_name = args.model_name
    dataset_path = args.dataset_path

    # load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    orig_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a hardware description expert. Provide a single, coherent technical paragraph describing the functionality of a Verilog module.
    Constraints:
    - Use complete English sentences.
    - Avoid mentioning variable names or including any Verilog syntax.
    - Ensure the description focuses on functionality, not implementation details.
    - Do not use lists, bullet points, or code snippets.
    - Maintain a logical flow without line breaks or special formatting.
                
    Example:
    ---
    **Module Description:**
    This module implements an edge detection mechanism. It accepts an 8-bit binary input and a clock signal, 
    producing an 8-bit output that reflects the input value one cycle after an edge is detected. 
    The circuit operates by comparing the current input with the previous input to identify edges, utilizing a counter to manage the delay in output generation.
    ---
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Provide a detailed description of the following Verilog module. Its {} code is {}. <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
    # load dataset, the dataset is a list of graphs, it includes graph.rtl, graph.netlist, graph.text (function description, append to the prompt)
    graph_list = torch.load(dataset_path)
    # set model to eval mode
    model.eval()
    loss_list = []
    with torch.no_grad():
        for graph in tqdm.tqdm(graph_list):
            start = timeit.default_timer()
            if args.use_data == "rtl":
                data = graph.rtl[:args.context_window]
            else:
                data = graph.netlist[:args.context_window]
            # prompt is the same for all graphs
            prompt = orig_prompt.format(args.use_data, data)
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
            loss_list.append(float(outputs.loss.detach().cpu().item()))
            # print(f"Time to calculate loss: {timeit.default_timer() - start}")
            del inputs, outputs
            torch.cuda.empty_cache()
            # add loss_list[-1] to tqdm progress bar
            # tqdm.tqdm.write(f"Perplexity: {loss_list[-1]}")


    # calculate the average perplexity
    avg_loss = sum(loss_list) / len(loss_list)
    import math
    # write the result to a csv (append mode), the csv file will have columns: model_name, dataset_path, use_data, avg_loss, perplexity
    with open("results.csv", "a") as f:
        f.write(f"{model_name},{dataset_path},{args.use_data},{avg_loss},{math.exp(avg_loss)}\n")

main(get_args())