# The script to evaluate performance of LLAMA3 in function description tasks (it will report the perplexity of the model on the given dataset)

import argparse
import tqdm
from transformers import AutoImageProcessor, AutoTokenizer
from transformers import AutoModel, AutoModelForCausalLM, PreTrainedModel, BitsAndBytesConfig
import torch
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="model name")
    parser.add_argument("--dataset_path", type=str, required=True, help="dataset path")
    # context window, int
    parser.add_argument("--context_window", type=int, default=2048, help="context window size")
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
    prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a hardware description expert. Provide a single, coherent technical paragraph describing the functionality of a Verilog module.
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
    for graph in tqdm.tqdm(graph_list):
        if args.use_data == "rtl":
            data = graph.rtl
        else:
            data = graph.netlist
        # prompt is the same for all graphs
        prompt = prompt.format(args.use_data, data)
        # append function description to the prompt
        prompt += graph.text
        # calculate the perplexity of the model
        inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=args.context_window)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        loss_list.append(loss.cpu().item())
    # calculate the average perplexity
    avg_loss = sum(loss_list) / len(loss_list)
    import math
    print(f"Average Perplexity: {math.exp(avg_loss)}")

main(get_args())