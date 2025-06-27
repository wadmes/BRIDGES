"""
data generation step 4
This script predicts the functional category of RTL (Register Transfer Level) designs and 
their Verilog code using two models: LLAMA3 and GPT-4o. It processes synthesis results, 
generates predictions for each design, and stores the results in labeled output files. 
The script includes error handling with retries for GPT predictions and supports instruction 
or description-based prompts.
"""
import typer
import time
import openai
import warnings
import pickle as pkl
from tqdm import tqdm
from pathlib import Path
from llama import Llama
warnings.filterwarnings("ignore", category=UserWarning)

def predict_function_category_Llama(prompt, verilog, prompt_type, system_setting, generator):
    dialogs = [
        [
            {"role": "system", "content": system_setting},
            {"role": "user", "content": f"""{prompt_type}: "{prompt}"\nVerilog: "{verilog}" """},
        ]
    ]
    try:
        results = generator.chat_completion(
            dialogs,
            temperature=0.0,
            top_p=0.0,
            logprobs=True,
        )
        reply = results[0]["generation"]["content"]
        tokens = results[0]["tokens"]
        logprob = results[0]["logprobs"]
        return reply, tokens, logprob
    except Exception:
        return None, None, None

def predict_function_category_GPT(prompt, verilog, prompt_type, system_setting, client):
    messages = [
        {"role": "system", "content": system_setting},
        {"role": "user", "content": f"""{prompt_type}: "{prompt}"\nVerilog: "{verilog}" """}
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.0,
        top_p=0.0,
        logprobs=True,
        top_logprobs=8,
    )
    reply = response.choices[0].message.content.strip()
    logprobs = response.choices[0].logprobs.content
    return reply, logprobs

def main(
    data_dir: Path = typer.Option(..., help="Base directory containing synthesis results and output folders."),
    prompt_type: bool = typer.Option(True, "--instruction/--description", help="Whether the RTL's related prompt is instruction (--instruction) or description (--description)."),
    ckpt_dir: Path = typer.Option(..., help="Path to the LLAMA3 checkpoint directory."),
    tokenizer_path: Path = typer.Option(..., help="Path to the LLAMA3 tokenizer."),
    max_seq_len: int = typer.Option(8192, help="Maximum sequence length for the model."),
    max_batch_size: int = typer.Option(1, help="Maximum batch size for the model."),
):
    """
    Main function to predict circuit unit types using LLAMA3.

    Args:
        data_dir (Path): Base directory containing synthesis results and output folders.
        prompt_type (bool): Flag indicating if the RTL's related information is an instruction or description.
        ckpt_dir (Path): Path to the LLAMA3 checkpoint directory.
        tokenizer_path (Path): Path to the LLAMA3 tokenizer.
        max_seq_len (int): Maximum sequence length for the model.
        max_batch_size (int): Maximum batch size for the model.
    """
    generator = Llama.build(
        ckpt_dir=str(ckpt_dir),
        tokenizer_path=str(tokenizer_path),
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    system_setting = rf"""
You are a professional VLSI digital design engineer. Categorize the following RTL (Register Transfer Level) design {prompt_type}s and Verilog code pairs into one of the functional categories below. The response should only contain the most relevant function category:

Functional Categories:
1. Encryption Units: Modules that handle encryption or cryptographic functions.
2. Data Path Units: Modules involved in data movement, selection, or manipulation (e.g., multiplexers, shifters).
3. Control Logic Units: Modules responsible for control flow or decision-making in systems (e.g., state machines).
4. Arithmetic Units: Modules performing arithmetic operations (e.g., adders, subtractors).
5. Communication Protocol Units: Modules implementing communication protocols (e.g., UART, SPI).
6. Signal Processing Units: Modules used for signal transformation or filtering.
7. Clock Management Units: Modules managing clock signals and synchronization.
8. Other Units: Modules not fitting the above categories.

Please reply with only the functional category name.

Examples:
1.
{prompt_type}: "This module is a 4-bit adder with carry-in and carry-out. The module has two 4-bit inputs, a single carry-in input, and a single carry-out output. The output is the sum of the two inputs plus the carry-in."
Verilog: "module adder (\n    input [3:0] a,\n    input [3:0] b,\n    input cin,\n    output cout,\n    output [3:0] sum\n);\n\n    assign {"{cout, sum}"} = a + b + cin;\n\nendmodule"
Response: "Arithmetic Units"
2.
{prompt_type}: "This module is a 2-to-1 multiplexer designed using Verilog. The module has two input ports and one output port. The output is the value of the first input port if the select input is 0, and the value of the second input port if the select input is 1. The design is implemented using only NAND gates."
Verilog: "module mux_2to1 (\n    input a,\n    input b,\n    input select,\n    output reg out\n);\n\n  wire nand1, nand2, nand3, nand4;\n\n  assign nand1 = ~(a & select);\n  assign nand2 = ~(b & ~select);\n  assign nand3 = ~(nand1 & nand2);\n  assign nand4 = ~(nand3 & nand3);\n\n  always @ (nand4) begin\n    out <= ~nand4;\n  end\n\nendmodule"
Response: "Data Path Units"

Now categorize the following RTL {prompt_type} and Verilog code pair:
"""

    output_dir = data_dir / "rtl_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "synthesis/synthesis_result.pkl", 'rb') as f:
        success, _, _ = pkl.load(f)

    prompt_type = "instruction" if prompt_type else "description"

    description_verilog_list = []
    for idx in success:
        with open(data_dir / f"synthesis/{idx}/{prompt_type}.txt", 'r') as f:
            prompt = f.read()
        with open(data_dir / f"synthesis/{idx}/rtl.v", 'r') as f:
            verilog = f.read()
        description_verilog_list.append((idx, prompt, verilog))

    results = []
    for idx, description, verilog in tqdm(description_verilog_list, desc="Llama3 is predicting"):
        reply, tokens, logprob = predict_function_category_Llama(description, verilog, prompt_type, system_setting, generator)
        results.append((idx, reply, tokens, logprob))
        with open(output_dir / "Llama3_70b_label.pkl", "wb") as f:
            pkl.dump(results, f)

    results = []
    client = openai.OpenAI()
    for idx, description, verilog in tqdm(description_verilog_list, desc="GPT4o is predicting"):
        retry_count = 0
        while retry_count <= 7:
            try:
                reply, logprobs = predict_function_category_GPT(description, verilog, prompt_type, system_setting, client)
                results.append((idx, reply, logprobs))
                break
            except Exception as e:
                print(f"Retry {retry_count + 1} due to error: {e}")
                time.sleep(10)
                retry_count += 1
        if retry_count > 7:
            print("error: GPT prediction retry too many times!")
        with open(output_dir / "GPT_4o_label.pkl", "wb") as f:
            pkl.dump(results, f)

if __name__ == "__main__":
    typer.run(main)
