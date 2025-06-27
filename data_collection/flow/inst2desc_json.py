"""
data generation step 6
This script processes a JSON file containing design instructions, converts each instruction
into a descriptive tone using the LLAMA3 model, and updates the JSON file with the generated
descriptions.
"""
import json
import typer
import warnings
from tqdm import tqdm
from llama import Llama
from typing import Optional
warnings.filterwarnings("ignore", category=UserWarning)

def main(
    json_path: str = typer.Option(..., help="Path to the JSON file containing instructions and descriptions."),
    ckpt_dir: str = typer.Option(..., help="Path to the LLAMA3 checkpoint directory."),
    tokenizer_path: str = typer.Option(..., help="Path to the LLAMA3 tokenizer."),
    max_seq_len: int = typer.Option(8192, help="Maximum sequence length for the model."),
    max_batch_size: int = typer.Option(1, help="Maximum batch size for the model."),
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    
    with open(json_path) as f:
       json_content = json.load(f)
    
    error = []
    for i in tqdm(json_content.keys()):
        dialogs = [
            [
                {"role": "user", "content": """Given a design instruction, change it into a tone of description. Do not change or add any details.
                \n Here are two examples.
                \n Instruction: Design a module that can detect any edge in an 8-bit binary vector and output the binary value of the vector one cycle after the edge is detected. The module should have two input ports: a clock input and an 8-bit binary input port. The output port should be an 8-bit binary vector that represents the input value one cycle after the edge is detected. The module must be designed using a counter and a comparator.
                \n Example description: This module is designed to detect any edge in an 8-bit binary vector and output the binary value of the vector one cycle after the edge is detected. The module has two input ports: a clock input (`clk`) and an 8-bit binary input port (`in`). The output port (`out`) is an 8-bit binary vector that represents the input value one cycle after the edge is detected. The design uses a counter and a comparator to achieve this functionality.
                \n Instruction: Please act as a professional Verilog designer. Design a pipelined module that implements a 4-to-2 priority encoder. The module should have four 1-bit inputs (I0, I1, I2, I3) and two 2-bit outputs (O0, O1). The output should be the binary encoding of the highest-priority input that is asserted. If multiple inputs are asserted, the output should correspond to the input with the highest index number (i.e., the last asserted input in the list). Use pipeline structure to achieve this functionality.
                \n Example description: This design is a pipelined 4-to-2 priority encoder module. The module has four 1-bit inputs (`I0`, `I1`, `I2`, `I3`) and two 2-bit outputs (`O0`, `O1`). The output is the binary encoding of the highest-priority input that is asserted. If multiple inputs are asserted, the output corresponds to the input with the highest index number. The design uses a pipeline structure to implement this functionality.
                \n Now, please change this instruction directly (do not include any pre-fix like `here is a rewritten description): """ + json_content[i]['instruction']},
            ]
        ]
        try:
            results = generator.chat_completion(
                dialogs,
                temperature=0.1,
                top_p=0.9,
            )
            print(results[0]['generation']['content'])
            json_content[i]['description'] = results[0]['generation']['content']
        except:
            error.append(i)
    print("The list of indices of json items that fail:", error)
    with open(json_path, "w") as f:
        json.dump(json_content, f)

if __name__ == "__main__":
    typer.run(main)
