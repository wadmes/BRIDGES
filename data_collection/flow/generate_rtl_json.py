"""
data generation step 5
This script processes Verilog RTL files, anonymizes module names, and compiles a JSON file containing detailed metadata 
about each RTL. It integrates synthesis results, optional PyVerilog analysis results, and optional type predictions to 
provide a comprehensive dataset for further use.
"""

import re
import json
import typer
import shutil
import pickle as pkl
from tqdm import tqdm
from pathlib import Path

def anonymize_modules(verilog_code):
    """
    Anonymize module names in Verilog code to prevent disclosing module information to language models.

    Args:
        verilog_code (str): The original Verilog code as a string.

    Returns:
        tuple: A tuple containing the anonymized Verilog code and a mapping of original to anonymized module names.
    """
    module_mapping = {}
    counter = 0
    pattern = r'(^module\s+)(\w+)(?:\s*#\s*\(.*?\))?\s*(\(|;)'

    def replace_module_name(match):
        nonlocal counter
        original_module_name = match.group(2)
        anonymized_module_name = f"anonymized_module_{counter}"
        module_mapping[anonymized_module_name] = original_module_name
        counter += 1
        return f"{match.group(1)}{anonymized_module_name}{match.group(3)}"

    anonymized_code = re.sub(pattern, replace_module_name, verilog_code, flags=re.MULTILINE | re.DOTALL)

    for original_name, anonymized_name in module_mapping.items():
        instantiation_pattern = rf'\b{original_name}\b'
        anonymized_code = re.sub(instantiation_pattern, anonymized_name, anonymized_code)

    return anonymized_code, module_mapping

def main(
    data_dir: Path = typer.Option(..., help="Base directory where all relevent outputs are stored."),
    prompt_type: bool = typer.Option(True, "--instruction/--description", help="Whether the RTL's related prompt is instruction (--instruction) or description (--description).")
):
    """
    Main function to process Verilog files, anonymize them, and compile results into a JSON file.

    Args:
        data_dir (Path): Base directory containing synthesis results and output folders.
        prompt_type (bool): Flag indicating if the RTL's related information is an instruction or description.
    """
    with open(data_dir / "synthesis/synthesis_result.pkl", 'rb') as f:
        success, timeout, fail = pkl.load(f)

    output_file = data_dir / "rtl_data/rtl.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    pyverilog_result_file = data_dir / "pyverilog/pyverilog_analysis.pkl"
    if pyverilog_result_file.exists():
        with open(pyverilog_result_file, 'rb') as f:
            _, _, dataflow_success, _ = pkl.load(f)
        graph_folder = data_dir / "rtl_data/dataflow_graph/"
        graph_folder.mkdir(parents=True, exist_ok=True)
        for idx, module_name in dataflow_success:
            shutil.copy(data_dir / f"pyverilog/{idx}/{module_name}.pkl", graph_folder / f"{idx}_{module_name}.pkl")
    
    llama_prediction = data_dir / "rtl_data/Llama3_70b_label.pkl"
    gpt_prediction = data_dir / "rtl_data/GPT_4o_label.pkl"
    type_prediction = llama_prediction.exists() and gpt_prediction.exists()
    if type_prediction:
        with open(llama_prediction, 'rb') as f:
            results = pkl.load(f)
        llama_label = {}
        for idx, label, _, _ in results:
            llama_label[idx] = label
        with open(gpt_prediction, 'rb') as f:
            results = pkl.load(f)
        gpt_label = {}
        for idx, label, _ in results:
            gpt_label[idx] = label

    is_first_entry = True
    with open(output_file, mode='w') as file:
        file.write("{\n")
    for rtl_id in tqdm(range(len(success) + len(timeout) + len(fail))):
        rtl_path = data_dir / f"synthesis/{rtl_id}"

        with open(rtl_path / "rtl.sv", 'r') as file:
            verilog = file.read()

        anonymized_verilog, module_mapping = anonymize_modules(verilog)

        if prompt_type:
            with open(f"{rtl_path}/instruction.txt", 'r') as file:
                prompt = file.read()
        else:
            with open(f"{rtl_path}/description.txt", 'r') as file:
                prompt = file.read()

        record = {
            'verilog': verilog,
            'anonymized_verilog': anonymized_verilog,
            'mapping': module_mapping,
            'instruction': prompt if prompt_type else '',
            'description': prompt if not prompt_type else '',
            'synthesis_status': rtl_id in success,
        }
        
        if pyverilog_result_file.exists():
            module_keys = list(module_mapping.keys())
            record['dataflow_status'] = bool(module_keys) and all((rtl_id, module_name) in dataflow_success for module_name in module_keys)
        if type_prediction:
            record['consistent_label'] = gpt_label[rtl_id] if (rtl_id in success) and (gpt_label[rtl_id] == llama_label[rtl_id]) else ''
            record['GPT_4o_label'] = gpt_label[rtl_id] if (rtl_id in success) else ''
            record['Llama3_70b_label'] = llama_label[rtl_id] if (rtl_id in success) else ''

        with open(output_file, mode='a') as file:
            if not is_first_entry:
                file.write(",\n")
            file.write(f'"{rtl_id}": ')
            json.dump(record, file, indent=4)
            is_first_entry = False

    with open(output_file, mode='a') as file:
        file.write("\n}")

if __name__ == "__main__":
    typer.run(main)
