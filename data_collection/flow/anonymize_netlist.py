"""
data generation step 10
This script anonymizes Verilog netlist files by renaming module definitions and their instantiations. 
It processes all Verilog files (*.v) in a specified input folder and saves anonymized versions to an output folder. 
"""
import re
import typer
from tqdm import tqdm
from pathlib import Path

def anonymize_netlist(original_verilog_folder, anonymized_verilog_folder):
    """
    Anonymize Verilog netlist files by renaming module names and their instantiations.

    Args:
        original_verilog_folder (str or Path): Path to the folder containing the original Verilog files.
        anonymized_verilog_folder (str or Path): Path to the folder where anonymized Verilog files will be saved.
    """
    def anonymize_modules(verilog_code):
        module_mapping = {}
        counter = 0
        pattern = r'(\bmodule\s+)(\w+)(\s*\()'

        def replace_module_name(match):
            nonlocal counter
            original_module_name = match.group(2)
            anonymized_module_name = f"anonymized_module_{counter}"
            module_mapping[original_module_name] = anonymized_module_name
            counter += 1
            return f"{match.group(1)}{anonymized_module_name}{match.group(3)}"
        
        anonymized_code = re.sub(pattern, replace_module_name, verilog_code, flags=re.MULTILINE)

        for original_name, anonymized_name in module_mapping.items():
            instantiation_pattern = rf'\b{original_name}\b'
            anonymized_code = re.sub(instantiation_pattern, anonymized_name, anonymized_code)

        return anonymized_code, module_mapping

    verilog_files = [f for f in original_verilog_folder.iterdir() if f.suffix == '.v']
    
    for filename in tqdm(verilog_files, desc="Anonymizing Verilog Files"):
        with open(original_verilog_folder / filename, 'r') as f:
            content = f.read()
        code, _ = anonymize_modules(content)
        with open(anonymized_verilog_folder / filename, 'w') as f:
            f.write(code)

def main(
    original_verilog_folder: Path = typer.Option(..., help="Path to the folder containing the original Verilog files (to be anonymized)."),
    anonymized_verilog_folder: Path = typer.Option(..., help="Path to the folder where anonymized Verilog files will be saved.")
):
    if not original_verilog_folder.exists():
        print("The original path does not exist.")
        return
    anonymized_verilog_folder.mkdir(parents=True, exist_ok=True)
    anonymize_netlist(original_verilog_folder, anonymized_verilog_folder)

if __name__ == "__main__":
    typer.run(main)