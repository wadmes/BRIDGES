"""
data generation step 1
iterator function to load MGVerilof RTL data from the Hugging Face Dataset object
"""
import re
from tqdm import tqdm
from datasets import load_from_disk

def iterator_MGVerilog(dataset_path):
    """Generator to iterate over the RTL objects from the Hugging Face Dataset object."""
    dataset = load_from_disk(str(dataset_path))
    for idx in tqdm(range(dataset.num_rows)):
        description = dataset['description'][idx]
        code = dataset['code'][idx]
        pattern = r"Assume that signals are positive clock/clk edge triggered unless otherwise stated\.\n\n(.*?)\n\n Module header:\n\n(.*?)\n \[/INST\]"
        match = re.search(pattern, description, re.DOTALL)
        description = match.group(1).strip() if match else None
        module_header = match.group(2).strip() if match else None
        if module_header:
            code = f"{module_header}\n{code}"
        yield (idx, description, code)
