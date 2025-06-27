"""
data generation step 1
iterator function to load RTLCoder RTL data from the json file
"""
import json

def iterator_RTLCoder(json_file_path):
    """Generator to iterate over the RTL objects from the JSON file."""
    with open(json_file_path) as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            data = json.loads(line)
            instruction = data.get("Instruction", "")
            code = data.get("Response", [""])[0]
            yield (idx, instruction, code)
