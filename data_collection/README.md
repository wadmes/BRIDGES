# Data generation and process for VLSI-LLM
This directory contains the code and guide documents that automates the data collection process.

## Data Generation

### 1. Loading Dataset

Define a custom iterator function to yield tuples of index, instruction/description and Verilog code from the dataset. `iterator_MGVerilog.py` and `iterator_RTLCoder.py` are two iterator functions used to produce datasets in our paper.

### 2. Synthesis

> **Prerequisite**:
> Ensure you have access to Cadence tools (specifically **Genus**) and set up the required environment variables before running the synthesis script. Add the following to your terminal session or shell config file:
>
> ```bash
> export CADENCE=/path/to/cadence/installation
> export LM_LICENSE_FILE=$LM_LICENSE_FILE:$CADENCE/path/to/license.dat
> export CDS_LIC_FILE=port@your-license-server.domain
> # 22.10 the Genus version used in paper
> export DDI=$CADENCE/ddi-22.10
> export PATH="$DDI/bin:$PATH"
> ```
>
> This configures the environment for using the Genus synthesis tool with proper licensing.

Synthesize RTL designs in the dataset. `iterator_file` is the custom iterator for loading the intended dataset `dataset_path`. Create a base directory `data_dir` to store all synthesis results and other flow outputs. `synthesis_lib` is the custom standard cell library compatible with the Genus synthesis tool. `prompt_type` is the text paired with each RTL design in the original dataset (instruction and description are supported in this flow). A synthesis summary -- a tuple `(success, timeout, failure)` where each element is a list of indices -- will be saved at `data_dir/synthesis/synthesis_result.pkl`.
```
python flow/synthesize.py \
    --iterator-file flow/iterator_RTLCoder.py \
    --dataset-path RTLCoder26532/Resyn27k.json \
    --data-dir RTLCoder26532/ \
    --synthesis-lib flow/standard_cells.lib \
    --synthesis-timelimit 1800 \
    --num-processes 8 \
    --instruction
```

### 3. PyVerilog Analysis (Optional)
Use PyHDI's PyVerilog to analyze all RTL designs. Then, generate a NetworkX graph for the dataflow of each module in an RTL design, developed based on PyVerilog. `parser` is used to parse the Verilog and to detect syntax errors. `dataflow_analyzer` is used to analyze the dataflow in the Verilog. `dataflow_graph_generator` (the one used in our paper can be found `flow/networkx_dataflow_graphgen.py`) is used to generate the module-wise dataflow graph in NetworkX format. An analysis summary -- a tuple `(syntax_success, syntax_fail, dataflow_success, dataflow_fail)` where `syntax_success` and `syntax_fail` are lists of indices, `dataflow_success` and `dataflow_fail` are lists of `(index, module_name)` pairs -- will be saved at `data_dir/pyverilog/pyverilog_analysis.pkl`. (The `parser` and `dataflow_analyzer` we used are PyVerilog's original examples.)

```
python flow/pyverilog.py \
    --num-data 26532 \
    --synthesis-dir RTLCoder26532/synthesis/ \
    --data-dir RTLCoder26532/ \
    --parser /Pyverilog/examples/example_parser.py \
    --dataflow-analyzer /Pyverilog/examples/example_dataflow_analyzer.py \
    --dataflow-graph-generator /Pyverilog/examples/networkx_dataflow_graphgen.py \
    --analysis-time-limit 30
```

### 4. Circuit Unit Type Identification (Optional)
Utilize local Llama3 and GPT API to predict circuit types and check prediction consistency. OpenAI key should be set beforehand using `export OPENAI_API_KEY="your_api_key_here"`. Llama3 should be set locally according to [https://github.com/meta-llama/llama3/]. The prediction results will be save at `data_dir/rtl_data/Llama3_70b_label.pkl` and `data_dir/rtl_data/GPT_4o_label.pkl`.
```
torchrun --nproc_per_node 2 flow/predict_circuit_type.py \
    --data_dir RTLCoder26532/ \
    --instruction
    --ckpt_dir /llama3/Meta-Llama-3-70B-Instruct-2-shards/ \
    --tokenizer_path /llama3/Meta-Llama-3-70B-Instruct/tokenizer.model \
    --max_seq_len 8192 \
    --max_batch_size 1
```

### 5. RTL JSON Generation
Use all relevant outputs up to this point to generate a JSON containing all information at `data_dir/rtl_data/rtl.json`. Synthesis is the only required step for JSON generation. Collects all dataflow graphs into `data_dir/rtl_data/dataflow_graph/` if PyVerilog analysis has been done before.

```
python flow/generate_rtl_json.py --data-dir RTLCoder26532/ --instruction
```

### 6. RTL Instruction to Description (Optional)
For some datasets that come with RTL instructions, this script can be run to generate descriptions by using local Llama3. This should only be run on the generated RTL JSON file in step 5. `"description"` and `"instruction"` fields are initialized in the JSON file while being left empty if there is no information yet.
```
torchrun --nproc_per_node 2 flow/inst2desc_json.py \
    --json_path RTLCoder26532/rtl_data/rtl.json \
    --ckpt_dir /llama3/Meta-Llama-3-70B-Instruct-2-shards/ \
    --tokenizer_path /llama3/Meta-Llama-3-70B-Instruct/tokenizer.model \
    --max_seq_len 8192 \
    --max_batch_size 1 
```

### 7. RTL CSV Generation (Optinal)
Create a CSV containing all metadata. Step 1 to 5 are required before generating this CSV.
```
python flow/generate_rtl_csv.py --data-dir RTLCoder26532/
```

### 8. Netlist Graph Generation
Use CircuitGraph to parse and transform netlist verilog generated through synthesis to NetworkX graphs. The graphs will be save at `data_dir/netlist_data/dataflow_graph/`. `netlist_graphgen_fail.pkl` is a generation fail summary -- a list of tuples `(index, effort)` (the netlist generated with effort `effort` for `index` fails to be transformed into graph) -- will be saved at `data_dir/netlist_data/netlist_graphgen_fail.pkl`.
```
python flow/generate_netlist_graph.py -num-cores 8 --data-dir RTLCoder26532/
```

### 9. Netlist JSON Generation
Use the netlist graph generation results to generate a JSON at `data_dir/netlist_data/netlist.json` to keep track of all netlists. Genus synthesis logs will be collected and saved at `data_dir/netlist_data/synthesis_log/`. Netlist Verilog files will be collected and saved at `data_dir/netlist_data/verilog/`.
```
python flow/generate_netlist_graph.py --data-dir RTLCoder26532/
```

### 10. Anonymize Netlist Verilog (Optinal)
Anonymize the module names of all Verilog files in one folder and save at another new folder.
```
python flow/anonymize_netlist.py \
    --original-verilog-folder RTLCoder26532/netlist_data/verilog/ \
    --anonymized-verilog-folder RTLCoder26532/netlist_data/anonymized_verilog/
```

### 11. Netlist CSV Generation (Optinal)
Create a CSV containing all metadata. Step 1,2 and 8,9 are required before generating this CSV.
```
python flow/generate_netlist_csv.py --data-dir RTLCoder26532/
```

## Data Format
The processed data includes these files:

### `data_dir/rtl_data/rtl.json`
- Content: A dictionary that maps (string) `rtl_id` (should be an integer but json requires strings as keys) to RTL data (dictionary).
- Each RTL data dictionary has the following mapping:
    1. `"verilog"`: (string) the Verilog code for the RTL design.
    2. `"anonymized_verilog"`: (string) the Verilog code for the RTL design whose module names have been masked.
    3. `"mapping"`: a dictionary that maps (string) `anonymized_module_name` to (string) `original_module_name`. 
    4. `"instruction"`: (string) the instruction used to generate the RTL design.
    5. `"description"`: (string) the description used to generate the RTL design. (If the original dataset uses instruction to generate the RTL design, this will be generated by Llama3 if necessary.)
    6. `"synthesis_status"`: (bool) whether this design synthesized successfully.
    7. `"dataflow_status"`: (bool) whether all modules in this design produced dataflow successfully.
    8. `"GPT_4o_label"`: (bool) the circuit type prediction generated by GPT-4o for this RTL design.
    8. `"Llama3_70b_label"`: (bool) the circuit type prediction generated by Llama3-70b for this RTL design.
    8. `"consistent_label"`: (bool) the circuit type prediction if GPT-40 and Llama3-70b's predictions are consistent.

### `data_dir/rtl_data/dataflow_graph/`
- Content: a folder containing python pickle files of the Verilog dataflow graph as a `NetworkX` object.
- The dataflow graph of a module `module_name` of a RTL design `rtl_id` is in `{rtl_id}_{module_name}.pkl`.

### `data_dir/rtl_data/rtl.csv`
- Content: a file containing all rtl related metadata.
- Headers: `"rtl_id", "module_number", "module_name_list", "dataflow_status", "synthesis_status", "GPT_4o_label", "Llama3_70b_label", "consistent_label", "verilog_file_length", "#dataflow_node", "#dataflow_edge", "dataflow_node_type_distribution", "dataflow_node_in_degree_distribution", "dataflow_node_out_degree_distribution"`.

### `data_dir/netlist_data/netlist.json`
- Content: A dictionary that maps (string) `netlist_id` to netlist data (dictionary).
- Each netlist data dictionary has the following mapping:
    1. `"rtl_id"`: (integer) the corresponding RTL design used to generate this netlist.
    2. `"synthesis_efforts"`: (string) the efforts used in genus to synthesize the RTL design (`"{generic_effort}_{mapping_effort}_{optimization_effort}"`).
    3. `"graphgen_status"`: (bool) whether this netlist successfully generates a graph.

### `data_dir/netlist_data/graph/`
- Content: a folder containing python pickle files of the netlist graph as a `NetworkX` object.
- The graph of netlist synthesized from RTL design `rtl_id` with effort parameters `synthesis_efforts` is in `{rtl_id}_{synthesis_efforts}.pkl`.

### `data_dir/netlist_data/verilog/`
- Content: a folder containing Verilog files of the synthesized netlists.
- The netlist Verilog synthesized from RTL design `rtl_id` with effort parameters `synthesis_efforts` is in `{rtl_id}_{synthesis_efforts}.v`.

### `data_dir/netlist_data/netlist.csv`
- Content: a file containing all netlist related metadata.
- Headers: `"id", "rtl_id", "generic_effort", "mapping_effort", "optimization_effort", "graphgen_status", "#input", "#output", "#node", "#edge", "indegree_distribution", "outdegree_distribution", "#not_node", "#nand_node", "#nor_node", "#xor_node", "#xnor_node", "#input_node", "#0_node", "#1_node", "#x_node", "#buf_node", "#and_node", "#or_node", "#bb_input_node", "#bb_output_node", "verilog_file_length"`.