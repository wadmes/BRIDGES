# The graph-based multi-modal LLM for VLSI tasks


## Environment
```
CUDA Version: 12.4
PyTorch Version: 1.16.0
```
## Dependencies
```
# Install Rust 
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
conda create -n vlsillm python=3.10
conda activate vlsillm
pip install torch torchvision torchaudio
python -m pip install -r requirements.txt
```


## Data format
list of graph objects, each graph object includes
 - consistent_label