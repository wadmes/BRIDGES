# BRIDGES
A graph-based multi-modal LLM flow for EDA tasks.

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

## Dataset and Generation Flow
`data_collection/` contains the BRIDGES data generation flow.

BRIDGES data can be found [here](https://huggingface.co/datasets/WillZ0123/BRIDGES/tree/main).


## Citation

If you use BRIDGES, please cite the following paper:

```bibtex
@misc{li2025bridges,
  title        = {BRIDGES: Bridging Graph Modality and Large Language Models within EDA Tasks},
  author       = {Wei Li and Yang Zou and Christopher Ellis and Ruben Purdy and Shawn Blanton and José M. F. Moura},
  year         = {2025},
  eprint       = {2504.05180},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2504.05180},
  doi          = {10.48550/arXiv.2504.05180}
}
```