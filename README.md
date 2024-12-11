# The graph-based multi-modal LLM for VLSI tasks


## Dependencies
```
# Install Rust 
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
conda create -n llm python=3.10
conda activate llm
pip install -r requirements.txt
```


## Reproduce: Stage 1

```
python stage1.py --gtm --lm --tune_gnn 
```
Stage 1 will also report the results for design retrieval

## Reproduce: Stage 2 -- type prediction

```
python stage2.py --tune_gnn --inference_batch_size 4 --batch_size 8 --llm_tune  --filename FILENAME
```

## Reproduce: Stage 2 -- function description

```
python stage2.py --tune_gnn --inference_batch_size 4 --batch_size 8 --llm_tune  --task func_desc
``````
