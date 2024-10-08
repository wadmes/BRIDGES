
srun --pty -w node-gpu01 --gres gpu:H100:1 /bin/bash
srun --time 3-23 --pty --gres gpu:H100:4 /bin/bash
conda activate molca
module load cudnn8.9-cuda12.3  cuda12.3/toolkit cuda12.3/fft cuda12.3/blas
python stage1.py --gtm --lm --batch_size 16 --match_batch_size 16 --devices '1' --tune_gnn

python stage2.py --devices '1' --filename "stage2" --stage1_path "all_checkpoints/stage1/epoch=499.ckpt" --opt_model 'facebook/galactica-1.3b' --max_epochs 500 --mode pretrain --tune_gnn --llm_tune freeze --inference_batch_size 4

python app.py --devices 0 --init_checkpoint "all_checkpoints/stage2/last-v2.ckpt"

The logic netlist graph tokens are {}. Please describe the logic netlist graph.