
srun --pty -w node-gpu01 --gres gpu:H100:1 /bin/bash
srun --time 3-23 --mem=400G --cpus-per-gpu=4 --pty --gres gpu:H100:1 /bin/bash 
conda activate molca
module load cudnn8.9-cuda12.3  cuda12.3/toolkit cuda12.3/fft cuda12.3/blas
module load cudnn8.6-cuda11.8/8.6.0.163  cuda11.8/toolkit/11.8.0 cuda11.8/fft/11.8.0 cuda11.8/blas/11.8.0

python stage1.py --gtm --lm --batch_size 16 --match_batch_size 16 --tune_gnn

python stage2.py --devices '1' --filename "stage2" --stage1_path "all_checkpoints/stage1/epoch=499.ckpt" --opt_model 'facebook/galactica-1.3b' --max_epochs 500 --mode pretrain --tune_gnn --llm_tune freeze --inference_batch_size 4

python app.py --devices 0 --init_checkpoint "all_checkpoints/stage2/last-v2.ckpt"

The logic netlist graph tokens are {}. Please describe the logic netlist graph.