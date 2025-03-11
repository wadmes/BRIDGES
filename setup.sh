
srun --pty -w node-gpu01 --gres gpu:H100:1 /bin/bash
srun --time 3-23 --mem=400G --cpus-per-gpu=4 --pty --gres gpu:H100:1 --reservation=ConfNov19 /bin/bash 
srun --time 3-23 --mem=400G --cpus-per-gpu=4 --pty --gres gpu:H200:1 /bin/bash 
srun --mem=500G --cpus-per-gpu=4 --reservation=ConfNov19 --pty /bin/bash 
conda activate llm
module load cudnn8.9-cuda12.3  cuda12.3/toolkit cuda12.3/fft cuda12.3/blas
module load cudnn8.9-cuda12.1  cuda12.1/toolkit cuda12.1/fft cuda12.1/blas
module load cudnn8.9-cuda12.4  cuda12.4/toolkit cuda12.4/fft cuda12.4/blas
module load cudnn8.6-cuda11.8/8.6.0.163  cuda11.8/toolkit/11.8.0 cuda11.8/fft/11.8.0 cuda11.8/blas/11.8.0

python stage1.py --gtm --lm --tune_gnn

python stage1-textonly.py --gtm --lm --tune_gnn

python stage2.py --stage1_path "all_checkpoints/stage1_test/1113.ckpt" --tune_gnn --inference_batch_size 2 --batch_size 2

python app.py --devices 0 --init_checkpoint "all_checkpoints/stage2/epoch=08-v1.ckpt"

The graph of this module is [START_NETLIST_GRAPH]{}[END__NETLIST_GRAPH].

../VLSI-LLM/data_collection/netlist_data/verilog/2277_medium_low_low.v