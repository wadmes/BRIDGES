
srun --pty -w node-gpu01 --gres gpu:H100:2 /bin/bash
conda activate molca
module load cudnn8.6-cuda11.8/8.6.0.163  cuda11.8/toolkit/11.8.0 cuda11.8/fft/11.8.0 cuda11.8/blas/11.8.0