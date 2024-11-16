
srun --time 3-23 --mem=400G --cpus-per-gpu=4 --pty -w node-gpu02 --gres gpu:H100:1 --reservation=ConfNov19 python stage1.py --gtm --lm --tune_gnn --filename train_0.05 --training_data_used 0.05 &
srun --time 3-23 --mem=400G --cpus-per-gpu=4 --pty -w node-gpu02 --gres gpu:H100:1 --reservation=ConfNov19 python stage1.py --gtm --lm --tune_gnn --filename train_0.01 --training_data_used 0.01 &
srun --time 3-23 --mem=400G --cpus-per-gpu=4 --pty -w node-gpu02 --gres gpu:H100:1 --reservation=ConfNov19 python stage1.py --gtm --lm --tune_gnn --filename train_0.1 --training_data_used 0.1 &
srun --time 3-23 --mem=400G --cpus-per-gpu=4 --pty -w node-gpu02 --gres gpu:H100:1 --reservation=ConfNov19 python stage1.py --gtm --lm --tune_gnn --filename train_0.2 --training_data_used 0.2 &
srun --time 3-23 --mem=400G --cpus-per-gpu=4 --pty -w node-gpu02 --gres gpu:H100:1 --reservation=ConfNov19 python stage1.py --gtm --lm --tune_gnn --filename train_0.5 --training_data_used 0.5 &
srun --time 3-23 --mem=400G --cpus-per-gpu=4 --pty -w node-gpu02 --gres gpu:H100:1 --reservation=ConfNov19 python stage1.py --gtm --lm --tune_gnn --filename train_0.8 --training_data_used 0.8 &