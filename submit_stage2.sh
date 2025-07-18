
# srun --time 3-23 --mem=400G --cpus-per-gpu=4 --pty -w node-gpu02 --gres gpu:H100:1 --reservation=ConfNov19 python stage2.py --stage1_path "all_checkpoints/stage1_test/1113.ckpt" --tune_gnn --inference_batch_size 2 --batch_size 2 --filename 3B &
# srun --time 3-23 --mem=400G --cpus-per-gpu=4 --pty -w node-gpu02 --gres gpu:H100:1 --reservation=ConfNov19 python stage2.py --stage1_path "all_checkpoints/stage1_test/1113.ckpt" --tune_gnn --inference_batch_size 4 --batch_size 4 --filename 1B --opt_model meta-llama/Llama-3.2-1B-Instruct&
# srun --time 3-23 --mem=400G --cpus-per-gpu=4 --pty -w node-gpu02 --gres gpu:H100:1 --reservation=ConfNov19 python stage2.py --stage1_path "all_checkpoints/stage1_test/1113.ckpt" --tune_gnn --inference_batch_size 2 --batch_size 2 --filename 8B --opt_model meta-llama/Llama-3.1-8B-Instruct&
# srun --time 3-23 --mem=400G --cpus-per-gpu=4 --pty -w node-gpu02 --gres gpu:H100:1 --reservation=ConfNov19 python stage2.py --stage1_path "all_checkpoints/stage1_test/1113.ckpt" --tune_gnn --inference_batch_size 2 --batch_size 2 --filename 3B_lora --llm_tune &

# srun --time 3-23 --mem=400G --cpus-per-gpu=4 --pty -w node-gpu02 --gres gpu:H100:1 --reservation=ConfNov19 python stage2.py --tune_gnn --inference_batch_size 2 --batch_size 2 --filename no_stage1_3B &
# srun --time 3-23 --mem=400G --cpus-per-gpu=4 --pty -w node-gpu02 --gres gpu:H100:1 --reservation=ConfNov19 python stage2.py --tune_gnn --inference_batch_size 2 --batch_size 2 --filename no_stage1_3B_lora --llm_tune &
# srun --time 3-23 --mem=400G --cpus-per-gpu=4 --pty -w node-gpu02 --gres gpu:H100:1 --reservation=ConfNov19 python stage2.py --tune_gnn --inference_batch_size 2 --batch_size 2 --caption_eval_epoch 5 --task func_desc --filename func_desc &
# srun --time 3-23 --mem=400G --cpus-per-gpu=4 --pty -w node-gpu02 --gres gpu:H100:1 --reservation=ConfNov19 python stage2.py --tune_gnn --inference_batch_size 2 --batch_size 2 --caption_eval_epoch 5 --task func_desc --add_file rtl --filename func_desc_w_rtl &
# srun --time 3-23 --mem=400G --cpus-per-gpu=4 --pty -w node-gpu02 --gres gpu:H100:1 --reservation=ConfNov19 python stage2.py --tune_gnn --inference_batch_size 2 --batch_size 2 --caption_eval_epoch 5 --task func_desc --add_file rtl --llm_tune --filename func_desc_lora_w_rtl &
# srun --time 3-23 --mem=400G --cpus-per-gpu=4 --pty -w node-gpu02 --gres gpu:H100:1 --reservation=ConfNov19 python stage2.py --tune_gnn --inference_batch_size 2 --batch_size 2 --caption_eval_epoch 5 --task func_desc --llm_tune --filename func_desc_lora&
# srun --time 3-23 --mem=400G --cpus-per-gpu=4 --pty -w node-gpu02 --gres gpu:H100:2 --reservation=ConfNov19 python stage2.py --tune_gnn --inference_batch_size 7 --batch_size 7 --llm_tune --add_file rtl --filename 3B-lora_w_rtl &
# srun --time 3-23 --mem=400G --cpus-per-gpu=4 --pty -w node-gpu02 --gres gpu:H100:6 python stage2.py --tune_gnn --inference_batch_size 4 --batch_size 8 --llm_tune --add_file netlist --filename 3B-lora_w_netlist &
# srun --time 3-23 --mem=400G --cpus-per-gpu=4 --pty -w node-gpu02 --gres gpu:H100:1 --reservation=ConfNov19 python stage2.py --tune_gnn --inference_batch_size 6 --batch_size 6 --llm_tune --opt_model meta-llama/Llama-3.1-8B-Instruct --filename 8B-lora &
# srun --time 3-23 --mem=400G --cpus-per-gpu=8 --pty -w node-gpu02 --gres gpu:H100:3 --reservation=ConfNov19 python stage2.py --tune_gnn --inference_batch_size 8 --batch_size 8 --llm_tune --opt_model meta-llama/Llama-3.1-8B-Instruct --add_file rtl --filename 8B-lora_rtl &
# srun --time 3-23 --mem=400G --cpus-per-gpu=4 --pty -w node-gpu02 --gres gpu:H100:1 --reservation=ConfNov19 python stage2.py --use_graph 0 --inference_batch_size 8 --batch_size 8 --caption_eval_epoch 5 --task func_desc --add_file rtl --llm_tune  --filename func_desc_lora_w_rtl_no_graph 

srun --time 1-23 --mem=400G --cpus-per-gpu=4 --pty -p HGPU --gres=gpu:H200:2 /bin/bash
# srun --time 3-23 --mem=400G --cpus-per-gpu=4 --pty -p HGPU --gres=gpu:H200:${1:-1} /bin/bash
# -p HGPU
# no q-former - type prediction
srun --time 3-23 --mem=400G --cpus-per-gpu=4 --pty --gres gpu:H100:1 python stage2.py --tune_gnn --inference_batch_size 2 --batch_size 4 --llm_tune --filename 3B-lora_no_qformer --use_qformer 0 --num_query_token 4 &
# no q-former - func desc
srun --time 3-23 --mem=400G --cpus-per-gpu=4 --pty --gres gpu:H100:2 python stage2.py --tune_gnn --inference_batch_size 8 --batch_size 8 --llm_tune --filename func_desc_no_qformer --task func_desc --add_file rtl --use_qformer 0 --num_query_token 4 &