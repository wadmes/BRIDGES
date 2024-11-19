
#model_name_list in meta-llama/Llama-3.2-1B-Instruct, meta-llama/Llama-3.2-3B-Instruct, meta-llama/Llama-3.1-8B-Instruct, meta-llama/Llama-3.1-70B-Instruct
model_name_list=("meta-llama/Llama-3.1-70B-Instruct")
dataset_path_list=("/scratch/weili3/RTLCoder26532_val.pt" "/scratch/weili3/MGVerilog11144_val.pt")
use_data_list=("netlist" "rtl")
for model_name in "${model_name_list[@]}"
do
    for use_data in "${use_data_list[@]}"
    do
        for dataset_path in "${dataset_path_list[@]}"
        do
            echo "model_name: $model_name, dataset_path: $dataset_path, use_data: $use_data"
            python func_desc.py --model_name $model_name --dataset_path $dataset_path --use_data $use_data
        done
    done
done