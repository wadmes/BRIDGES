
dataset_path_list=("./RTLCoder26532_val.pt" "./MGVerilog11144_val.pt")

# dataset_path_list=("./RTLCoder26532_val.pt")
use_data_list=("netlist")
for dataset_path in "${dataset_path_list[@]}"
do
    for use_data in "${use_data_list[@]}"
    do
        echo " dataset_path: $dataset_path, use_data: $use_data"
        python type_pred_openai.py --dataset_path $dataset_path --use_data $use_data

    done
done