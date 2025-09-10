
dataset_path_list=("/home/weili3/VLSI-LLM-Graph/VLSI_util/RTLCoder26532_design_info_test.pt")

# dataset_path_list=("./RTLCoder26532_val.pt")
use_data_list=("RTL")
for dataset_path in "${dataset_path_list[@]}"
do
    for use_data in "${use_data_list[@]}"
    do
        echo " dataset_path: $dataset_path, use_data: $use_data"
        python PPA_openai.py --dataset_path $dataset_path --use_data $use_data

    done
done