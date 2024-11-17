# python data_textonly.py --netlist_path /home/weili3/VLSI-LLM/data_collection/RTLCoder26532/netlist_data/netlist.json --rtl_path /home/weili3/VLSI-LLM/data_collection/RTLCoder26532/rtl_data/rtl.json --name RTLCoder26532-textonly
# python data.py
# python data.py --netlist_path /home/weili3/VLSI-LLM/data_collection/RTLCoder26532/netlist_data/netlist.json --rtl_path /home/weili3/VLSI-LLM/data_collection/RTLCoder26532/rtl_data/rtl.json --name RTLCoder26532
cp ./RTLCoder26532.pt /scratch/weili3/
cp ./MGVerilog11144.pt /scratch/weili3/
python split_data.py --graph_path /scratch/weili3/RTLCoder26532.pt
python split_data.py --graph_path /scratch/weili3/MGVerilog11144.pt

# python split_data.py --graph_path /home/weili3/VLSI-LLM-Graph/VLSI_util/RTLCoder26532.pt
# python split_data.py --graph_path /home/weili3/VLSI-LLM-Graph/VLSI_util/MGVerilog11144.pt
