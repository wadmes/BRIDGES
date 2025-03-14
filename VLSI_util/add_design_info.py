"""
The script adds design information to each netlist and rtl graph. e.g. area, and power.
"""

import argparse
import pickle 
import torch
# argment, str graph_path, default is `/home/weili3/VLSI-LLM-Graph/VLSI_util/RTLCoder26532_train.pt`
# argment, str design_info_path, default is `/home/weili3/VLSI-LLM-Graph/VLSI_util/design_info.csv`


parser = argparse.ArgumentParser()
parser.add_argument("--graph_path", type=str, default="/home/weili3/VLSI-LLM-Graph/VLSI_util/RTLCoder26532_train.pt", help="graph path")
parser.add_argument("--design_info_path", type=str, default="/home/weili3/VLSI-LLM-Graph/VLSI_util/design_info.pkl", help="design info path")
args = parser.parse_args()

# load design info
with open(args.design_info_path, "rb") as f:
    """
    design_info
    key (rtl index) - {synthesis_command (str): (area (float), power (float))} 
    """
    design_info = pickle.load(f)

# load graph
graphs = torch.load(args.graph_path,weights_only=False)
# for each graph, add the area and power information
new_graphs = []
for graph in graphs:
    # get the rtl index
    rtl_index = graph.rtl_id
    # get the design info
    if rtl_index in design_info:
        if design_info[rtl_index][0] > 0:
            graph.area = design_info[rtl_index][0]
            graph.power = design_info[rtl_index][1]
            new_graphs.append(graph)

# save the new_graphs
# new path is the same as the old path, but with `_design_info` added
new_path = args.graph_path.replace(".pt", "_design_info.pt")
print("original graph length: ", len(graphs))
print("new graph length: ", len(new_graphs))
torch.save(new_graphs, new_path)

print(design_info)
