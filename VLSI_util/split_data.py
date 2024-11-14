"""
11-13-2024. 
I decide to write a script to split and save data (train, val, test) for the stage1 * stage 2 model.
"""

import torch
import argparse
import random

"""
Random split the graph list based on the given ratio in ratio_list.
ratio_list: list of float, sum(ratio_list) == 1
graph_list: list of graph
rtl_list: list of rtl id to be randomly split

First randomly split the rtl_list based on the ratio_list
Then split the graph_list based on the rtl_list using graph.rtl_id

Return:
new_graph_list: list of graph list, len(new_graph_list) == len(ratio_list)
"""
def random_split_by_rtl(ratio_list, rtl_list, graph_list, seed = 42):
    # Set the random seed for reproducibility
    random.seed(seed)
    
    # Shuffle the rtl_list
    random.shuffle(rtl_list)
    
    # Calculate the split points for the rtl_list based on ratio_list
    total_rtl = len(rtl_list)
    split_points = [int(r * total_rtl) for r in ratio_list]
    
    # Adjust split points to ensure the sum matches the length of rtl_list
    cumulative_splits = []
    cumulative_sum = 0
    for split_point in split_points[:-1]:  # Leave the last part to catch all remaining elements
        cumulative_sum += split_point
        cumulative_splits.append(cumulative_sum)
    split_rtl_lists = [rtl_list[start:end] for start, end in zip([0] + cumulative_splits, cumulative_splits + [None])]
    
    # Create a lookup for fast rtl_id to split index mapping
    rtl_to_split_index = {}
    for i, split in enumerate(split_rtl_lists):
        for rtl_id in split:
            rtl_to_split_index[rtl_id] = i
    
    # Partition graphs based on the rtl_id split
    new_graph_list = [[] for _ in ratio_list]
    for graph in graph_list:
        split_index = rtl_to_split_index.get(graph.rtl_id)
        if split_index is not None:
            new_graph_list[split_index].append(graph)
    
    return new_graph_list

parser = argparse.ArgumentParser()
# add graph_path, str, reqired = True
parser.add_argument('--graph_path', type=str, required=True, help='path to the graph file')
# seed, default is 42
parser.add_argument('--seed', type=int, default=42, help='random seed')
# add train_ratio, default is 0.9
parser.add_argument('--train_ratio', type=float, default=0.9, help='ratio of training data')

args = parser.parse_args()


# load the graph
ds = torch.load(args.graph_path)
ds  = random_split_by_rtl([args.train_ratio, (1-args.train_ratio)/2, (1-args.train_ratio)/2], ds['rtl_id_list'], ds['graphs'], args.seed)
# save the data to the same directory but rename as name_train.pt, name_val.pt, name_test.pt
torch.save(ds[0], args.graph_path.replace('.pt', '_train.pt'))
torch.save(ds[1], args.graph_path.replace('.pt', '_val.pt'))
torch.save(ds[2], args.graph_path.replace('.pt', '_test.pt'))
