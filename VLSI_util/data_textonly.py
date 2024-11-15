"""
The data utilization module for VLSI data processing.
"""

import circuitgraph as cg
from torch_geometric.data import HeteroData,Data,InMemoryDataset
import torch
import torch_geometric.transforms as T
import os
import json
import pickle
import tqdm

class stage1dataset(InMemoryDataset):
    """
    used for Stage1DM_v2
    """
    def __init__(self, graphs):
        super(stage1dataset,self).__init__()
        self.graphs = graphs

    def len(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]
    


"""
In v2, we will not split dataset here (instead, we will split the dataset in the training script)
"""
def create_datasets_v2(netlist_path, rtl_path, seed = 42, train_ratio = 0.9, eval_ratio = 0.05, type ='hetedata'):
    rtl_file = json.load(open(rtl_path))
    syn_success_rtl_id_list = []
    # first report a simple statistics, how many `synthesis_status` is false, how many dataflow_status is false
    
    for key in rtl_file.keys():
        if rtl_file[key]['synthesis_status']:
            syn_success_rtl_id_list.append(int(key))
    netlist_file = json.load(open(netlist_path))
    netlist_file_dir_path = os.path.dirname(netlist_path)
    # for each .v file in path
    graphs = []
    error_files = []
    for key in tqdm.tqdm(netlist_file.keys()):
        if int(netlist_file[key]['rtl_id']) not in syn_success_rtl_id_list:
            continue
        func_desc = rtl_file[str(netlist_file[key]['rtl_id'])]['description']

        syn_efforts = netlist_file[key]['synthesis_efforts']
        generic_effort = syn_efforts.split('_')[0]
        mapping_effort = syn_efforts.split('_')[1]
        optimization_effort = syn_efforts.split('_')[2]
        func_desc += ' The synthesis efforts are: generic_effort - ' + generic_effort + ', mapping_effort - ' + mapping_effort + ', optimization_effort - ' + optimization_effort + '.'
        netlist_file_path = os.path.join(netlist_file_dir_path, str(netlist_file[key]['rtl_id']) + '_' + netlist_file[key]['synthesis_efforts'] + '.v')
        # load netlist_file_path as a string
        try:
            with open(netlist_file_path, 'r') as f:
                netlist_str = f.read()
        except:
            error_files.append(netlist_file_path)
            continue
        if type == 'hetedata':
            graphs.append(Data())
        else:
            raise NotImplementedError
        graphs[-1].text = func_desc
        graphs[-1].rtl_id = int(netlist_file[key]['rtl_id'])
        graphs[-1].netlist = netlist_str
        graphs[-1].rtl = rtl_file[str(netlist_file[key]['rtl_id'])]['verilog']
    print('error files: ', error_files)
    result = {}
    result['graphs'] = graphs
    result['rtl_id_list'] = syn_success_rtl_id_list # list of rtl_id in the dataset, used for splitting the dataset


    return result



"""
The main function is to generate train dataset, eval, and test dataset.
"""
if __name__ == '__main__':
    # train_set = torch.load("test_w_syn.pt")
    # train_set_size = int(len(train_set) * 0.8)
    # valid_set_size = len(train_set) - train_set_size
    
    # import torch.utils.data as data
    # seed = torch.Generator().manual_seed(42)
    # train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)
    # print(train_set[0], len(train_set))
    # print(valid_set[0], len(valid_set))
    # # Combine the two datasets
    # train_dataset = data.ConcatDataset([train_set, valid_set])
    # print(train_dataset[0])
    # print(len(train_dataset))
    # exit()
    import argparse
    bbs =[]
    parser = argparse.ArgumentParser()
    parser.add_argument('--netlist_path', type=str, default='/home/weili3/VLSI-LLM/data_collection/MGVerilog11144/netlist_data/netlist.json')
    parser.add_argument('--rtl_path', type=str, default='/home/weili3/VLSI-LLM/data_collection/MGVerilog11144/rtl_data/rtl.json')
    parser.add_argument('--name', type=str, default="MGVerilog11144-textonly")
    args = parser.parse_args()
    dataset = create_datasets_v2(args.netlist_path,args.rtl_path, "hetedata")
    torch.save(dataset, args.name + '.pt')