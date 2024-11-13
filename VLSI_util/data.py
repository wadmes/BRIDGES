"""
The data utilization module for VLSI data processing.
"""

import circuitgraph as cg
from torch_geometric.data import HeteroData,Data,InMemoryDataset
import torch
import torch_geometric.transforms as T
import os
addable_types = [
    "buf",
    "and",
    "or",
    "xor",
    "not",
    "nand",
    "nor",
    "xnor",
    "fflop",
    "0",
    "1",
    "x",
    "output",
    "input",
]

supported_types = addable_types + ["bb_input", "bb_output","key","virtual_key"]

"""
Convert a circuitgraph to a hetedata object in pytorch geometric.
The node attributes has two dimension:
1. the type of the gate (0 - 13)
2. whether the gate is a output (0 or 1)
"""
def cg2hetedata(circuit_graph):
    G = circuit_graph.graph
    node_attribute = []
    edge_dict = {'pos':[[],[]],'inv':[[],[]]}
    for node in circuit_graph:
        node_attribute.append([G.nodes[node]['type'], G.nodes[node]['output']])
        G.nodes[node]['index'] = len(node_attribute) - 1
    for edge in circuit_graph.edges():
        edge_dict['pos'][0].append(G.nodes[edge[0]]['index'])
        edge_dict['pos'][1].append(G.nodes[edge[1]]['index'])
        edge_dict['inv'][0].append(G.nodes[edge[1]]['index'])
        edge_dict['inv'][1].append(G.nodes[edge[0]]['index'])
    data = HeteroData()
    data['node'].x = torch.tensor(node_attribute).long()
    data['node','pos','node'].edge_index = torch.tensor(edge_dict['pos'])
    data['node','inv','node'].edge_index = torch.tensor(edge_dict['inv'])
    return data


"""
Convert a circuitgraph to a hetedata object in pytorch geometric.
The node attributes has two dimension:
1. the type of the gate (0 - 13)
2. whether the gate is a output (0 or 1)
"""
def networkx2hetedata(G):
    # node attribute [#_nodes,#_gate_types]
    node_attribute = []
    edge_dict = {'pos':[[],[]],'inv':[[],[]]}
    for node in G.nodes():
        node_attribute.append([G.nodes[node]['type'], G.nodes[node]['output']])
        G.nodes[node]['index'] = len(node_attribute) - 1

    for edge in G.edges():
        edge_dict['pos'][0].append(G.nodes[edge[0]]['index'])
        edge_dict['pos'][1].append(G.nodes[edge[1]]['index'])
        edge_dict['inv'][0].append(G.nodes[edge[1]]['index'])
        edge_dict['inv'][1].append(G.nodes[edge[0]]['index'])
    data = HeteroData()
    data['node'].x = torch.tensor(node_attribute).long()
    data['node','pos','node'].edge_index = torch.tensor(edge_dict['pos']).long()
    data['node','inv','node'].edge_index = torch.tensor(edge_dict['inv']).long()
    return data


def cg2homodata(circuit_graph):
    G = circuit_graph.graph
    # node attribute [#_nodes,#_gate_types]
    node_attribute = []
    node_index = 0
    edge_dict = [[],[]]
    for node in circuit_graph:
        node_type = supported_types.index(G.nodes[node]['type'])
        node_attribute.append(node_type)
        G.nodes[node]['index'] = len(node_attribute) - 1

    for edge in circuit_graph.edges():
        edge_dict[0].append(G.nodes[edge[0]]['index'])
        edge_dict[1].append(G.nodes[edge[1]]['index'])
    data = Data()
    data.edge_index = torch.tensor(edge_dict) # (E, 2)
    data.x = torch.tensor(node_attribute).long() # (N, k), k is the dimension of the node features
    data = T.ToUndirected()(data)
    return data


import json
import pickle
import tqdm
class netlistDataset_old(InMemoryDataset):
    """
    path: the path to the netlist json file.
    rtl_path: the path to the rtl json file.
    type: graph type for the netlist, could be 'hetedata' or 'homodata'
    """
    def __init__(self, path, rtl_path, type):
        super(netlistDataset_old,self).__init__()
        print('Loading netlist dataset, path: {}, type: {}'.format(path,type))
        netlist_file = json.load(open(path))
        netlist_file_dir_path = os.path.dirname(path)
        rtl_file = json.load(open(rtl_path))


        # for each .v file in path
        self.graphs = []
        self.type = type
        bbs = []
        for key in tqdm.tqdm(netlist_file.keys()):
            graph_path = os.path.join(netlist_file_dir_path,'graph', str(netlist_file[key]['rtl_id'])+'_'+netlist_file[key]['synthesis_efforts']+'.pkl')
            # verilog_path = os.path.join(netlist_file_dir_path,'netlist', str(netlist_file[key]['rtl_id'])+'_'+netlist_file[key]['synthesis_efforts']+'.v')
            func_desc = rtl_file[str(netlist_file[key]['rtl_id'])]['rtl_description']
            # circuit = cg.from_file(os.path.join(path,verilog_path),blackboxes=bbs)
            networkx_graph = pickle.load(open(graph_path,'rb'))
            if type == 'hetedata':
                self.graphs.append(networkx2hetedata(networkx_graph))
            # elif type == 'homodata':
            #     self.graphs.append(cg2homodata(circuit))
            else:
                raise NotImplementedError
            self.graphs[-1].text = func_desc
    def len(self):
        return len(self.graphs)

    def __getitem__(self, index):
        self.graphs[index]['node','pos','node'].edge_index = self.graphs[index]['node','pos','node'].edge_index.long()
        self.graphs[index]['node','inv','node'].edge_index = self.graphs[index]['node','inv','node'].edge_index.long()
        return self.graphs[index], self.graphs[index].text
    


class netlistDataset(InMemoryDataset):
    """
    path: the path to the netlist json file.
    rtl_list: the list of rtl_id
    rtl_path: the path to the rtl json file.
    type: graph type for the netlist, could be 'hetedata' or 'homodata'
    add_syn: whether add synthesis efforts in function description
    """
    def __init__(self, path, rtl_list, rtl_path, type, add_syn = True):
        super(netlistDataset,self).__init__()
        print('Loading netlist dataset, path: {}, type: {}'.format(path,type))
        netlist_file = json.load(open(path))
        netlist_file_dir_path = os.path.dirname(path)
        rtl_file = json.load(open(rtl_path))
        # for each .v file in path
        self.graphs = []
        self.type = type
        bbs = []
        error_files = []
        for key in tqdm.tqdm(netlist_file.keys()):
            if int(netlist_file[key]['rtl_id']) not in rtl_list:
                continue
            graph_path = os.path.join(netlist_file_dir_path,'graph', str(netlist_file[key]['rtl_id'])+'_'+netlist_file[key]['synthesis_efforts']+'.pkl')
            func_desc = rtl_file[str(netlist_file[key]['rtl_id'])]['description']
            try:
                networkx_graph = pickle.load(open(graph_path,'rb'))
            except:
                error_files.append(graph_path)
                continue
            if type == 'hetedata':
                self.graphs.append(networkx2hetedata(networkx_graph))
            else:
                raise NotImplementedError
            if add_syn:
                syn_efforts = netlist_file[key]['synthesis_efforts']
                generic_effort = syn_efforts.split('_')[0]
                mapping_effort = syn_efforts.split('_')[1]
                optimization_effort = syn_efforts.split('_')[2]
                func_desc += ' The synthesis efforts are: generic_effort - ' + generic_effort + ', mapping_effort - ' + mapping_effort + ', optimization_effort - ' + optimization_effort + '.'
            self.graphs[-1].text = func_desc
            self.graphs[-1].rtl_id = int(netlist_file[key]['rtl_id'])
        print('error files: ', error_files)

    def len(self):
        return len(self.graphs)

    def __getitem__(self, index):
        self.graphs[index]['node','pos','node'].edge_index = self.graphs[index]['node','pos','node'].edge_index.long()
        self.graphs[index]['node','inv','node'].edge_index = self.graphs[index]['node','inv','node'].edge_index.long()
        return self.graphs[index], self.graphs[index].text
    

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
        self.graphs[index]['node','pos','node'].edge_index = self.graphs[index]['node','pos','node'].edge_index.long()
        self.graphs[index]['node','inv','node'].edge_index = self.graphs[index]['node','inv','node'].edge_index.long()
        return self.graphs[index], self.graphs[index].text
    


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
        graph_path = os.path.join(netlist_file_dir_path,'graph', str(netlist_file[key]['rtl_id'])+'_'+netlist_file[key]['synthesis_efforts']+'.pkl')
        func_desc = rtl_file[str(netlist_file[key]['rtl_id'])]['description']
        try:
            networkx_graph = pickle.load(open(graph_path,'rb'))
        except:
            error_files.append(graph_path)
            continue
        if type == 'hetedata':
            graphs.append(networkx2hetedata(networkx_graph))
        else:
            raise NotImplementedError
        syn_efforts = netlist_file[key]['synthesis_efforts']
        generic_effort = syn_efforts.split('_')[0]
        mapping_effort = syn_efforts.split('_')[1]
        optimization_effort = syn_efforts.split('_')[2]
        func_desc += ' The synthesis efforts are: generic_effort - ' + generic_effort + ', mapping_effort - ' + mapping_effort + ', optimization_effort - ' + optimization_effort + '.'
        graphs[-1].text = func_desc
        graphs[-1].rtl_id = int(netlist_file[key]['rtl_id'])

    print('error files: ', error_files)
    result = {}
    result['graphs'] = graphs
    result['rtl_id_list'] = syn_success_rtl_id_list # list of rtl_id in the dataset, used for splitting the dataset

    return result




def create_datasets(netlist_path, rtl_path, seed = 42, train_ratio = 0.9, eval_ratio = 0.05, type ='hetedata'):
    rtl_file = json.load(open(rtl_path))
    rtl_id_list = list(rtl_file.keys()) # list of rtl_id, which is the key of the rtl_file
    syn_success_rtl_id_list = []
    # first report a simple statistics, how many `synthesis_status` is false, how many dataflow_status is false
    import numpy as np
    stat_matrix = np.zeros((2,2)) # synthesis_status, dataflow_status
    for key in rtl_file.keys():
        stat_matrix[int(rtl_file[key]['synthesis_status'])][int(rtl_file[key]['dataflow_status'])] += 1
        if rtl_file[key]['synthesis_status']:
            syn_success_rtl_id_list.append(int(key))
    print('synthesis_status (first dim), dataflow_status (second dim)')
    print(stat_matrix)
    print("success synthesis: ", len(syn_success_rtl_id_list))
    import random
    rtl_id_list = syn_success_rtl_id_list
    random.seed(seed)
    random.shuffle(rtl_id_list)
    train_size = int(len(rtl_id_list) * train_ratio)
    eval_size = int(len(rtl_id_list) * eval_ratio)
    test_size = int(len(rtl_id_list) - train_size - eval_size)
    train_list = rtl_id_list[:train_size]
    eval_list = rtl_id_list[train_size:train_size+eval_size]
    test_list = rtl_id_list[train_size+eval_size:]
    print("load netlist dataset: ", netlist_path)
    print('train size: {}, eval size: {}, test size: {}'.format(train_size,eval_size,test_size))
    train_dataset = netlistDataset(netlist_path,train_list, rtl_path, type)
    eval_dataset = netlistDataset(netlist_path,eval_list, rtl_path, type)
    test_dataset = netlistDataset(netlist_path,test_list, rtl_path, type)
    return train_dataset, eval_dataset, test_dataset


"""
Create train/eval/test dataset
"""
def create_datasets_old(netlist_path, rtl_path, seed = 42, train_ratio = 0.8, eval_ratio = 0.1, type ='hetedata'):
    rtl_file = json.load(open(rtl_path))
    rtl_id_list = list(rtl_file.keys()) # list of rtl_id, which is the key of the rtl_file
    syn_success_rtl_id_list = []
    # first report a simple statistics, how many `synthesis_status` is false, how many dataflow_status is false
    import numpy as np
    stat_matrix = np.zeros((2,2)) # synthesis_status, dataflow_status
    for key in rtl_file.keys():
        stat_matrix[int(rtl_file[key]['synthesis_status'])][int(rtl_file[key]['dataflow_status'])] += 1
        if rtl_file[key]['synthesis_status']:
            syn_success_rtl_id_list.append(int(key))
    print('synthesis_status (first dim), dataflow_status (second dim)')
    print(stat_matrix)
    print("success synthesis: ", len(syn_success_rtl_id_list))
    import random
    rtl_id_list = syn_success_rtl_id_list
    random.seed(seed)
    random.shuffle(rtl_id_list)
    train_size = int(len(rtl_id_list) * train_ratio)
    eval_size = int(len(rtl_id_list) * eval_ratio)
    test_size = int(len(rtl_id_list) - train_size - eval_size)
    train_list = rtl_id_list[:train_size]
    eval_list = rtl_id_list[train_size:train_size+eval_size]
    test_list = rtl_id_list[train_size+eval_size:]
    print("load netlist dataset: ", netlist_path)
    print('train size: {}, eval size: {}, test size: {}'.format(train_size,eval_size,test_size))
    train_dataset = netlistDataset(netlist_path,train_list, rtl_path, type)
    eval_dataset = netlistDataset(netlist_path,eval_list, rtl_path, type)
    test_dataset = netlistDataset(netlist_path,test_list, rtl_path, type)
    return train_dataset, eval_dataset, test_dataset

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
    parser.add_argument('--name', type=str, default="MGVerilog11144")
    args = parser.parse_args()
    dataset = create_datasets_v2(args.netlist_path,args.rtl_path, "hetedata")
    torch.save(dataset, args.name + '.pt')