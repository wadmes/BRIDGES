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
"""
def cg2hetedata(circuit_graph):
    G = circuit_graph.graph
    # node attribute [#_nodes,#_gate_types]
    node_attribute = []
    node_index = 0
    edge_dict = {'pos':[[],[]],'inv':[[],[]]}
    # key mask, is 1 if the node is a key gate
    key_mask = []
    # store key value, for non-key node, the value is set as 0
    key_values  = []
    for node in circuit_graph:
        node_type = supported_types.index(G.nodes[node]['type'])
        node_attribute.append(node_type)
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
"""
def networkx2hetedata(G):
    # node attribute [#_nodes,#_gate_types]
    node_attribute = []
    node_index = 0
    edge_dict = {'pos':[[],[]],'inv':[[],[]]}
    # key mask, is 1 if the node is a key gate
    key_mask = []
    # store key value, for non-key node, the value is set as 0
    key_values  = []
    for node in G.nodes():
        node_type = supported_types.index(G.nodes[node]['type'])
        node_attribute.append(node_type)
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
class netlistDataset(InMemoryDataset):
    """
    path: the path to the netlist json file.
    rtl_path: the path to the rtl json file.
    type: graph type for the netlist, could be 'hetedata' or 'homodata'
    """
    def __init__(self, path, rtl_path, type):
        super(netlistDataset,self).__init__()
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
    



        

"""
The main function is to generate train dataset, eval, and test dataset.
"""
if __name__ == '__main__':
    bbs =[]
    test = netlistDataset('/home/weili3/VLSI-LLM/data_collection/netlist_data/netlist.json',"/home/weili3/VLSI-LLM/data_collection/rtl_data/RTL_with_desc.json", "hetedata")
    # save test to './netlist.pt'
    torch.save(test, './netlist.pt')