"""
The data utilization module for VLSI data processing.
"""

import circuitgraph as cg
from torch_geometric.data import HeteroData,Data,InMemoryDataset
import torch
import torch.nn.functional as F
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
    data.edge_index = torch.tensor(edge_dict)
    data.x = torch.tensor(node_attribute).long()
    data = T.ToUndirected()(data)
    return data


class netlistDataset(InMemoryDataset):
    """
    path: the path to the netlist file, which should only include .v logic netlist files
    type: graph type for the netlist, could be 'hetedata' or 'homodata'
    """
    def __init__(self, path, type):
        super(netlistDataset,self).__init__()
        print('Loading netlist dataset, path: {}, type: {}'.format(path,type))
        # for each .v file in path
        self.graphs = []
        self.type = type
        for file in os.listdir(path):
            if file.endswith('.v'):
                for i in range(50): # repeat each netlist for 50 times, should be removed in the future
                    # file is {module type}_{bit number}_bit.v
                    module_type = file.split('_')[0]
                    bit_number = file.split('_')[1]
                    bbs = []
                    circuit = cg.from_file(os.path.join(path,file),blackboxes=bbs)
                    if type == 'hetedata':
                        self.graphs.append(cg2hetedata(circuit))
                    elif type == 'homodata':
                        self.graphs.append(cg2homodata(circuit))
                    else:
                        raise NotImplementedError
                    description = "The logic netlist is a {}-bit {} module.".format(bit_number,module_type)
                    self.graphs[-1].text = description
    def len(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index], self.graphs[index].text
    



        

"""
The main function is to generate train dataset, eval, and test dataset.
"""
if __name__ == '__main__':
    bbs =[]
    small = netlistDataset('../netlist_data/arith', "hetedata")
    # repeat small for 50 times
    print(small[0])
    # save small to ../netlist_data/arith/small.pt
    torch.save(small, '../netlist_data/arith/hete_small.pt')
    small = netlistDataset('../netlist_data/arith', "homodata")
    # repeat small for 50 times
    print(small[0])
    # save small to ../netlist_data/arith/small.pt
    torch.save(small, '../netlist_data/arith/homo_small.pt')