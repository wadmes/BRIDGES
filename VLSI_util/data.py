"""
The data utilization module for VLSI data processing.
"""

import circuitgraph as cg
from torch_geometric.data import HeteroData,Data
import torch
import torch.nn.functional as F
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
    data['node'].x = F.one_hot(torch.tensor(node_attribute),num_classes = len(supported_types)).float()
    data['node','pos','node'].edge_index = torch.tensor(edge_dict['pos'])
    data['node','inv','node'].edge_index = torch.tensor(edge_dict['inv'])
    return data


if __name__ == '__main__':
    bbs =[]
    circuit = cg.from_file('../netlist_data/arith/adder_4_bit.v',blackboxes=bbs)
    g = cg2hetedata(circuit)
    print(g)