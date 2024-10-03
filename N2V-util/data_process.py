"""
Read the data in ../N2V-data/ and process it for training.
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

# add ../ to sys.path
import sys
sys.path.append('../')
from VLSI_util.data import cg2hetedata, cg2homodata
import tqdm
class N2VDataset(InMemoryDataset):
    """
    path: the path to the N2V data
    type: graph type for the netlist, could be 'hetedata' or 'homodata'
    text: text info to store, could be 'desc' or 'rtl'
    """
    def __init__(self, path, type, text):
        super(N2VDataset,self).__init__()
        print('Loading (netlist to verilog) dataset, path: {}, type: {}, text info is {}'.format(path,type, text))
        # for each .v file in path
        self.graphs = []
        self.type = type
        self.text = text
        bbs = []
        success_verilog = 0
        fail_verilog = 0
        netlist_num = 0
        # for each directory in path
        for directory in tqdm.tqdm(os.listdir(path)):
            # read rtl.sv as rtl text
            try:
                with open(os.path.join(path,directory,'rtl.sv'),'r') as f:
                    rtl = str(f.read())
            except FileNotFoundError:
                rtl = None

            # read instruction.txt as desc text
            try:
                with open(os.path.join(path,directory,'instruction.txt'),'r') as f:
                    desc = str(f.read())
            except FileNotFoundError:
                desc = None
            
            # read syn.v in path/syn/* directory
            try:
                for sub_directory in os.listdir(os.path.join(path,directory,'syn')):
                    circuit = cg.from_file(os.path.join(path,directory,'syn',sub_directory,'syn.v'),blackboxes=bbs)
                    if type == 'hetedata':
                        self.graphs.append(cg2hetedata(circuit))
                    elif type == 'homodata':
                        self.graphs.append(cg2homodata(circuit))
                    else:
                        raise NotImplementedError
                    self.graphs[-1].rtl = rtl 
                    self.graphs[-1].desc = desc
                    netlist_num += 1
                success_verilog += 1
            except FileNotFoundError:
                fail_verilog += 1
        print('Data Read. Success RTL: {}, Fail RTL: {}, Total netlist: {}'.format(success_verilog,fail_verilog,netlist_num))

    def len(self):
        return len(self.graphs)
    
    def set_text(self, text):
        self.text = text

    def __getitem__(self, index):
        if self.text == 'desc':
            return self.graphs[index], self.graphs[index].desc
        elif self.text == 'rtl':
            return self.graphs[index], self.graphs[index].rtl


"""
The main function is to generate train dataset, eval, and test dataset.
"""
if __name__ == '__main__':
    bbs =[]
    hete = N2VDataset('../N2V-data', "hetedata", 'rtl')
    # save small to ../netlist_data/arith/small.pt
    torch.save(hete, '../N2V-data/hete_small.pt')
    homo = N2VDataset('../N2V-data', "homodata", 'rtl')
    # save small to ../netlist_data/arith/small.pt
    torch.save(homo, '../netlist_data/arith/homo_small.pt')