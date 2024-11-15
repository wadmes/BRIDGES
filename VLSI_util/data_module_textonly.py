"""
DataModule class for PyTorch Lightning
"""
import torch
from pytorch_lightning import LightningDataModule
from VLSI_util.data_textonly import stage1dataset
from torch_geometric.data import Batch
from torch.utils.data import DataLoader

class TrainCollater(object):
    def __init__(self, tokenizer, text_max_len):
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len
    
    def __call__(self, batch):
        # data_list= zip(*batch)
        graph_batch = Batch.from_data_list(batch)     
        text_batch = self.tokenizer(graph_batch.text, padding='max_length', truncation=True, max_length=self.text_max_len, return_tensors='pt')
        # netlist_batch = self.tokenizer(graph_batch.netlist, padding='max_length', truncation=True, max_length=self.text_max_len, return_tensors='pt')
        rtl_batch = self.tokenizer(graph_batch.rtl, padding='max_length', truncation=True, max_length=self.text_max_len, return_tensors='pt')
        return graph_batch, text_batch.input_ids, text_batch.attention_mask, rtl_batch.input_ids, rtl_batch.attention_mask, rtl_batch.input_ids, rtl_batch.attention_mask # use rtl to match the text instead
        # return graph_batch, text_batch.input_ids, text_batch.attention_mask, netlist_batch.input_ids, netlist_batch.attention_mask, rtl_batch.input_ids, rtl_batch.attention_mask





import random
from collections import defaultdict

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

"""
v2 version will handle train, eval, and test dataset separately
if len(dataset_path) == 2 and mix = False, then the first is train, the second is test/eval (divided equally)
if len(dataset_path) == 3 and mix = False, then the first is train, the second is eval, the third is test
train_test_ratio: [train_test_ratio, (1-train_test_ratio)/2, (1-train_test_ratio)/2] ]
"""
class Stage1DM_v2(LightningDataModule):
    def __init__(
        self,
        num_workers: int = 0,
        mix: bool = False, # whether to mix the dataset
        dataset_path: list = [],
        text_max_len: int = 256,
        tokenizer=None,
        seed: int = 42,
        args=None,
        train_test_ratio: float = 0.9,
    ):
        super().__init__()
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.match_batch_size = args.match_batch_size
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len
        print('Loading Netlist dataset...')

        raw_datasets = [torch.load(ds_path) for ds_path in dataset_path]
        if mix is False:
            print("Not mix the dataset")
            if len(raw_datasets) == 2:
                self.train_dataset = stage1dataset(raw_datasets[0]['graphs'])
                split_datasets = random_split_by_rtl([0.5,0.5], raw_datasets[1]['rtl_id_list'], raw_datasets[1]['graphs'], seed)
                self.val_dataset = stage1dataset(split_datasets[0])
                self.test_dataset = stage1dataset(split_datasets[1])
                print(f"Use TWO DATASETS! train: {len(self.train_dataset)}, val: {len(self.val_dataset)}, test: {len(self.test_dataset)}")
                del split_datasets, raw_datasets
            elif len(raw_datasets) == 1:
                split_datasets = random_split_by_rtl([train_test_ratio, (1-train_test_ratio)/2, (1-train_test_ratio)/2], raw_datasets[0]['rtl_id_list'], raw_datasets[0]['graphs'], seed)
                self.train_dataset = stage1dataset(split_datasets[0])
                self.val_dataset = stage1dataset(split_datasets[1])
                self.test_dataset = stage1dataset(split_datasets[2])
                del split_datasets, raw_datasets
                print(f"Use ONE DATASET! train: {len(self.train_dataset)}, val: {len(self.val_dataset)}, test: {len(self.test_dataset)}")
            elif len(raw_datasets) == 3:
                self.train_dataset = stage1dataset(raw_datasets[0]['graphs'])
                self.val_dataset = stage1dataset(raw_datasets[1]['graphs'])
                self.test_dataset = stage1dataset(raw_datasets[2]['graphs'])
                print(f"Use THREE DATASETS! train: {len(self.train_dataset)}, val: {len(self.val_dataset)}, test: {len(self.test_dataset)}")
            else:
                raise ValueError("Invalid dataset path length")
        else:
            # if mix datasets, we need to change the rtl_id
            train_graphs = []
            val_graphs = []
            test_graphs = []
            max_rtlid = 0
            for ds in raw_datasets:
                print(f"Dataset length: {len(ds['graphs'])}")
                split_datasets = random_split_by_rtl([train_test_ratio, (1-train_test_ratio)/2, (1-train_test_ratio)/2], ds['rtl_id_list'], ds['graphs'], seed)
                for split_dataset in split_datasets:
                    for graph in split_dataset:
                        graph.rtl_id += max_rtlid
                train_graphs.extend(split_datasets[0])
                val_graphs.extend(split_datasets[1])
                test_graphs.extend(split_datasets[2])
                max_rtlid += max(ds['rtl_id_list']) + 1
                print(f"max_rtlid: {max_rtlid}")
                del split_datasets
            print(f"Use MIXED DATASETS! train: {len(train_graphs)}, val: {len(val_graphs)}, test: {len(test_graphs)}")
            self.train_dataset = stage1dataset(train_graphs)
            self.val_dataset = stage1dataset(val_graphs)
            self.test_dataset = stage1dataset(test_graphs)


        # match loader is to check the RAG peformance (only difference with the original dataloader is the batch size )

        self.val_match_loader = DataLoader(self.val_dataset, batch_size=self.match_batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=False, drop_last=False, persistent_workers=True, collate_fn=TrainCollater(self.tokenizer, text_max_len))
        self.test_match_loader = DataLoader(self.test_dataset, batch_size=self.match_batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=False, drop_last=False, persistent_workers=True, collate_fn=TrainCollater(self.tokenizer, text_max_len))
    
    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=False, drop_last=True, persistent_workers=True, collate_fn=TrainCollater(self.tokenizer, self.text_max_len))
        return loader

    def val_dataloader(self):

        loader = DataLoader(self.val_dataset, batch_size=self.eval_batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=False, drop_last=False, persistent_workers=True, collate_fn=TrainCollater(self.tokenizer, self.text_max_len))
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--train_batch_size', type=int, default=16)
        parser.add_argument('--eval_batch_size', type=int, default=4)
        parser.add_argument('--match_batch_size', type=int, default=64)
        parser.add_argument("--dataset_path", action="extend", nargs="+", type=str, default=["/home/weili3/VLSI-LLM-Graph/VLSI_util/RTLCoder26532-textonly.pt","/home/weili3/VLSI-LLM-Graph/VLSI_util/MGVerilog11144-textonly.pt"])
        # mix: whether to mix the dataset, default is True
        import argparse
        parser.add_argument('--mix', default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument('--text_max_len', type=int, default=2048)
        return parent_parser
    