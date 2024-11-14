"""
DataModule class for PyTorch Lightning
"""
import torch
from pytorch_lightning import LightningDataModule
from VLSI_util.data import netlistDataset
from VLSI_util.data import stage1dataset
from torch_geometric.data import Batch
from torch.utils.data import DataLoader

class TrainCollater(object):
    def __init__(self, tokenizer, text_max_len):
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len
    
    def __call__(self, batch):
        data_list, text_list = zip(*batch)
        # print("data_list", data_list)
        graph_batch = Batch.from_data_list(data_list)        
        text_batch = self.tokenizer(text_list, padding='max_length', truncation=True, max_length=self.text_max_len, return_tensors='pt')
        return graph_batch, text_batch.input_ids, text_batch.attention_mask



class Stage1DM(LightningDataModule):
    def __init__(
        self,
        num_workers: int = 0,
        root: str = './VLSI_util/',
        text_max_len: int = 128,
        tokenizer=None,
        args=None,
    ):
        super().__init__()
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.match_batch_size = args.match_batch_size
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len
        print('Loading Netlist dataset...')
        self.train_dataset = torch.load(root + 'train_w_syn.pt')
        self.val_dataset = torch.load(root + 'eval_w_syn.pt')
        self.test_dataset = torch.load(root + 'test_w_syn.pt')

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
        parser.add_argument('--root', type=str, default='./VLSI_util/')
        parser.add_argument('--text_max_len', type=int, default=256)
        return parent_parser
    



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
        text_max_len: int = 128,
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
        if mix is False:
            print("Not mix the dataset")
            raise NotImplementedError
        else:
            # if mix datasets, we need to change the rtl_id
            train_graphs = []
            val_graphs = []
            test_graphs = []
            max_rtlid = 0
            for ds_path in dataset_path:
                ds = torch.load(ds_path)
                this_train = torch.load(ds_path.replace('.pt', '_train.pt'))
                this_val = torch.load(ds_path.replace('.pt', '_val.pt'))
                this_test = torch.load(ds_path.replace('.pt', '_test.pt'))
                for split_dataset in [this_train, this_val, this_test]:
                    for graph in split_dataset:
                        graph.rtl_id += max_rtlid
                train_graphs.extend(this_train)
                val_graphs.extend(this_val)
                test_graphs.extend(this_test)
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
        parser.add_argument("--dataset_path", action="extend", nargs="+", type=str, default=["/scratch/weili3/RTLCoder26532.pt","/scratch/weili3/MGVerilog11144.pt"])
        # mix: whether to mix the dataset, default is True
        import argparse
        parser.add_argument('--mix', default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument('--text_max_len', type=int, default=256)
        return parent_parser
    