"""
DataModule class for PyTorch Lightning
"""
import torch
from pytorch_lightning import LightningDataModule
from VLSI_util.data import netlistDataset
from torch_geometric.data import Batch
from torch.utils.data import DataLoader

class TrainCollater(object):
    def __init__(self, tokenizer, text_max_len):
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len
    
    def __call__(self, batch):
        data_list, text_list = zip(*batch)
        graph_batch = Batch.from_data_list(data_list)        
        text_batch = self.tokenizer(text_list, padding='max_length', truncation=True, max_length=self.text_max_len, return_tensors='pt')
        return graph_batch, text_batch.input_ids, text_batch.attention_mask



class Stage1DM(LightningDataModule):
    def __init__(
        self,
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = './netlist_data/',
        text_max_len: int = 128,
        tokenizer=None,
        args=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.match_batch_size = args.match_batch_size
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len
        print('Loading Netlist dataset')
        self.train_dataset = torch.load(root + 'arith/hete_small.pt')
        self.val_dataset = torch.load(root + 'arith/hete_small.pt')
        self.val_dataset_match = torch.load(root + 'arith/hete_small.pt')
        self.test_dataset_match = torch.load(root + 'arith/hete_small.pt')
        self.val_match_loader = DataLoader(self.val_dataset_match, batch_size=self.match_batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=False, drop_last=False, persistent_workers=True, collate_fn=TrainCollater(self.tokenizer, text_max_len))
        self.test_match_loader = DataLoader(self.test_dataset_match, batch_size=self.match_batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=False, drop_last=False, persistent_workers=True, collate_fn=TrainCollater(self.tokenizer, text_max_len))
    
    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=False, drop_last=True, persistent_workers=True, collate_fn=TrainCollater(self.tokenizer, self.text_max_len))
        return loader

    def val_dataloader(self):

        loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=False, drop_last=False, persistent_workers=True, collate_fn=TrainCollater(self.tokenizer, self.text_max_len))
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=4)
        parser.add_argument('--match_batch_size', type=int, default=4)
        parser.add_argument('--root', type=str, default='./netlist_data/')
        parser.add_argument('--text_max_len', type=int, default=128)
        return parent_parser
    