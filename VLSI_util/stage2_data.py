# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
from pytorch_lightning import LightningDataModule
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader, Dataset
from torch_geometric.loader.dataloader import Collater
from rdkit import RDLogger
from torch_geometric.data import Batch
RDLogger.DisableLog('rdApp.*')

class TrainCollater:
    def __init__(self, tokenizer, text_max_len, mol_ph, graph_token_id, prompt):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.graph_token_id = graph_token_id
        self.prompt = prompt
        
    def __call__(self, batch):
        graphs, texts = zip(*batch)
        graph_batch = Batch.from_data_list(graphs)     
        
        self.tokenizer.paddding_side = 'left' # By setting the value to 'left', you're instructing the tokenizer to add padding tokens to the left side of a text sequence.
        prompt = [self.prompt.format(self.mol_ph)] * len(texts)
        prompt_tokens = self.tokenizer(text=prompt, 
                                              truncation=False,
                                              padding='longest',
                                              add_special_tokens=True,
                                              return_tensors='pt',
                                              return_attention_mask=True)

        # in smiles_handler, the token graph_ph_token (<graph>) is inserted
        is_graph_token = prompt_tokens.input_ids == self.graph_token_id # self.opt_tokenizer.graph_token_id = self.opt_tokenizer("<graph>", add_special_tokens=False).input_ids[0]
        prompt_tokens['is_graph_token'] = is_graph_token 
        # print(smiles_prompt_tokens.input_ids, self.graph_token_id)
        # print(is_graph_token)
        self.tokenizer.paddding_side = 'right'
        text_tokens = self.tokenizer(text=texts,
                                     truncation=True,
                                     padding='longest',
                                     add_special_tokens=True,
                                     max_length=self.text_max_len,
                                     return_tensors='pt',
                                     return_attention_mask=True)
        return graph_batch, prompt_tokens, text_tokens

    

class InferenceCollater:
    def __init__(self, tokenizer, text_max_len, mol_ph, graph_token_id, prompt):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.graph_token_id = graph_token_id
        self.prompt = prompt
    def __call__(self, batch):
        graphs, texts = zip(*batch)
        graph_batch = Batch.from_data_list(graphs)   
        ## deal with prompt
        self.tokenizer.paddding_side = 'left' # By setting the value to 'left', you're instructing the tokenizer to add padding tokens to the left side of a text sequence.
        prompt = [self.prompt.format(self.mol_ph)] * len(texts)
        prompt_tokens = self.tokenizer(text=prompt, 
                                              truncation=False,
                                              padding='longest',
                                              add_special_tokens=True,
                                              return_tensors='pt',
                                              return_attention_mask=True)

        # in smiles_handler, the token graph_ph_token (<graph>) is inserted
        is_graph_token = prompt_tokens.input_ids == self.graph_token_id # self.opt_tokenizer.graph_token_id = self.opt_tokenizer("<graph>", add_special_tokens=False).input_ids[0]
        prompt_tokens['is_graph_token'] = is_graph_token 
        return graph_batch, prompt_tokens, texts
    


class Stage2Netlist(LightningDataModule):
    def __init__(
        self,
        mode: str = 'pretrain',
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = './VLSI_util/',
        text_max_len: int = 128,
        tokenizer=None,
        args=None,
    ):
        super().__init__()
        self.args = args
        self.mode = mode
        self.batch_size = batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = num_workers
        self.text_max_len = text_max_len
        self.prompt = args.prompt
        self.data_set = torch.load(root + '/netlist.pt')
        self.train_dataset = self.data_set
        self.val_dataset = self.data_set
        self.test_dataset = self.data_set
        self.init_tokenizer(tokenizer)
        self.graph_ph_token = '<graph>' * self.args.num_query_token # ph is short for placeholder
        
    
    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        # self.pretrain_dataset.tokenizer = tokenizer
        self.train_dataset.tokenizer = tokenizer
        self.val_dataset.tokenizer = tokenizer
        self.test_dataset.tokenizer = tokenizer
        self.graph_token_id = self.tokenizer.graph_token_id
        self.tokenizer.graph_token_id = tokenizer("<graph>", add_special_tokens=False).input_ids[0]

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True,
            collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.graph_ph_token, self.graph_token_id, self.prompt),
        )
        return loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.graph_ph_token, self.graph_token_id, self.prompt),
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater(self.tokenizer, self.text_max_len, self.graph_ph_token, self.graph_token_id, self.prompt),
        )
        return [val_loader, test_loader]
    
    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater(self.tokenizer, self.text_max_len, self.graph_ph_token, self.graph_token_id, self.prompt),
        )
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--inference_batch_size', type=int, default=8)
        parser.add_argument('--use_smiles', action='store_true', default=False)
        parser.add_argument('--root', type=str, default='/scratch/scratch/weili3')
        parser.add_argument('--text_max_len', type=int, default=128)
        parser.add_argument('--prompt', type=str, default='The graph of this module is [START_NETLIST_GRAPH]{}[END__NETLIST_GRAPH].')
        return parent_parser
    