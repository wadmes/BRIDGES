# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
from pytorch_lightning import LightningDataModule
import torch_geometric
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
from torch_geometric.loader.dataloader import Collater
from data_provider.molecule_caption_dataset import MoleculeCaption, MoleculeCaptionV2
import re

# we split individual characters inside special tokens like [START_DNA]
CUSTOM_SEQ_RE = re.compile(r"(\[START_(DNA|SMILES|I_SMILES|AMINO)])(.*?)(\[END_\2])")

# token added to implement a custom sequence tokenization. This token is added at
# corpus cleaning step and removed in pretokenization. The digits are added to increase the chance
# that they do not occur in the corpus. The digits are escaped so that the token does not appear
# literally in the source code in case we ever include it in the training data.
SPLIT_MARKER = f"SPL{1}T-TH{1}S-Pl3A5E"

def _insert_split_marker(m: re.Match):
    """
    Applies split marker based on a regex match of special tokens such as
    [START_DNA].

    Parameters
    ----------
    n : str
        Input text to split

    Returns
    ----------
    str - the text with the split token added
    """
    start_token, _, sequence, end_token = m.groups()
    sequence = re.sub(r"(.)", fr"{SPLIT_MARKER}\1", sequence, flags=re.DOTALL)
    return f"{start_token}{sequence}{SPLIT_MARKER}{end_token}"


def smiles_handler(text, mol_ph, is_gal=True, graph_only=False):
    smiles_list = []
    for match in CUSTOM_SEQ_RE.finditer(text):
        smiles = match.group(3)
        smiles_list.append(smiles)

    if graph_only:
        text = CUSTOM_SEQ_RE.sub(r'%s' % (mol_ph), text)
        return text, smiles_list
    if is_gal:
        text = CUSTOM_SEQ_RE.sub(r'\1\3\4%s' % (mol_ph), text)
        text = escape_custom_split_sequence(text)
        return text, smiles_list
    else:
        text = CUSTOM_SEQ_RE.sub(r'\3%s' % (mol_ph), text)
        return text, smiles_list


def escape_custom_split_sequence(text):
    """
    Applies custom splitting to the text for GALILEO's tokenization

    Parameters
    ----------
    text : str
        Input text to split

    Returns
    ----------
    str - the text with the split token added
    """
    return CUSTOM_SEQ_RE.sub(_insert_split_marker, text)

class TrainCollater:
    def __init__(self, tokenizer, text_max_len, mol_ph, graph_token_id, is_gal=True, graph_only=False):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.graph_token_id = graph_token_id
        self.is_gal = is_gal
        self.graph_only = graph_only
        
    def __call__(self, batch):
        graphs, texts, smiles_prompt = zip(*batch)
        graphs = self.collater(graphs)
        
        ## deal with prompt
        smiles_prompt = [smiles_handler(p, self.mol_ph, self.is_gal, self.graph_only)[0] for p in smiles_prompt]

        self.tokenizer.padding_side = 'left'
        smiles_prompt_tokens = self.tokenizer(text=smiles_prompt, 
                                              truncation=False,
                                              padding='longest',
                                              add_special_tokens=True,
                                              return_tensors='pt',
                                              return_attention_mask=True)

        is_graph_token = smiles_prompt_tokens.input_ids == self.graph_token_id
        smiles_prompt_tokens['is_graph_token'] = is_graph_token

        self.tokenizer.padding_side = 'right'
        text_tokens = self.tokenizer(text=texts,
                                     truncation=True,
                                     padding='longest',
                                     add_special_tokens=True,
                                     max_length=self.text_max_len,
                                     return_tensors='pt',
                                     return_attention_mask=True)
        return graphs, smiles_prompt_tokens, text_tokens
    

class InferenceCollater:
    def __init__(self, tokenizer, text_max_len, mol_ph, graph_token_id, is_gal=True, graph_only=False):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.graph_token_id = graph_token_id
        self.is_gal = is_gal
        self.graph_only = graph_only
        
    def __call__(self, batch):
        graphs, texts, smiles_prompt = zip(*batch)
        graphs = self.collater(graphs)
        smiles_prompt = [smiles_handler(p, self.mol_ph, self.is_gal, self.graph_only)[0] for p in smiles_prompt]

        ## deal with prompt
        self.tokenizer.padding_side = 'left'
        smiles_prompt_tokens = self.tokenizer(smiles_prompt, 
                                              return_tensors='pt', 
                                              add_special_tokens=True,
                                            # max_length=self.text_max_len, 
                                              padding='longest', 
                                              truncation=False, 
                                              return_attention_mask=True)
        
        is_graph_token = smiles_prompt_tokens.input_ids == self.graph_token_id
        smiles_prompt_tokens['is_graph_token'] = is_graph_token
        return graphs, smiles_prompt_tokens, texts
    

class Stage2DM(LightningDataModule):
    def __init__(
        self,
        mode: str = 'pretrain',
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = './data/',
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
        self.graph_only = args.graph_only
        if False:
            self.pretrain_dataset = MoleculeCaption(root+f'/pretrain/', text_max_len, self.prompt, args.filtered_cid_path)
            self.train_dataset = MoleculeCaption(root+f'/train/', text_max_len, self.prompt)
            self.val_dataset = MoleculeCaption(root + '/valid/', text_max_len, self.prompt)
            self.test_dataset = MoleculeCaption(root + '/test/', text_max_len, self.prompt)
        else:
            self.pretrain_dataset = MoleculeCaptionV2(root+f'pretrain.pt', text_max_len, self.prompt)
            self.train_dataset = MoleculeCaptionV2(root+f'train.pt', text_max_len, self.prompt)
            self.val_dataset = MoleculeCaptionV2(root + f'valid.pt', text_max_len, self.prompt)
            self.test_dataset = MoleculeCaptionV2(root + f'test.pt', text_max_len, self.prompt)
        self.init_tokenizer(tokenizer)
        self.mol_ph_token = '<graph>' * self.args.num_query_token
        self.is_gal = args.opt_model.find('galactica') >= 0
        
    
    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.pretrain_dataset.tokenizer = tokenizer
        self.train_dataset.tokenizer = tokenizer
        self.val_dataset.tokenizer = tokenizer
        self.test_dataset.tokenizer = tokenizer
        self.graph_token_id = self.tokenizer.graph_token_id
        # self.tokenizer.graph_token_id = tokenizer("<graph>", add_special_tokens=False).input_ids[0]

    def train_dataloader(self):
        if self.mode == 'pretrain':
            loader = DataLoader(
                self.pretrain_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.graph_token_id, self.is_gal, self.graph_only),
            )
        elif self.mode == 'ft':
            loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.graph_token_id, self.is_gal, self.graph_only),
            )
        else:
            raise NotImplementedError
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
            collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.graph_token_id, self.is_gal, self.graph_only),
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.graph_token_id, self.is_gal, self.graph_only),
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
            collate_fn=InferenceCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.graph_token_id, self.is_gal, self.graph_only),
        )
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--use_smiles', action='store_true', default=False)
        parser.add_argument('--root', type=str, default='./data')
        parser.add_argument('--text_max_len', type=int, default=128)
        parser.add_argument('--prompt', type=str, default='The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. ')
        parser.add_argument('--filtered_cid_path', type=str, default=None)
        parser.add_argument('--graph_only', action='store_true', default=False)
        return parent_parser
    