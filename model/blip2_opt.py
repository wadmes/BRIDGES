"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel
from torch_geometric.loader.dataloader import Collater
from torch_geometric.data import Data
import numpy as np
from lavis.models.blip2_models.blip2 import (
    # Blip2Base,
    disabled_train,
)
from model.blip2 import Blip2Base
from transformers import AutoTokenizer
from transformers import OPTForCausalLM
# from opendelta import LoraModel
# from opendelta.delta_models.lora import LoraConfig
# from opendelta.delta_configs

opt_model_list = [
    "facebook/galactica-125m",
    "facebook/galactica-1.3b",
    "facebook/galactica-6.7b",
    "facebook/galactica-30b",
]

def mask_by_len(input, lens, fill_value=0):
    '''
    input: shape = [N, D]
    lens: shape = [N]
    '''
    mask = torch.arange(input.shape[1], device=input.device).reshape(1, -1)
    mask = mask < lens.reshape(-1, 1)
    input[mask] = fill_value
    return input

import re
SPLIT_MARKER = f"SPL{1}T-TH{1}S-Pl3A5E"

CUSTOM_SEQ_RE = re.compile(r"(\[START_(DNA|SMILES|I_SMILES|AMINO)])(.*?)(\[END_\2])")


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

def smiles_handler(text, mol_ph):
    smiles_list = []
    for match in CUSTOM_SEQ_RE.finditer(text):
        smiles = match.group(3)
        smiles_list.append(smiles)
    
    text = CUSTOM_SEQ_RE.sub(r'\1\3\4%s' % (mol_ph), text)
    text = escape_custom_split_sequence(text)
    return text, smiles_list


class Blip2OPT(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """
    def __init__(
        self,
        bert_name,
        gin_num_layers,
        gin_hidden_dim,
        gin_drop_ratio,
        tune_gnn=False,
        num_query_token=32,
        cross_attention_freq=2,
        llm_tune='freeze',
        peft_dir='',
        opt_model="facebook/galactica-1.3b",
        prompt="",
        args=None,
    ):
        super().__init__()
        self.args = args

        self.graph_encoder, self.ln_graph = self.init_graph_encoder(gin_num_layers, gin_hidden_dim, gin_drop_ratio)
        self.tune_gnn = tune_gnn
        if not tune_gnn:
            for name, param in self.graph_encoder.named_parameters():
                param.requires_grad = False
            self.graph_encoder = self.graph_encoder.eval()
            self.graph_encoder.train = disabled_train
            logging.info("freeze graph encoder")
        
        self.num_query_token = num_query_token
        self.Qformer, self.query_tokens = self.init_Qformer(bert_name, num_query_token, self.graph_encoder.num_features, cross_attention_freq)
        ### remove the unused parameters
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        ## initialize opt model
        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False, padding_side='right')
        self.opt_tokenizer.add_special_tokens({'pad_token': '<pad>'})
        self.opt_tokenizer.add_tokens('<graph>') # graph placeholder
        self.opt_tokenizer.add_tokens('[START_NETLIST_GRAPH]') 
        self.opt_tokenizer.add_tokens('[END_NETLIST_GRAPH]')
        self.mol_token = '<graph>'
        self.opt_tokenizer.graph_token_id = self.opt_tokenizer("<graph>", add_special_tokens=False).input_ids[0]

        self.collater = Collater([], [])
        
        if opt_model == 'facebook/galactica-125m':
            self.opt_model = OPTForCausalLM.from_pretrained(opt_model, torch_dtype=torch.bfloat16)
        else:
            if torch.cuda.is_bf16_supported():
                self.opt_model = OPTForCausalLM.from_pretrained(opt_model, torch_dtype=torch.bfloat16)
            else:
                self.opt_model = OPTForCausalLM.from_pretrained(opt_model, torch_dtype=torch.float16)
        
        # this is to tell the model to use the new token
        self.opt_model.resize_token_embeddings(len(self.opt_tokenizer)) ## this will cause bug when full fine-tuning the opt model

        self.llm_tune = llm_tune
        if llm_tune == 'lora':
            if peft_dir:
                self.opt_model = PeftModel.from_pretrained(self.opt_model, peft_dir, is_trainable=True)
            else:
                if self.args.peft_config:
                    peft_config = LoraConfig(**LoraConfig.from_json_file(self.args.peft_config))
                else:
                    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
                self.peft_config = peft_config
                self.opt_model = get_peft_model(self.opt_model, peft_config)
                self.opt_model.print_trainable_parameters()
        elif llm_tune == 'freeze':
            for name, param in self.opt_model.named_parameters():
                param.requires_grad = False
        elif llm_tune == 'full':
            pass
        else:
            raise NotImplementedError()

        ## fixme: this is different from the original BLIP2
        self.eos_token_id = self.opt_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]

        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )
        
        ## fixme: no prompt yet
        self.prompt = prompt
        # prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        # self.prompt_length = prompt_tokens.attention_mask.sum(1)


    def forward(self, batch):
        # graphs, smiles_prompt_tokens, text_tokens
        graphs, prompt_tokens, text_tokens = batch
        graph_embeds, graph_masks = self.graph_encoder(graphs)
        if not self.tune_gnn:
            graph_embeds = graph_embeds.detach()
        graph_embeds = self.ln_graph(graph_embeds)
        device = graph_embeds.device
        query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=graph_embeds,
             # fixme: check whether this mask is correct
            return_dict=True,
        )
        mol_tokens = self.opt_proj(query_output.last_hidden_state)
        
        empty_targets = torch.ones(prompt_tokens.attention_mask.shape, dtype=torch.long).to(device).fill_(-100)
        targets = text_tokens.input_ids.masked_fill(
            text_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)
        prompt_embeds[prompt_tokens.is_graph_token] = mol_tokens.flatten(0, 1) # change mol placeholder to the actual mol tokens from graph
        inputs_embeds = self.opt_model.get_input_embeddings()(text_tokens.input_ids)
        inputs_embeds = torch.cat((prompt_embeds, inputs_embeds), dim=1)
        attention_mask = torch.cat([prompt_tokens.attention_mask, text_tokens.attention_mask], dim=1)
        
        outputs = self.opt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        return {"loss": loss}

    def forward_reaction(self, batch):
        reaction_tokens, notes_tokens, graphs = batch
        graph_embeds, graph_masks = self.graph_encoder(graphs)
        if not self.tune_gnn:
            graph_embeds = graph_embeds.detach()
        graph_embeds = self.ln_graph(graph_embeds)
        device = graph_embeds.device
        query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=graph_embeds,
             # fixme: check whether this mask is correct
            return_dict=True,
        )
        mol_tokens = self.opt_proj(query_output.last_hidden_state) # shape = [mol_num, num_query_token, D]

        if False:
            if self.llm_tune:
                react_embeds = self.opt_model.model.get_decoder().embed_tokens(reaction_tokens.input_ids) # shape = [B, max_len, D]
                notes_embeds = self.opt_model.model.get_decoder().embed_tokens(notes_tokens.input_ids)
            else:
                react_embeds = self.opt_model.model.decoder.embed_tokens(reaction_tokens.input_ids) # shape = [B, max_len, D]
                notes_embeds = self.opt_model.model.decoder.embed_tokens(notes_tokens.input_ids) # shape = [B, max_len, D]
        else:
            react_embeds = self.opt_model.get_input_embeddings()(reaction_tokens.input_ids)
            notes_embeds = self.opt_model.get_input_embeddings()(notes_tokens.input_ids)

        react_embeds[reaction_tokens.is_ph_token] = mol_tokens.flatten(0, 1)
        inputs_embeds = torch.cat((react_embeds, notes_embeds), dim=1)

        targets = notes_tokens.input_ids.masked_fill(
            notes_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        empty_targets = (
            torch.ones(reaction_tokens.attention_mask.shape, dtype=torch.long).to(device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)
        attention_mask = torch.cat([reaction_tokens.attention_mask, notes_tokens.attention_mask], dim=1)

        outputs = self.opt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        do_sample=False,
        num_beams=5,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        graphs = samples['graphs']
        prompt_tokens = samples['prompt_tokens']
        # prompt_lens = samples['prompt_lens']
        # with self.maybe_autocast():
        graph_embeds, graph_masks = self.graph_encoder(graphs)
        graph_embeds = self.ln_graph(graph_embeds)

        query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=graph_embeds,
            
            return_dict=True,
        )
        mol_tokens = self.opt_proj(query_output.last_hidden_state)
        
        prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)
        prompt_embeds[prompt_tokens.is_graph_token] = mol_tokens.flatten(0, 1)

        outputs = self.opt_model.generate(
            inputs_embeds=prompt_embeds,
            attention_mask=prompt_tokens.attention_mask,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            # pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
            # use_cache=False,
        )
        output_text = self.opt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        output_text = [text.strip() for text in output_text]
        return output_text

    """
    This should be a deprecated function
    """
    @torch.no_grad()
    def blip_qa(
        self, 
        samples,
        do_sample=False,
        num_beams=5,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        output_scores=False,
        ):

        device = next(self.parameters()).device
        
        ## data processing
        prompts = samples['prompts'] # assume list of strings
        prepared_prompts = []
        mol_list = []
        for p in prompts:
            text, smiles = smiles_handler(p, self.mol_token * self.num_query_token)
            prepared_prompts.append(text)
            mol_list.extend(smiles) # this should be 
        
        prompt_tokens = self.opt_tokenizer(prepared_prompts,
                                           truncation=False,
                                           padding='longest',
                                           add_special_tokens=True,
                                        #    max_length=self.args.max_len[],
                                           return_tensors='pt',
                                           return_attention_mask=True).to(device)
        
        ## forward function
        prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)
        
        if len(mol_list) > 0:
            graphs = self.collater(mol_list).to(device)
            is_graph_token = (prompt_tokens.input_ids == self.mol_token) # shape = [B, max_len]
            ## graph forward
            graph_embeds, graph_masks = self.graph_encoder(graphs)
            graph_embeds = self.ln_graph(graph_embeds)
            query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=graph_embeds,
                 # fixme: check whether this mask is correct
                return_dict=True,
            )
            mol_tokens = self.opt_proj(query_output.last_hidden_state) # shape = [mol_num, num_query_token, D]
            ## replace mol tokens
            prompt_embeds[is_graph_token] = mol_tokens.flatten(0, 1)
        
        if output_scores:
            outputs = self.opt_model.generate(
                    inputs_embeds=prompt_embeds,
                    attention_mask=prompt_tokens.attention_mask,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_length=max_length,
                    min_length=min_length,
                    # pad_token_id=self.pad_token_id,
                    eos_token_id=self.eos_token_id,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                    output_scores=True,
                    return_dict_in_generate=True
                    # use_cache=False,
            )
            return outputs
        else:
            outputs = self.opt_model.generate(
                    inputs_embeds=prompt_embeds,
                    attention_mask=prompt_tokens.attention_mask,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_length=max_length,
                    min_length=min_length,
                    # pad_token_id=self.pad_token_id,
                    eos_token_id=self.eos_token_id,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                    # use_cache=False,
                )
            output_text = self.opt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text = [text.strip() for text in output_text]
            return output_text
    
    @torch.no_grad()
    def opt_qa(
        self, 
        samples,
        do_sample=False,
        num_beams=5,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        output_scores=False,
        ):

        device = next(self.parameters()).device
        ## data processing
        prompts = samples['prompts'] # assume list of strings
        prompts = [escape_custom_split_sequence(p) for p in prompts]
        
        prompt_tokens = self.opt_tokenizer(prompts,
                                           truncation=False,
                                           padding='longest',
                                           add_special_tokens=True,
                                        #    max_length=self.args.max_len[],
                                           return_tensors='pt',
                                           return_attention_mask=True).to(device)
        
        prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)

        if output_scores:
            ## forward function
            outputs = self.opt_model.generate(
                    inputs_embeds=prompt_embeds,
                    attention_mask=prompt_tokens.attention_mask,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_length=max_length,
                    min_length=min_length,
                    # pad_token_id=self.pad_token_id,
                    eos_token_id=self.eos_token_id,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                    # use_cache=False,
                    output_scores=True,
                    return_dict_in_generate=True
                )
            return outputs
        else:
            ## forward function
            outputs = self.opt_model.generate(
                    inputs_embeds=prompt_embeds,
                    attention_mask=prompt_tokens.attention_mask,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_length=max_length,
                    min_length=min_length,
                    # pad_token_id=self.pad_token_id,
                    eos_token_id=self.eos_token_id,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                    # use_cache=False,
                )
            output_text = self.opt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text = [text.strip() for text in output_text]
            return output_text
        
    @torch.no_grad()
    def probe_qformer(
        self, 
        batch,
        do_sample=False,
        num_beams=5,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        ):
        with self.maybe_autocast():
            device = next(self.parameters()).device
            
            graphs, smiles_prompt_tokens, texts = batch
            graphs = graphs.to(device)
            ## graph forward
            graph_embeds, graph_masks = self.graph_encoder(graphs)
            graph_embeds = self.ln_graph(graph_embeds)
            query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=graph_embeds,
                 # fixme: check whether this mask is correct
                return_dict=True,
            )
            mol_tokens = self.opt_proj(query_output.last_hidden_state) # shape = [mol_num, num_query_token, D]
            B, num_q, D = mol_tokens.shape
            
            ## 
            embed_func = self.opt_model.get_input_embeddings()
            embed_weight = embed_func.weight # shape = [vocab_size, D]
            
            dis_metric = 'cos'
            topk = 10
            if dis_metric == 'cos':
                mol_tokens = F.normalize(mol_tokens, dim=-1, p=2)
                embed_weight = F.normalize(embed_weight, dim=-1, p=2)
                sim = mol_tokens.flatten(0, 1) @ embed_weight.T # shape = [mol_num * num_query_token, vocab_size]
            elif dis_metric == 'euc':
                sim = - torch.cdist(mol_tokens.flatten(0, 1), embed_weight, p=2)
                assert sim.shape == (B * num_q, embed_weight.shape[0])
            else:
                raise NotImplementedError()
            _, topk_ids = torch.topk(sim, k=topk, dim=-1) # shape = [mol_num * num_query_token, k]
            knn_decode_strings = self.opt_tokenizer.batch_decode(topk_ids.flatten())
            knn_decode_strings = np.asarray(knn_decode_strings).reshape(B, num_q, topk).tolist() # shape = [mol_num, num_query_token, topk]
            knn_decode_strings = [[' '.join(ii) for ii in i] for i in knn_decode_strings] # shape = [mol_num, num_query_token]
            if False:
                ### print for presentation
                assert len(knn_decode_strings) == len(texts)
                for predict, text in zip(knn_decode_strings, texts):
                    print('----------------------------')
                    print(predict)
                    print(text)
            return knn_decode_strings
            