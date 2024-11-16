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
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel

from lavis.models.blip2_models.blip2 import (
    # Blip2Base,
    disabled_train,
)
from model.blip2 import Blip2Base
from transformers import AutoImageProcessor, AutoTokenizer
from transformers import AutoModel, AutoModelForCausalLM, PreTrainedModel, BitsAndBytesConfig
from model.modeling_llama import LlamaForCausalLM


 
llama_model_list = [
    "decapoda-research/llama-13b-hf",
    "decapoda-research/llama-7b-hf",
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

# @registry.register_model("blip2")
# @registry.register_model("blip2_feature_extractor")
class Blip2Llama(Blip2Base):
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
        lora_tuning=False,
        peft_dir='',
        llm_model="meta-llama/Llama-3.2-1B",
        prompt="",
        args=None,
    ):
        super().__init__()
        self.graph_encoder, self.ln_graph = self.init_graph_encoder(gin_num_layers, gin_hidden_dim, gin_drop_ratio)
        self.tune_gnn = tune_gnn
        if not tune_gnn:
            for name, param in self.graph_encoder.named_parameters():
                param.requires_grad = False
            self.graph_encoder = self.graph_encoder.eval()
            self.graph_encoder.train = disabled_train
            logging.info("freeze graph encoder")
        
        self.Qformer, self.query_tokens = self.init_Qformer(bert_name, num_query_token, self.graph_encoder.num_features, cross_attention_freq)
        ### remove the unused parameters
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        ## initialize opt model
        
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model, use_fast=False, add_eos_token = True)
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'additional_special_tokens': ['[START_NETLIST_GRAPH]','[END_NETLIST_GRAPH]', '<graph>']})

        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model,                                                                    
                                                            resume_download=True,
                                                            torch_dtype=torch.bfloat16,
                                                            quantization_config=BitsAndBytesConfig(load_in_4bit=True) if args.load_in_4bit else None,
                                                            attn_implementation="flash_attention_2",)
        # self.llm_model = LlamaForCausalLM.from_pretrained(llm_model)
        # self.gradient_checkpointing_enable()
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        
        self.lora_tuning = lora_tuning
        if lora_tuning:
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
            self.llm_model = get_peft_model(self.llm_model, peft_config)
            self.llm_model.print_trainable_parameters()
        else:
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False

        self.eos_token_id = self.llm_tokenizer.eos_token_id
        self.pad_token_id = self.llm_tokenizer.pad_token_id

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )
        
        ## fixme: no prompt yet
        self.prompt = prompt
        # prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        # self.prompt_length = prompt_tokens.attention_mask.sum(1)

    def forward(self, batch):
        # graphs, smiles_prompt_tokens, text_tokens
        graphs, prompt_tokens, text_tokens, _ = batch # text is the function description
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
        graph_tokens = self.llm_proj(query_output.last_hidden_state)
        
        empty_targets = torch.ones(prompt_tokens.attention_mask.shape, dtype=torch.long,device = self.llm_model.device).fill_(-100)
        targets = text_tokens.input_ids.masked_fill(text_tokens.input_ids == self.llm_tokenizer.pad_token_id, -100)
        targets = torch.cat([empty_targets, targets], dim=1)

        prompt_embeds = self.llm_model.get_input_embeddings()(prompt_tokens.input_ids)
        prompt_embeds[prompt_tokens.is_graph_token] = graph_tokens.flatten(0, 1) # change graph placeholder to the actual graph tokens from graph
        inputs_embeds = self.llm_model.get_input_embeddings()(text_tokens.input_ids)
        inputs_embeds = torch.cat((prompt_embeds, inputs_embeds), dim=1)
        attention_mask = torch.cat([prompt_tokens.attention_mask, text_tokens.attention_mask], dim=1)
        
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        return {"loss": loss}

    @torch.no_grad()
    def generate_old(
        self,
        samples,
        do_sample=False,
        num_beams=5,
        max_length=256,
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
        with self.maybe_autocast():
            graph_embeds, graph_masks = self.graph_encoder(graphs)
            graph_embeds = self.ln_graph(graph_embeds)

            query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=graph_embeds,
                
                return_dict=True,
            )

            device = graph_embeds.device
            inputs_llm = self.llm_proj(query_output.last_hidden_state)
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long, device=device)

            attention_mask = torch.cat([atts_llm, prompt_tokens.attention_mask], dim=1)
            

            inputs_embeds = self.llm_model.get_input_embeddings()(prompt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, prompt_tokens.attention_mask], dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
                # use_cache=False,
            )
            # outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
            output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text = [text.strip() for text in output_text]
            return output_text
        
    @torch.no_grad()
    def generate(
        self,
        samples,
        do_sample=False,
        num_beams=1,
        max_length=512,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        11-13-2024: This function is modified to match the prompt - graph token relations used in the forward function
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
        import timeit
        start = timeit.default_timer()
        graphs = samples['graphs']
        prompt_tokens = samples['prompt_tokens']
        # prompt_lens = samples['prompt_lens']
        graph_embeds, graph_masks = self.graph_encoder(graphs)
        graph_embeds = self.ln_graph(graph_embeds)

        query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=graph_embeds,
            
            return_dict=True,
        )
        graph_tokens = self.llm_proj(query_output.last_hidden_state)
        prompt_embeds = self.llm_model.get_input_embeddings()(prompt_tokens.input_ids)
        prompt_embeds[prompt_tokens.is_graph_token] = graph_tokens.flatten(0, 1)
        print("Time taken to generate prompt_embeds: ", timeit.default_timer() - start)
        outputs = self.llm_model.generate(
            inputs_embeds=prompt_embeds,
            attention_mask=prompt_tokens.attention_mask,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
            # use_cache=False,
        )
        # outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
        # print("Time taken to generate outputs: ", timeit.default_timer() - start)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]
        # print("Time taken to decode outputs: ", timeit.default_timer() - start)
        return output_text
    
    @torch.no_grad()
    def generate_with_candidates(
        self,
        samples,
        candidate_tokens,
        candidates
    ):
        """
        11-13-2024: This function is modified to match the prompt - graph token relations used in the forward function
        Args:
            samples (dict): A dictionary containing the following keys:
                - graph, 
                - prompt_tokens,
            candidate_tokens,
            candidates
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        import timeit
        start = timeit.default_timer()
        graphs = samples['graphs']
        prompt_tokens = samples['prompt_tokens']
        # prompt_lens = samples['prompt_lens']
        graph_embeds, graph_masks = self.graph_encoder(graphs)
        graph_embeds = self.ln_graph(graph_embeds)

        query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=graph_embeds,
            return_dict=True,
        )
        graph_tokens = self.llm_proj(query_output.last_hidden_state)
        prompt_embeds = self.llm_model.get_input_embeddings()(prompt_tokens.input_ids)
        prompt_embeds[prompt_tokens.is_graph_token] = graph_tokens.flatten(0, 1) # shape is [N,seq_len, D]
        prompt_attention_mask = prompt_tokens["attention_mask"]  # Shape: (N, seq_len)
        
        candidate_input_ids = candidate_tokens["input_ids"]  # Shape: (K, response_len)
        candidate_attention_mask = candidate_tokens["attention_mask"]  # Shape: (K, response_len)
        candidate_embeds = self.llm_model.get_input_embeddings()(candidate_input_ids) # Shape: (K, response_len, D)

        num_prompts, prompt_len = prompt_tokens.input_ids.size()
        num_candidates, candidate_len = candidate_input_ids.size()
        hidden_dimenstion = prompt_embeds.size(-1)

        # Expand prompts to match candidates
        expanded_prompt_input_embeds = prompt_embeds.unsqueeze(1).expand(num_prompts, num_candidates, prompt_len,hidden_dimenstion)
        expanded_prompt_attention_mask = prompt_attention_mask.unsqueeze(1).expand(num_prompts, num_candidates, prompt_len)

        # Expand candidates to match prompts
        expanded_candidate_input_embeds = candidate_embeds.unsqueeze(0).expand(num_prompts, num_candidates, candidate_len)
        expanded_candidate_attention_mask = candidate_attention_mask.unsqueeze(0).expand(num_prompts, num_candidates, candidate_len,hidden_dimenstion)

        # Concatenate prompts and candidates
        concatenated_input_ids = torch.cat([expanded_prompt_input_embeds, concatenated_input_ids], dim=-2) # Shape: (N, K, seq_len + response_len, D)
        concatenated_attention_mask = torch.cat([expanded_prompt_attention_mask, expanded_candidate_attention_mask], dim=-1)

        # Flatten for batch processing
        concatenated_input_ids = concatenated_input_ids.view(-1, concatenated_input_ids.size(-1))  # Shape: (N*K, seq_len + response_len)
        concatenated_attention_mask = concatenated_attention_mask.view(-1, concatenated_attention_mask.size(-1))  # Shape: (N*K, seq_len + response_len)

        # Prepare labels (same as input_ids, but pad tokens are ignored)
        labels = concatenated_input_ids.clone()
        labels[concatenated_attention_mask == 0] = -100  # Ignore padding tokens in loss calculation
        with torch.no_grad():
            outputs = self.llm_model(inputs_embeds=prompt_embeds,attention_mask=concatenated_attention_mask)
            logits = outputs.logits  # Shape: (N*K, seq_len + response_len, vocab_size)

        # Compute element-wise cross-entropy loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        token_losses = token_losses.view(shift_labels.size())  # Shape: (N*K, seq_len + response_len - 1)

        sequence_losses = token_losses[:,prompt_len-1:].sum(dim=-1)   # Sum losses for response tokens only
        sequence_losses = sequence_losses.view(num_prompts, num_candidates)  # Reshape to (N, K)

        probabilities = torch.exp(-sequence_losses)  # Convert loss to probabilities
        best_responses = [candidates[i] for i in probabilities.argmax(dim=1)]
        return best_responses
        