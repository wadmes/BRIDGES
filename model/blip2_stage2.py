import os
from typing import Any, Dict
import torch
from model.blip2_opt import Blip2OPT
from model.blip2_llama import Blip2Llama
from model.blip2_t5 import Blip2T5
import pytorch_lightning as pl
from torch import optim
from lavis.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
import json
import torch.distributed as dist
from peft import LoraConfig, TaskType
from model.help_funcs import caption_evaluate, AttrDict
from transformers import Adafactor
import timeit

def load_ignore_unexpected(model, state_dict):
    keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in keys}
    
    ## try to print keys that are not included
    model.load_state_dict(state_dict, strict=True) # for Q-former, it is not strict (we have expriments without Q-former)

def get_module_state_dict(state_dict, module_name):
    module_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(module_name):
            key = key[len(module_name) + 1:]
            if key == '':
                return value
            module_state_dict[key] = value
    return module_state_dict
# peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
class Blip2Stage2(pl.LightningModule):
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # checkpoint.pop('optimizer_states')
        to_be_removed = []
        for key, value in checkpoint['state_dict'].items():
            try:
                if not self.get_parameter(key).requires_grad:
                    to_be_removed.append(key)
            except AttributeError:
                to_be_removed.append(key)
        for key in to_be_removed:
            checkpoint['state_dict'].pop(key)
    
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)

        self.args = args
        if not hasattr(args, 'do_sample'):
            args.do_sample = False
        self.caption_eval_epoch = args.caption_eval_epoch
        self.do_sample = args.do_sample
        self.num_beams = args.num_beams
        self.max_len = args.max_len
        self.min_len = args.min_len
        self.reaction_weight = args.reaction_weight
        self.llm_tune = args.llm_tune
        if args.opt_model.find('galactica') >= 0:
            self.blip2opt = Blip2OPT(args.bert_name, args.gin_num_layers, args.gin_hidden_dim, args.drop_ratio, args.tune_gnn, args.num_query_token, args.cross_attention_freq, args.llm_tune, args.peft_dir, args.opt_model, args.prompt, args)
        elif args.opt_model.find('llama') >= 0 or args.opt_model.find('vicuna') >= 0:
            self.blip2opt = Blip2Llama(args.bert_name, args.gin_num_layers, args.gin_hidden_dim, args.drop_ratio, args.tune_gnn, args.num_query_token, args.cross_attention_freq, args.llm_tune, args.peft_dir, args.opt_model, args.prompt, args)
        elif args.opt_model.find('t5') >= 0:
            self.blip2opt = Blip2T5(args.bert_name, args.gin_num_layers, args.gin_hidden_dim, args.drop_ratio, args.tune_gnn, args.num_query_token, args.cross_attention_freq, args.llm_tune, args.peft_dir, args.opt_model, args.prompt, args)
        else:
            raise NotImplementedError()
        self.tokenizer = self.blip2opt.init_tokenizer()
        self.save_hyperparameters(args)
        
        if self.args.task == 'type_pred':
            self.candidates = ['Encryption Unit<|end_of_text|>', 'Data Path Unit<|end_of_text|>', 'Control Logic Unit<|end_of_text|>', 'Arithmetic Unit<|end_of_text|>', 'Communication Protocol Unit<|end_of_text|>', 'Signal Processing Unit<|end_of_text|>', 'Clock Management Unit<|end_of_text|>', 'Others<|end_of_text|>']
            self.candidate_tokens = self.blip2opt.llm_tokenizer(self.candidates,truncation=True, padding='longest', return_tensors='pt', return_attention_mask=True,add_special_tokens=False).to("cuda")
    

    def load_from_stage1_checkpoint(self, path,use_qformer = 1):
        ckpt = torch.load(path, map_location='cpu')
        state_dict = ckpt['state_dict']
        graph_encoder_dict = get_module_state_dict(state_dict, 'blip2qformer.graph_encoder')
        qformer_dict = get_module_state_dict(state_dict, 'blip2qformer.Qformer')
        ln_graph_dict = get_module_state_dict(state_dict, 'blip2qformer.ln_graph')
        qs_weight = get_module_state_dict(state_dict, 'blip2qformer.query_tokens')
        if use_qformer:
            load_ignore_unexpected(self.blip2opt.Qformer, qformer_dict)
            self.blip2opt.query_tokens.data.copy_(qs_weight)
        self.blip2opt.graph_encoder.load_state_dict(graph_encoder_dict)
        self.blip2opt.ln_graph.load_state_dict(ln_graph_dict)
        
        return self
    
    # def load_from_stage1_checkpoint(self, path):
    #     ckpt = torch.load(path, map_location='cpu')
    #     state_dict = ckpt['state_dict']
    #     state_dict = {k[13:]: v for k,v in state_dict.items()}
    #     load_ignore_mismatch(self.blip2opt, state_dict)
    #     return self
    
    def configure_optimizers(self):
        if self.args.optimizer == 'adafactor':
            print('Using adafactor optimizer')
            optimizer = Adafactor(
                self.parameters(),
                lr=1e-3,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False
            )
            self.scheduler = None
        else:
            self.trainer.fit_loop.setup_data()
            warmup_steps = min(len(self.trainer.train_dataloader), self.args.warmup_steps)
            optimizer = optim.AdamW(self.parameters(), lr=self.args.init_lr, weight_decay=self.args.weight_decay)
            if self.args.scheduler == 'linear_warmup_cosine_lr':
                self.scheduler = LinearWarmupCosineLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, warmup_steps, self.args.warmup_lr)
            elif self.args.scheduler == 'linear_warmup_step_lr':
                self.scheduler = LinearWarmupStepLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, self.args.lr_decay_rate, self.args.warmup_lr, warmup_steps)
            elif self.args.scheduler == 'None':
                self.scheduler = None
            else:
                raise NotImplementedError()
        return optimizer

    def on_test_epoch_end(self, outputs):
        list_predictions, list_targets, list_netlist_ids = zip(*outputs)
        predictions = [i for ii in list_predictions for i in ii]
        targets = [i for ii in list_targets for i in ii]
        netlist_ids = [i for ii in list_netlist_ids for i in ii]

        all_predictions = [None for _ in range(self.trainer.world_size)]
        all_targets = [None for _ in range(self.trainer.world_size)]
        all_netlist_ids = [None for _ in range(self.trainer.world_size)]
        
        dist.all_gather_object(all_predictions, predictions)
        dist.all_gather_object(all_targets, targets)
        dist.all_gather_object(all_netlist_ids, netlist_ids)
        if self.global_rank == 0:
            all_predictions = [i for ii in all_predictions for i in ii]
            all_targets = [i for ii in all_targets for i in ii]
            self.save_predictions(all_predictions, all_targets, all_netlist_ids, str(self.current_epoch) + '_test_')
            ## fixme: I am not sure if the max length is the same as previous experiments
            if self.args.task == 'func_desc':
                bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score = \
                    caption_evaluate(all_predictions, all_targets, self.blip2opt.llm_tokenizer, self.max_len * 2) 
                self.log("bleu2", bleu2, sync_dist=False)
                self.log("bleu4", bleu4, sync_dist=False)
                self.log("rouge_1", rouge_1, sync_dist=False)
                self.log("rouge_2", rouge_2, sync_dist=False)
                self.log("rouge_l", rouge_l, sync_dist=False)
                self.log("meteor_score", meteor_score, sync_dist=False)

    def save_predictions(self, predictions, all_netlist_ids, targets,name):
        assert len(predictions) == len(targets)
        assert len(predictions) == len(all_netlist_ids)
        # mkdir if args.file_names does not exist
        if not os.path.exists("./predictions"):
            os.mkdir("./predictions")
        with open(os.path.join("./predictions", name + '_' + self.args.filename + '.txt'), 'w', encoding='utf8') as f:
            for p, id, t in zip(predictions, all_netlist_ids, targets):
                line = {'netlist_id':id, 'prediction': p, 'target': t}
                f.write(json.dumps(line, ensure_ascii=True) + '\n')

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        graphs, prompt_tokens, text_tokens, texts = batch
        batch_size = text_tokens.input_ids.shape[0]
        loss = self.blip2opt(batch)
        # print(f"batch_size: {batch_size}, time: {timeit.default_timer() - start_time}")
        ###============== Overall Loss ===================###
        self.log("test graph loss (perplexity)", float(loss['loss']), batch_size=batch_size, sync_dist=True)
        ###============== Captioning Results ===================###
        
        samples = {'graphs': graphs, 'prompt_tokens': prompt_tokens}
        if self.args.task == 'func_desc':
            
            predictions = self.blip2opt.generate(
                samples, 
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len
            )
        elif self.args.task == 'type_pred':
            predictions = self.blip2opt.generate_from_candidates(
                samples, 
                self.candidate_tokens,
                self.candidates
            )
        return predictions, texts, graphs.netlist_id.cpu().tolist()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        start_time = timeit.default_timer()
        graphs, prompt_tokens, text_tokens, texts = batch
        batch_size = text_tokens.input_ids.shape[0]
        loss = self.blip2opt(batch)
        # print(f"batch_size: {batch_size}, time: {timeit.default_timer() - start_time}")
        ###============== Overall Loss ===================###
        self.log("val graph loss (perplexity)", float(loss['loss']), batch_size=batch_size, sync_dist=True)
        
        if (self.current_epoch+1) % self.caption_eval_epoch != 0:
            return loss['loss']
        start_time = timeit.default_timer()
        ###============== Captioning Results ===================###
        samples = {'graphs': graphs, 'prompt_tokens': prompt_tokens}
        if self.args.task == 'func_desc':
            predictions = self.blip2opt.generate(
                samples, 
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len
            )
        elif self.args.task == 'type_pred':
            predictions = self.blip2opt.generate_from_candidates(
                samples, 
                self.candidate_tokens,
                self.candidates
            )
        self.list_predictions.append(predictions)
        self.list_targets.append(texts)

        self.list_netlist_ids.append(graphs.netlist_id.cpu().tolist())
        # print(f"caption time: {timeit.default_timer() - start_time}")
    
    def on_validation_epoch_start(self) -> None:
        self.list_predictions = []
        self.list_targets = []
        self.list_netlist_ids = []
    
    def on_validation_epoch_end(self) -> None:
    # def validation_epoch_end(self, outputs):
        if (self.current_epoch+1) % self.caption_eval_epoch != 0:
            return 
        # caption_outputs = outputs[1]
        # list_predictions, list_targets = zip(*caption_outputs)
        list_predictions = self.list_predictions
        list_targets = self.list_targets
        list_netlist_ids = self.list_netlist_ids
        predictions = [i for ii in list_predictions for i in ii]
        targets = [i for ii in list_targets for i in ii]
        netlist_ids = [i for ii in list_netlist_ids for i in ii]

        all_predictions = [None for _ in range(self.trainer.world_size)]
        all_targets = [None for _ in range(self.trainer.world_size)]
        all_netlist_ids = [None for _ in range(self.trainer.world_size)]
        try:
            dist.all_gather_object(all_predictions, predictions)
            dist.all_gather_object(all_targets, targets)
            dist.all_gather_object(all_netlist_ids, netlist_ids)
        except RuntimeError:
            all_predictions = [predictions]
            all_targets = [targets]
            all_netlist_ids = [netlist_ids]
        if self.global_rank == 0:
            all_predictions = [i for ii in all_predictions for i in ii]
            all_targets = [i for ii in all_targets for i in ii]
            all_netlist_ids = [i for ii in all_netlist_ids for i in ii]
            self.save_predictions(all_predictions,all_netlist_ids, all_targets ,  str(self.current_epoch) + '_val_')
            ## fixme: I am not sure if the max length is the same as previous experiments
            if self.args.task == 'func_desc':
                bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score = \
                    caption_evaluate(all_predictions, all_targets, self.blip2opt.llm_tokenizer, self.max_len * 2) 
                self.log("bleu2", bleu2, sync_dist=False)
                self.log("bleu4", bleu4, sync_dist=False)
                self.log("rouge_1", rouge_1, sync_dist=False)
                self.log("rouge_2", rouge_2, sync_dist=False)
                self.log("rouge_l", rouge_l, sync_dist=False)
                self.log("meteor_score", meteor_score, sync_dist=False)
        

    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)
        if isinstance(batch, list) and len(batch) == 2:
            molecule_batch, reaction_batch = batch
            batch_size = molecule_batch[-1].size(0)
            ###============== molecule Loss ===================###
            molecule_loss = self.blip2opt(molecule_batch)['loss']
            self.log("stage1 loss", float(molecule_loss), batch_size=batch_size, sync_dist=True)
            
            ###============== reaction Loss ===================###
            reaction_loss = self.blip2opt.forward_reaction(reaction_batch)['loss']
            self.log("reaction loss", float(reaction_loss), batch_size=batch_size, sync_dist=True)

            self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)
            return molecule_loss + self.reaction_weight * reaction_loss
        else:
            batch_size = batch[-2].input_ids.size(0)
            ###============== Overall Loss ===================###
            loss = self.blip2opt(batch)
            self.log("total loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)
            self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)
            return loss['loss']

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GINSimclr")
        # train mode
        # GIN
        parser.add_argument('--gin_hidden_dim', type=int, default=512)
        parser.add_argument('--gin_num_layers', type=int, default=5)
        parser.add_argument('--drop_ratio', type=float, default=0.0)
        parser.add_argument('--tune_gnn', action='store_true', default=False)
        # Bert
        parser.add_argument('--bert_name', type=str, default='scibert')
        parser.add_argument('--cross_attention_freq', type=int, default=2)
        parser.add_argument('--num_query_token', type=int, default=8)
        # OPT

        parser.add_argument('--opt_model', type=str, default="meta-llama/Llama-3.2-3B-Instruct", help='LLM name')
        # parser.add_argument('--prompt', type=str, default='a molecule of ')
        parser.add_argument('--num_beams', type=int, default=2)
        parser.add_argument('--do_sample', action='store_true', default=False)
        parser.add_argument('--max_len', type=int, default=256, help = "used in generation")
        parser.add_argument('--min_len', type=int, default=2)
        parser.add_argument('--llm_tune', action='store_true', default=False, help = "tune llm using lora or not")
        parser.add_argument('--peft_config', type=str, default=None)
        parser.add_argument('--peft_dir', type=str, default='')

        
        parser.add_argument('--save_every_n_epochs', type=int, default=1)

        ## lora config
        parser.add_argument('--lora_r', type=int, default=8)
        parser.add_argument('--lora_alpha', type=int, default=32)
        parser.add_argument('--lora_dropout', type=int, default=0.1)

        # optimization
        parser.add_argument('--reaction_weight', type=float, default=1.0, help = "deprecated in our project (for reaction)")
        parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
        parser.add_argument('--init_lr', type=float, default=1e-4, help='optimizer init learning rate')
        parser.add_argument('--min_lr', type=float, default=5e-5, help='optimizer min learning rate')
        parser.add_argument('--warmup_lr', type=float, default=1e-6, help='optimizer warmup learning rate')
        parser.add_argument('--warmup_steps', type=int, default=1000, help='optimizer warmup steps')
        parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='optimizer lr decay rate')
        parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr', help='type of scheduler') # or linear_warmup_step_lr
        parser.add_argument('--optimizer', type=str, default='adamw', help='type of scheduler')
        parser.add_argument('--stage1_path', type=str, default='')
        parser.add_argument('--stage2_path', type=str, default='')
        parser.add_argument('--init_checkpoint', type=str, default='')
        parser.add_argument('--caption_eval_epoch', type=int, default=1)
        # load_in_4bit, default is False
        parser.add_argument('--load_in_4bit', action='store_true', default=False)
        return parent_parser
