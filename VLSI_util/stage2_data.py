# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
from pytorch_lightning import LightningDataModule
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader, Dataset
from torch_geometric.loader.dataloader import Collater
from torch_geometric.data import Batch
from VLSI_util.data import stage1dataset


"""
task: str, one of ['func_desc','type_pred']
add_file: str, one of ['', 'netlist', 'rtl']
TrainCollater returns tokenized ids
"""
class TrainCollater:
    def __init__(self, tokenizer, text_max_len, graph_ph, graph_token_id, prompt, task, add_file):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.graph_ph = graph_ph
        self.graph_token_id = graph_token_id
        self.prompt = prompt
        self.task = task
        self.add_file = add_file
        
    def __call__(self, batch):
        graphs = batch
    
        graph_batch = Batch.from_data_list(graphs)     
        
        self.tokenizer.paddding_side = 'left' # Reason for left padding here: pad pad pad prompt: geration result pad pad pad
        if self.add_file == '':
            prompt = [self.prompt.format(self.graph_ph)] * len(graphs)
        elif self.add_file == 'netlist':
            prompt = []
            for graph in graphs:
                prompt.append(self.prompt.format(graph.netlist[:512], self.graph_ph))
        elif self.add_file == 'rtl':
            prompt = []
            for graph in graphs:
                prompt.append(self.prompt.format(graph.rtl[:512], self.graph_ph))

        prompt_tokens = self.tokenizer(text=prompt, 
                                              truncation=True,
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
        if self.task == 'func_desc':
            text = graph_batch.text
        elif self.task == 'type_pred':
            text = graph_batch.consistent_label
            for t in text:
                t = t.replace('Units', 'Unit')
        else:
            raise NotImplementedError
        
        # add tokenizer.eos_token to each end of text
        text = [f"{t}<|end_of_text|>" for t in text]

        text_tokens = self.tokenizer(text=text,
                                     truncation=True,
                                     padding='longest',
                                     add_special_tokens=False,
                                     max_length=self.text_max_len,
                                     return_tensors='pt',
                                     return_attention_mask=True)
        return graph_batch, prompt_tokens, text_tokens, text

    
    


class Stage2Netlist(LightningDataModule):
    def __init__(
        self,
        mode: str = 'pretrain',
        num_workers: int = 0,
        batch_size: int = 256,
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
        if args.task == 'func_desc':
            self.text_max_len = 512
            if args.add_file == '':
                self.prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a hardware description expert. Provide a single, coherent technical paragraph describing the functionality of a Verilog module.
Constraints:
- Use complete English sentences.
- Avoid mentioning variable names or including any Verilog syntax.
- Ensure the description focuses on functionality, not implementation details.
- Do not use lists, bullet points, or code snippets.
- Maintain a logical flow without line breaks or special formatting.
            
Example:
---
**Module Description:**
This module implements an edge detection mechanism. It accepts an 8-bit binary input and a clock signal, 
producing an 8-bit output that reflects the input value one cycle after an edge is detected. 
The circuit operates by comparing the current input with the previous input to identify edges, utilizing a counter to manage the delay in output generation.
---
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Provide a detailed description of the following Verilog module. Its graph tokens are {}.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
            elif args.add_file == 'netlist':
                self.prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a hardware description expert. Provide a single, coherent technical paragraph describing the functionality of a Verilog module.
Constraints:
- Use complete English sentences.
- Avoid mentioning variable names or including any Verilog syntax.
- Ensure the description focuses on functionality, not implementation details.
- Do not use lists, bullet points, or code snippets.
- Maintain a logical flow without line breaks or special formatting.
            
Example:
---
**Module Description:**
This module implements an edge detection mechanism. It accepts an 8-bit binary input and a clock signal, 
producing an 8-bit output that reflects the input value one cycle after an edge is detected. 
The circuit operates by comparing the current input with the previous input to identify edges, utilizing a counter to manage the delay in output generation.
---
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Provide a detailed description of the following Verilog module. Its verilog code is {}. Its graph representations are {}.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
            elif args.add_file == 'rtl':
                self.prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a hardware description expert. Provide a single, coherent technical paragraph describing the functionality of a Verilog module.
Constraints:
- Use complete English sentences.
- Avoid mentioning variable names or including any Verilog syntax.
- Ensure the description focuses on functionality, not implementation details.
- Do not use lists, bullet points, or code snippets.
- Maintain a logical flow without line breaks or special formatting.
            
Example:
---
**Module Description:**
This module implements an edge detection mechanism. It accepts an 8-bit binary input and a clock signal, 
producing an 8-bit output that reflects the input value one cycle after an edge is detected. 
The circuit operates by comparing the current input with the previous input to identify edges, utilizing a counter to manage the delay in output generation.
---
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Provide a detailed description of the following Verilog module. Its RTL code is {}. Its graph representations are {}.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
        elif args.task == 'type_pred':
            self.text_max_len = 228
            if args.add_file == '':
                self.prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a specialized Verilog code analyzer focused on classifying hardware designs into specific categories. Your task is to analyze Verilog code and determine its primary design type from the following categories:
Encryption Unit: Designs implementing cryptographic algorithms, secure hash functions, or other security-related operations
Data Path Unit: Components handling data flow, multiplexers, decoders, registers, and data routing
Control Logic Unit: State machines, sequence controllers, and decision-making logic
Arithmetic Unit: Mathematical operations, ALUs, multipliers, dividers, and computational blocks
Communication Protocol Unit: Implementations of protocols like UART, I2C, SPI, or other communication interfaces
Signal Processing Unit: Filters, FFT implementations, signal conditioning, and digital signal processing
Clock Management Unit: Clock generators, PLL implementations, clock dividers, and timing control
Others: Designs that don't clearly fit into the above categories
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Please analyze the following Verilog graph and classify it into one of the specified design types. Its graph tokens are {}.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
            
            elif args.add_file == 'netlist':
                self.prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a specialized Verilog code analyzer focused on classifying hardware designs into specific categories. Your task is to analyze Verilog code and determine its primary design type from the following categories:
Encryption Unit: Designs implementing cryptographic algorithms, secure hash functions, or other security-related operations
Data Path Unit: Components handling data flow, multiplexers, decoders, registers, and data routing
Control Logic Unit: State machines, sequence controllers, and decision-making logic
Arithmetic Unit: Mathematical operations, ALUs, multipliers, dividers, and computational blocks
Communication Protocol Unit: Implementations of protocols like UART, I2C, SPI, or other communication interfaces
Signal Processing Unit: Filters, FFT implementations, signal conditioning, and digital signal processing
Clock Management Unit: Clock generators, PLL implementations, clock dividers, and timing control
Others: Designs that don't clearly fit into the above categories
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Please analyze the following Verilog graph and classify it into one of the specified design types. Its verilog code is {}. Its graph tokens are {}.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
            elif args.add_file == 'rtl':
                self.prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a specialized Verilog code analyzer focused on classifying hardware designs into specific categories. Your task is to analyze Verilog code and determine its primary design type from the following categories:
Encryption Unit: Designs implementing cryptographic algorithms, secure hash functions, or other security-related operations
Data Path Unit: Components handling data flow, multiplexers, decoders, registers, and data routing
Control Logic Unit: State machines, sequence controllers, and decision-making logic
Arithmetic Unit: Mathematical operations, ALUs, multipliers, dividers, and computational blocks
Communication Protocol Unit: Implementations of protocols like UART, I2C, SPI, or other communication interfaces
Signal Processing Unit: Filters, FFT implementations, signal conditioning, and digital signal processing
Clock Management Unit: Clock generators, PLL implementations, clock dividers, and timing control
Others: Designs that don't clearly fit into the above categories
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Please analyze the following Verilog graph and classify it into one of the specified design types. Its RTL code is {}. Its graph tokens are {}.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
        
        train_graphs = []
        val_graphs = []
        test_graphs = []
        max_rtlid = 0
        for dataset_idx, ds_path in enumerate(args.dataset_path):
            
            this_train = torch.load(ds_path.replace('.pt', '_train.pt'))
            this_val = torch.load(ds_path.replace('.pt', '_val.pt'))
            this_test = torch.load(ds_path.replace('.pt', '_test.pt'))
            if args.task == 'type_pred':
                # remove graph with empty consistent_label
                for graphs in [this_train, this_val, this_test]:
                    graphs = [graph for graph in graphs if len(graph.consistent_label) > 0]
                    
            for graphs in [this_train, this_val, this_test]:
                for graph in graphs:
                    graph.netlist_id += 1000000 * dataset_idx # hardcode to avoid netlist_id conflict
                    graph.rtl_id += 1000000 * dataset_idx
            
            train_graphs.extend(this_train)
            val_graphs.extend(this_val)
            test_graphs.extend(this_test)
        print(f"Use MIXED DATASETS! train: {len(train_graphs)}, val: {len(val_graphs)}, test: {len(test_graphs)}")
        self.train_dataset = stage1dataset(train_graphs)
        self.val_dataset = stage1dataset(val_graphs)
        self.test_dataset = stage1dataset(test_graphs)

        self.init_tokenizer(tokenizer)
        self.graph_ph_token = '<graph>' * self.args.num_query_token # ph is short for placeholder
        
    
    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        # self.pretrain_dataset.tokenizer = tokenizer
        self.train_dataset.tokenizer = tokenizer
        self.val_dataset.tokenizer = tokenizer
        self.test_dataset.tokenizer = tokenizer
        self.tokenizer.graph_token_id = tokenizer("<graph>", add_special_tokens=False).input_ids[0]
        self.graph_token_id = self.tokenizer.graph_token_id

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True,
            collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.graph_ph_token, self.graph_token_id, self.prompt, self.args.task, self.args.add_file),
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
            collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.graph_ph_token, self.graph_token_id, self.prompt, self.args.task, self.args.add_file),
        )
        return val_loader
    
    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.graph_ph_token, self.graph_token_id, self.prompt, self.args.task, self.args.add_file),
        )
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--inference_batch_size', type=int, default=8)
        parser.add_argument("--dataset_path", action="extend", nargs="+", type=str, default=["/scratch/weili3/RTLCoder26532.pt","/scratch/weili3/MGVerilog11144.pt"])
        parser.add_argument('--text_max_len', type=int, default=256)
        parser.add_argument('--prompt', type=str, default='The graph of this module is [START_NETLIST_GRAPH]{}[END__NETLIST_GRAPH].')
        # task, default is 'type_pred'
        parser.add_argument('--task', type=str, default='type_pred', help='one of [func_desc, type_pred]')
        # add_file, default is ''
        parser.add_argument('--add_file', type=str, default='', help='add file options to the prompt, can be empty, `netlist`, `rtl`')
        return parent_parser
    