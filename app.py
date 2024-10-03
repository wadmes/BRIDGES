import torch
import gradio as gr
# from model.blip2_opt import Blip2OPT
from stage2 import get_args
from model.blip2_stage2 import Blip2Stage2
from model.blip2_opt import smiles2data
from torch_geometric.loader.dataloader import Collater
from data_provider.stage2_dm import smiles_handler
from rdkit import Chem
from rdkit.Chem import Draw
from VLSI_util.data import cg2hetedata
import circuitgraph as cg
from torch_geometric.data import Batch
torch.set_default_dtype(torch.float16)

@torch.no_grad()
def netlist_caption(netlist_path, prompt, temperature):
    # temperature /= 100
    bbs = []
    circuit = cg.from_file(netlist_path,blackboxes=bbs)
    graphs = []
    graphs.append(cg2hetedata(circuit))
    graph_batch = Batch.from_data_list(graphs).to(args.devices)     

    prompt = prompt.format('<graph>' * 8) # 8 is the number of queries 
    molca.opt_tokenizer.padding_side = 'left'
    prompt_batch = molca.opt_tokenizer([prompt,],
                                       truncation=False,
                                       padding='longest',
                                       add_special_tokens=True,
                                       return_tensors='pt',
                                       return_attention_mask=True).to(args.devices)
    is_mol_token = prompt_batch.input_ids == molca.opt_tokenizer.mol_token_id
    prompt_batch['is_mol_token'] = is_mol_token
    
    samples = {'graphs': graph_batch, 'prompt_tokens': prompt_batch}
    
    ## generate results
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        text = molca.generate(samples, temperature=temperature, max_length=256, num_beams=5, do_sample=False)[0]
    return text



if __name__ == '__main__':


    args = get_args()
    args.devices = f'cuda:{args.devices}'
    args.test_ui = False


    if not args.test_ui:
        # load model
        collater = Collater([], [])
        model = Blip2Stage2.load_from_checkpoint(args.init_checkpoint, strict=False, args=args)
        molca = model.blip2opt

        del model
        molca = molca.half().eval().to(args.devices)
        # text = netlist_caption('./netlist_data/arith/adder_4_bit.v', 'The logic netlist graph tokens are {}. Please describe the logic netlist graph.', 1)
        # print(text)
        # exit()
    with gr.Blocks() as demo:
        gr.HTML(
        """
        <center><h1><b>VLSI LLM</b></h1></center>
        <p style="font-size:20px; font-weight:bold;">This is the demo page of <i>Graph-based LLM for VLSI</i></p>
        <p style="font-size:20px; font-weight:bold;"> You can input one netlist below, and we will generate the netlist's text descriptions. </p>
        """)
        
        paths = gr.Textbox(placeholder="Input one netlist path", label='Input netlist path')
        ## list of examples
        example_list = ['./netlist_data/arith/adder_4_bit.v', './netlist_data/arith/adder_8_bit.v']
        gr.Examples(example_list, [paths,], fn=netlist_caption, label='Example net-list path')

        prompt = gr.Textbox(placeholder="Customized your own prompt. Note this can give unpredictable results given our model was not pretrained for other prompts.", label='Customized prompt (Default to None)', value='')
        temperature = gr.Slider(0.1, 1, value=1, label='Temperature')
        btn = gr.Button("Submit")

        with gr.Row():
            out = gr.Textbox(label='Caption Output', placeholder='Netlist caption results')
        btn.click(fn=netlist_caption, inputs=[paths, prompt, temperature], outputs=[out])
    demo.launch(share=True)

