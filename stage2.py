import os
import torch
import argparse
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import WandbLogger
from model.blip2_stage2 import Blip2Stage2
from VLSI_util.stage2_data import Stage2Netlist
from VLSI_util.data import netlistDataset
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# torch.set_default_dtype(torch.float16)

os.environ['OPENBLAS_NUM_THREADS'] = '1'
## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
## for A5000 gpus
# torch.set_float32_matmul_precision('medium') # can be medium (bfloat16), high (tensorfloat32), highest (float32)
# torch.set_default_dtype(torch.bfloat16)
# strategy = strategies.DDPStrategy(find_unused_parameters=find_unused_parameters, start_method='spawn')
# class MyDDPSpawnStrategy(strategies.DDPSpawnStrategy):
#     def load_model_state_dict(self, checkpoint):
#         assert self.lightning_module is not None
#         self.lightning_module.load_state_dict(checkpoint["state_dict"], strict=False)

class MyDDPStrategy(strategies.DDPStrategy):
    def load_model_state_dict(self, checkpoint):
        assert self.lightning_module is not None
        self.lightning_module.load_state_dict(checkpoint["state_dict"], strict=False)

def main(args):
    pl.seed_everything(args.seed)
    # model
    if args.init_checkpoint:
        model = Blip2Stage2.load_from_checkpoint(args.init_checkpoint, strict=False, args=args)
        print(f"loaded init checkpoint from {args.init_checkpoint}")
    elif args.stage2_path:
        model = Blip2Stage2(args)
        ckpt = torch.load(args.stage2_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        print(f"loaded stage2 model from {args.stage2_path}")
    elif args.stage1_path:
        model = Blip2Stage2(args)
        model.load_from_stage1_checkpoint(args.stage1_path)
        print(f"loaded stage1 model from {args.stage1_path}")
    else:
        model = Blip2Stage2(args)

    print('total params:', sum(p.numel() for p in model.parameters()))

    if args.opt_model.find('galactica') >= 0 or args.opt_model.find('t5') >= 0:
        tokenizer = model.blip2opt.opt_tokenizer
    elif args.opt_model.find('llama') >= 0 or args.opt_model.find('vicuna') >= 0:
        tokenizer = model.blip2opt.llm_tokenizer
    else:
        raise NotImplementedError
    # data
    dm = Stage2Netlist(args.mode, args.num_workers, args.batch_size, args.text_max_len, tokenizer, args)
    callbacks = []
    callbacks.append(EarlyStopping(monitor='val graph loss (perplexity)', mode = 'min', patience=2))
    
    ## fixme save only used parameters
    # callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/", every_n_epochs=10, save_top_k=-1))
    callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/", 
                                         filename='{epoch:02d}', 
                                         every_n_epochs=args.save_every_n_epochs, 
                                         save_last=True, 
                                         save_top_k=-1,
                                         save_on_train_epoch_end=True))
    logger = WandbLogger(project='stage-2-' + args.task, dir='./wandb-log', name = args.filename)
    trainer = Trainer(fast_dev_run = False,precision=args.precision, max_epochs=args.max_epochs, val_check_interval=args.val_check_interval, callbacks=callbacks, logger=logger, strategy=DDPStrategy(find_unused_parameters=True, static_graph=True))
    if args.mode in {'pretrain', 'ft'}:
        trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)
    elif args.mode == 'eval':
        trainer.fit_loop.epoch_progress.current.completed = args.caption_eval_epoch - 1
        trainer.validate(model, datamodule=dm)
    else:
        raise NotImplementedError()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default="stage2-v1")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # MM settings
    parser.add_argument('--mode', type=str, default='pretrain')
    parser.add_argument('--strategy_name', type=str, default='auto')
    parser.add_argument('--ckpt_path', type=str, default=None)
    # parser = Trainer.add_argparse_args(parser)
    parser = Blip2Stage2.add_model_specific_args(parser)  # add model args
    parser = Stage2Netlist.add_model_specific_args(parser)
    parser.add_argument('--precision', type=str, default='bf16-mixed', help= "the precision argument for the trainer, could be bf16-mixed, transformer-engine, for details, refer to https://lightning.ai/docs/pytorch/2.4.0/common/trainer.html#precision")
    parser.add_argument('--max_epochs', type=int, default=6)
    parser.add_argument('--val_check_interval', type=float, default=1.0)
    args = parser.parse_args()

    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    return args

if __name__ == '__main__':
    main(get_args())

