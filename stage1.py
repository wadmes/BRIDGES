import os
import argparse
import torch
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc 
from pytorch_lightning.loggers import WandbLogger
from model.blip2_stage1 import Blip2Stage1
from VLSI_util.data_module import Stage1DM_v2
from VLSI_util.data import netlistDataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
os.environ['OPENBLAS_NUM_THREADS'] = '1'
## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
## for A5000 gpus
# torch.set_float32_matmul_precision('medium') # can be medium (bfloat16), high (tensorfloat32), highest (float32)
# torch.set_default_dtype(torch.bfloat16)

def main(args):
    pl.seed_everything(args.seed)

    # model
    if args.init_checkpoint:
        model = Blip2Stage1.load_from_checkpoint(args.init_checkpoint, device=args.devices, args=args)
        print(f"loading model from {args.init_checkpoint}")
    else:
        model = Blip2Stage1(args)
    
    print('total params:', sum(p.numel() for p in model.parameters()))

    tokenizer = model.blip2qformer.tokenizer

    dm = Stage1DM_v2(args.num_workers, args.mix, args.dataset_path, args.text_max_len, tokenizer, args.seed, args)
    model.val_match_loader = dm.val_match_loader
    model.test_match_loader = dm.test_match_loader

    callbacks = []
    callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/", 
                                         filename='{epoch:02d}', 
                                         every_n_epochs=args.save_every_n_epochs, 
                                         save_top_k=-1))
    callbacks.append(EarlyStopping(monitor='val_fullset_t2g_rtlid_acc', mode = 'max', patience=3))
    
    # logger = CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/')
    logger = WandbLogger(project='LLM-graph-stage1-v3-1115', name = args.filename)
    trainer = Trainer(fast_dev_run = False,precision=args.precision, max_epochs=args.max_epochs, val_check_interval=args.val_check_interval, callbacks=callbacks, logger = logger)
    if args.mode == 'train':
        trainer.fit(model, datamodule=dm)
    elif args.mode == 'eval':
        trainer.fit_loop.epoch_progress.current.completed = 49 ## avoid 
        trainer.validate(model, datamodule=dm)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--filename', type=str, default="stage1_test")
    # GPU
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # MM settings
    parser.add_argument('--gtm', action='store_true', help='use graph-text matching or not', default=True)
    parser.add_argument('--lm', action='store_true', help='use language modeling or not', default=True)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--devices', type=str, default='0,1,2,3')
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    parser.add_argument('--max_epochs', type=int, default=11)
    parser.add_argument('--val_check_interval', type=float, default=1.0)
    # parser.add_argument('--save_every_n_epochs', type=int, default=1)
    # parser = Trainer.add_argparse_args(parser)
    parser = Blip2Stage1.add_model_specific_args(parser)  # add model args
    parser = Stage1DM_v2.add_model_specific_args(parser)
    # parser.set_defaults(accelerator='gpu',
    #                     devices='0,1,2,3',
    #                     precision='bf16',
    #                     max_epochs=50,
    #                     val_check_interval=1)
    args = parser.parse_args()
    
    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    main(args)

