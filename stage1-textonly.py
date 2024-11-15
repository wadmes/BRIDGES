import os
import argparse
import torch
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
# import DDPStrategy(find_unused_parameters=True, static_graph=True)
import pytorch_lightning.callbacks as plc 
from pytorch_lightning.loggers import WandbLogger
from model.blip2_stage1_textonly import Blip2Stage1
from pytorch_lightning.strategies import DDPStrategy
from VLSI_util.data_module_textonly import Stage1DM_v2
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
    
    # logger = CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/')
    logger = WandbLogger(project=args.filename)
    trainer = Trainer(fast_dev_run = False,precision=args.precision, max_epochs=args.max_epochs, val_check_interval=0.5, callbacks=callbacks, logger = logger,strategy=DDPStrategy(find_unused_parameters=True, static_graph=True))
    if args.mode == 'train':
        trainer.fit(model, datamodule=dm)
    elif args.mode == 'eval':
        trainer.fit_loop.epoch_progress.current.completed = 49 ## avoid 
        trainer.validate(model, datamodule=dm)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--filename', type=str, default="stage1_text_rtl_2048")
    # GPU
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # MM settings
    parser.add_argument('--gtm', action='store_true', help='use graph-text matching or not', default=False)
    parser.add_argument('--lm', action='store_true', help='use language modeling or not', default=False)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--devices', type=str, default='0,1,2,3')
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    # parser.add_argument('--save_every_n_epochs', type=int, default=1)
    # parser = Trainer.add_argparse_args(parser)
    parser = Blip2Stage1.add_model_specific_args(parser)  # add model args
    parser = Stage1DM_v2.add_model_specific_args(parser)
    # parser.set_defaults(accelerator='gpu',
    #                     devices='0,1,2,3',
    #                     precision='bf16',
    #                     max_epochs=50,
    #                     check_val_every_n_epoch=1)
    args = parser.parse_args()
    
    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    main(args)

