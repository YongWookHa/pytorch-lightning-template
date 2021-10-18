"""
pytorch lightning template for model implementation
"""

import torch
import pytorch_lightning as pl
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
from model import Model
from dataset import DataModule, custom_collate
from torch.utils.data import DataLoader, random_split

from utils import load_setting

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", "-s", type=str, default="settings/default.yaml",
                        help="Experiment settings")
    parser.add_argument("--version", "-v", type=int, default=0,
                        help="Train experiment version")
    parser.add_argument("--num_workers", "-nw", type=int, default=0,
                        help="Number of workers for dataloader")
    parser.add_argument("--resume_train", "-rt", type=str, default="",
                        help="Resume train from certain checkpoint")
    args = parser.parse_args()
    
    # setting
    cfg = load_setting(args.setting)
    cfg.update(vars(args))
    print("setting:", cfg)

    train_set = DataModule(cfg.train_data_path)
    val_set = DataModule(cfg.val_data_path)    

    collate = custom_collate()

    train_dataloader = DataLoader(train_set, batch_size=cfg.batch_size, 
                                  num_workers=cfg.num_workers, collate_fn=collate)
    valid_dataloader = DataLoader(val_set, batch_size=cfg.batch_size,
                                  num_workers=cfg.num_workers, collate_fn=collate)

    model = Model(cfg)
    
    if cfg.resume_train:
        model = model.load_from_checkpoint(cfg.resume_train)

    logger = TensorBoardLogger("tb_logs", name="model", version=cfg.version)
    
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        dirpath=f"checkpoints/version_{cfg.version}",
        filename="checkpoints-{epoch:02d}-{val_acc:.2f}",
        save_top_k=3,
        mode="max",
    )
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')

    device_cnt = torch.cuda.device_count()
    trainer = pl.Trainer(gpus=device_cnt, max_epochs=cfg.epochs,
                        logger=logger, num_sanity_val_steps=1,
                        accelerator="ddp" if device_cnt > 1 else None, 
                        callbacks=[ckpt_callback, lr_callback],
                        resume_from_checkpoint=cfg.resume_train if cfg.resume_train else None)

    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=valid_dataloader)
