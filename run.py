"""
im2latex pytorch lightning ported version

model will be changed little bit more
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
    parser.add_argument("--setting", "-s", type=str, default="settings/debug.yaml",
                        help="Experiment settings")
    parser.add_argument("--version", "-v", type=int, default=0,
                        help="Train experiment version")
    parser.add_argument("--max_epochs", "-me", type=int, default=10,
                        help="Max epochs for training")
    parser.add_argument("--num_workers", "-nw", type=int, default=0,
                        help="Number of workers for dataloader")
    parser.add_argument("--batch_size", "-bs", type=int,
                        help="Batch size for training and validate")
    parser.add_argument("--resume_train", "-rt", type=str, default="",
                        help="Resume train from certain checkpoint")
    parser.add_argument("--split_data", "-sd", nargs='+', type=float, default=None,
                        help="Split total data into train and validate data (train, val)")
    args = parser.parse_args()
    
    # setting
    cfg = load_setting(args.setting)
    cfg.update(vars(args))
    print("setting:", cfg)

    if cfg.split_data:
        train_set, val_set = random_split(DataModule(cfg.total_data_path),
                                          tuple(cfg.split_data))
    else:
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

    trainer = pl.Trainer(gpus=torch.cuda.device_count(), max_epochs=cfg.max_epochs,
                        logger=logger, num_sanity_val_steps=1, accelerator="ddp", 
                        callbacks=[ckpt_callback, ], resume_from_checkpoint=cfg.resume_train)

    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=valid_dataloader)

    trainer = pl.Trainer()
