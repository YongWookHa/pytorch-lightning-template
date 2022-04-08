"""
pytorch lightning template for model implementation
"""

import torch
import pytorch_lightning as pl
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.core.saving import load_hparams_from_yaml, update_hparams
from model import Model
from dataset import DataModule, custom_collate
from torch.utils.data import DataLoader


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
    hparams = load_hparams_from_yaml(args.setting)
    update_hparams(hparams, vars(args))
    print("setting:", hparams)

    train_set = DataModule(hparams.train_data_path)
    val_set = DataModule(hparams.val_data_path)

    collate = custom_collate()

    train_dataloader = DataLoader(train_set, batch_size=hparams.batch_size,
                                  num_workers=hparams.num_workers, collate_fn=collate)
    valid_dataloader = DataLoader(val_set, batch_size=hparams.batch_size,
                                  num_workers=hparams.num_workers, collate_fn=collate)

    model = Model(hparams)

    if hparams.resume_train:
        model = model.load_from_checkpoint(hparams.resume_train)

    logger = TensorBoardLogger("tb_logs", name="model", version=hparams.version,
                               default_hp_metric=False)

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        dirpath=f"checkpoints/version_{hparams.version}",
        filename="checkpoints-{epoch:02d}-{val_acc:.2f}",
        save_top_k=3,
        mode="max",
    )
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')

    device_cnt = torch.cuda.device_count()
    trainer = pl.Trainer(gpus=device_cnt, max_epochs=hparams.epochs,
                        logger=logger, num_sanity_val_steps=1,
                        accelerator="ddp" if device_cnt > 1 else None,
                        callbacks=[ckpt_callback, lr_callback],
                        resume_from_checkpoint=hparams.resume_train if hparams.resume_train else None)

    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=valid_dataloader)
