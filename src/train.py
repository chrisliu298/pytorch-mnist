import argparse
import os

import wandb
from addict import Dict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from dataset import MNISTDataModule
from model import FCN


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument("--data_dir", type=str, default="./")
    # Model
    parser.add_argument("--layer_dims", type=str, default="784x512x256x128x10")
    # Training
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adam", "sgd"]
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", default=0)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "gpu"]
    )
    # Experiment
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=str, default="mnist")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    cfg = parser.parse_args()
    cfg = Dict(vars(cfg))
    # Auto num_workers
    if cfg.num_workers == "all":
        cfg.num_workers = os.cpu_count()
    return cfg


def main():
    """Main function."""
    cfg = parse_args()
    seed_everything(cfg.seed)
    # Data
    datamodule = MNISTDataModule(
        data_dir=cfg.data_dir, batch_size=cfg.batch_size, num_workers=cfg.num_workers
    )
    # Model
    model = FCN(cfg)
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc_epoch",
        filename="{epoch:02d}-{val_acc_epoch:.4f}",
        mode="max",
        save_top_k=10,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stopping = EarlyStopping(monitor="val_acc_epoch", patience=10, mode="max")
    callbacks = [checkpoint_callback, lr_monitor, early_stopping]
    # Logger
    wandb_logger = WandbLogger(
        offline=not cfg.wandb, project=cfg.project, config=cfg, entity="chrisliu298"
    )
    # Trainer
    trainer = Trainer(
        accelerator=cfg.device,
        devices=-1,
        max_epochs=cfg.max_epochs,
        callbacks=callbacks,
        logger=wandb_logger,
        profiler="simple",
        check_val_every_n_epoch=1,
        benchmark=True,
        enable_progress_bar=cfg.verbose,
    )
    # Train
    trainer.fit(model, datamodule=datamodule)
    # Validate
    trainer.validate(ckpt_path="best", datamodule=datamodule, verbose=cfg.verbose)
    # Test
    trainer.test(ckpt_path="best", datamodule=datamodule, verbose=cfg.verbose)
    wandb.finish(quiet=not cfg.verbose)


if __name__ == "__main__":
    main()
