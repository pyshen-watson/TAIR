import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from argparse import ArgumentParser

from TAIR.utils.logger import setup_logger
from TAIR.dataset import DataModuleConfig, DataModule
from TAIR.model.task_model.classifier import ClassifierConfig, create_classifier

torch.set_float32_matmul_precision('high')

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log", action="store_true")

    # Dataset
    parser.add_argument("--dataset_name", type=str, default="CUB200", choices=["CUB200"])
    parser.add_argument("--dataset_dir", type=str, default="data/CUB_200_2011")
    parser.add_argument("--degraded_type", type=str, default="fog", choices=["fog", "rain", "snow", "cloud"])
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=12)

    # Trainer
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--devices", type=str, default="auto", choices=["auto", "0", "1"])
    parser.add_argument("--patience", type=int, default=3)

    # Experiment
    parser.add_argument("--exp_name", type=str, default="classifier")
    parser.add_argument("--model_name", type=str, default="RN18", choices=["RN18", "RN50"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--degraded", "-D", action="store_true")

    return parser.parse_args()


def get_dataModule(args):

    config = DataModuleConfig(
        ds_name=args.dataset_name,
        ds_dir=args.dataset_dir,
        dg_type=args.degraded_type,
        size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    return DataModule(config)


def get_model(args, n_classes, log_dir):
    config = ClassifierConfig(exp_name=args.exp_name, lr=args.lr, wd=args.wd, degraded=args.degraded, log_dir=log_dir, hp=vars(args))
    return create_classifier(args.model_name, n_classes, config)


def get_trainer(args):

    devices = [int(args.devices)] if args.devices != "auto" else args.devices
    ckpt_name = "{epoch}-{step}-{val_loss:.4f}-{val_acc:.4f}"

    return pl.Trainer(
        accelerator="auto",
        devices=devices,
        benchmark=True,
        max_epochs=args.max_epochs,
        log_every_n_steps=1,
        logger=TensorBoardLogger("lightning_logs", name=args.exp_name),
        callbacks=[
            ModelCheckpoint(monitor="val_loss", filename=ckpt_name, save_top_k=1, mode="min"),
            EarlyStopping(monitor="val_loss", patience=args.patience, mode="min"),
        ],
    )


if __name__ == "__main__":

    args = get_args()
    setup_logger(export_log=args.log)
    pl.seed_everything(args.seed)

    dm = get_dataModule(args)
    trainer = get_trainer(args)
    model = get_model(args, dm.n_classes, trainer.logger.log_dir)
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())
    trainer.test(model, dm.test_dataloader())
