import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import logging
from pathlib import Path
from functools import partial
from dataclasses import dataclass
from .classifier import Classifier

@dataclass
class ClassifierConfig:
    exp_name: str
    lr: float = 1e-4
    wd: float = 1e-4
    degraded: bool = False
    log_dir: str = "logs"
    hp: dict = None

class ClassifierWrapper(pl.LightningModule):
    
    def __init__(self, classifier: Classifier, config: ClassifierConfig):
        super(ClassifierWrapper, self).__init__()
        
        # Model
        self.model = classifier
        self.accuracies = self.model.accuracies
        
        # Config
        self.config = config
        self.save_hyperparameters(config.hp)
        self.logging = partial(self.log, sync_dist=True, prog_bar=True, on_step=False, on_epoch=True)
        self.log_dir = Path(config.log_dir)

    def configure_optimizers(self):
        C = self.config
        return torch.optim.Adam(self.parameters(), lr=C.lr, weight_decay=C.wd)

    def forward(self, img):
        return self.model(img)

    def calc_loss(self, batch, split: str):

        # Decide the image to use
        img_c, img_d, label = batch
        img = img_d if self.config.degraded else img_c

        # Forward
        logit = self(img)
        loss = F.cross_entropy(logit, label)
        pred = torch.argmax(logit, dim=1)
        acc = self.accuracies[split](pred, label)

        # Log
        self.logging(f"{split}_loss", loss)
        self.logging(f"{split}_acc", acc)
        return loss

    def training_step(self, batch, batch_idx):
        return self.calc_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.calc_loss(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.calc_loss(batch, "test")
    
    def on_save_checkpoint(self, checkpoint):
        if self.trainer.is_global_zero:
            save_path = self.log_dir / f"{self.config.exp_name}.pt"
            torch.save(self.model.state_dict(), save_path)
            logging.info(f"Save the checkpoint at {save_path}.")