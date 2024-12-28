import logging
from pathlib import Path
from functools import partial
from dataclasses import dataclass

import pytorch_lightning as pl
from torchvision import transforms as T
from torch.utils.data import DataLoader
from .dataset import CUB200

DG_TYPES = ["fog", "rain", "snow", "cloud"]
DATASET_DICT = {"CUB200": CUB200}


@dataclass
class DataModuleConfig:

    ds_name: str = "CUB200"
    ds_dir: str = "data/CUB_200_2011"
    dg_type: str = "fog"
    size: int = 224
    batch_size: int = 128
    num_workers: int = 12

    def __post_init__(self):

        self.root = Path(self.ds_dir)
        self.arg_check()

        self.train_t = T.Compose(
            [
                T.ToTensor(),
                T.RandomHorizontalFlip(),
                T.Resize((self.size, self.size)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.test_t = T.Compose(
            [
                T.ToTensor(),
                T.Resize((self.size, self.size)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def arg_check(self):

        if self.ds_name not in DATASET_DICT.keys():
            logging.error(f'The dataset "{self.ds_name}" is not supported.')
            exit()

        if not self.root.exists():
            logging.error(f'The root directory "{self.root}" does not exist.')
            exit()

        if self.dg_type not in DG_TYPES:
            logging.error(f'The degraded type "{self.dg_type}" is not supported.')
            exit()

        logging.info("DataModuleConfig: Verified.")


class DataModule(pl.LightningDataModule):

    def __init__(self, config: DataModuleConfig):
        super(DataModule, self).__init__()
        self.config = config
        self.prepare_data()
        self.create_loader = partial(DataLoader, batch_size=config.batch_size, num_workers=config.num_workers)

    def prepare_data(self):
        C = self.config
        dataset_class = DATASET_DICT[C.ds_name]
        dataset_class_ = partial(dataset_class, root=C.root, dg_type=C.dg_type)
        self.train_ds = dataset_class_(is_train=True, transform=C.train_t)
        self.test_ds = dataset_class_(is_train=False, transform=C.test_t)

    def train_dataloader(self):
        return self.create_loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self.create_loader(self.test_ds, shuffle=False)

    def test_dataloader(self):
        return self.create_loader(self.test_ds, shuffle=False)

    @property
    def n_classes(self):
        try:
            return self.train_ds.n_classes

        except AttributeError:
            logging.error(f'The dataset "{self.config.ds_name}" does not provide the attribute "n_classes".')
            exit()
