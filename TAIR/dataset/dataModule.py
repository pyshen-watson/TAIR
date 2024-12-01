import pytorch_lightning as pl
from torchvision import transforms as T
from torch.utils.data import DataLoader
from dataclasses import dataclass
from functools import partial

from .dataset import CUB200


@dataclass
class DataModule(pl.LightningDataModule):

    name: str = "CUB200"
    root: str = "data/CUB_200_2011"
    dg_type: str = "fog"
    size: int = 224
    batch_size: int = 32
    num_workers: int = 12
    
    def __post_init__(self):
        
        self.train_t = T.Compose([
            T.ToTensor(),
            T.RandomHorizontalFlip(),
            T.Resize((self.size, self.size)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.test_t = T.Compose([
            T.ToTensor(),
            T.Resize((self.size, self.size)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.create_loader = partial(DataLoader, batch_size=self.batch_size, num_workers=self.num_workers)
        self.prepare_data()
        
    def prepare_data(self) -> None:
        
        if self.name == "CUB200":
            CUB200_ = partial(CUB200, root=self.root, dg_type=self.dg_type)
            self.train_ds = CUB200_(is_train=True, transform=self.train_t)
            self.test_ds = CUB200_(is_train=False, transform=self.test_t)
            
        else:
            raise NotImplementedError(f"The dataset {self.name} is not supported.")

    def train_dataloader(self):
        return self.create_loader(self.train_ds, shuffle=True)
    
    def val_dataloader(self):
        return self.create_loader(self.test_ds, shuffle=False)
    
    def test_dataloader(self):
        return self.create_loader(self.test_ds, shuffle=False)