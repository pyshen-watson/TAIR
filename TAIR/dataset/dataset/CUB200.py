from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import Dataset
from torchvision import transforms as T


@dataclass
class CUB200(Dataset):
    
    root: str  # The root directory of the dataset. Ex. 'data/CUB_200_2011'
    is_train: bool = True  # If True, use the training set, otherwise return the test set.
    dg_type: str = "fog"  # The degraded type of the dataset. Support fog, rain, snow, and cloud.
    transform: T.Compose = None  # The transform function to apply to the images.

    def __post_init__(self):
        # Set the directories
        self.root = Path(self.root)
        self.images = self._read_file(self.root / "images.txt")
        self.labels = self._read_file(self.root / "image_class_labels.txt", int)
        self.n_classes = len(set(self.labels))

        # Create the split
        train_test_split = self._read_file(self.root / "train_test_split.txt", int)  # 1 for train and 0 for test
        self.indices = [ i for i, split in enumerate(train_test_split) if split == self.is_train ]
        
        # Initialized the image directories
        self.clear_dir = self.root / "images"
        self.degraded_dir = self.root / f"images_{self.dg_type}"
        assert self.clear_dir.exists(), f"The clear image directory: {self.clear_dir} does not exist."
        assert self.degraded_dir.exists(), f"The {self.dg_type} image directory: {self.degraded_dir} does not exist. You should run the create_degraded_images.py first."

    def _read_file(self, path: Path, dtype=str) -> list[str | int]:
        """
        The first column of the files is the image ID and the second column is the infomation we want.
        In this function, we read the second column of the file and return a list of the values.
        """
        with open(path, "r") as f:
            return [dtype(line.strip().split()[1]) for line in f.readlines()]
        
    def _read_image(self, path: Path) -> Image:
        img =  Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        
        """
        Return format: (clear_image, degraded_image, label)
        Return type: (PIL.Image, PIL.Image, int)
        """
        
        real_idx = self.indices[idx] # Get the split-free index
        clear_path = self.clear_dir / self.images[real_idx]
        degraded_path = self.degraded_dir / self.images[real_idx]       
            
        img_c = self._read_image(clear_path)
        img_d = self._read_image(degraded_path)
        label = self.labels[real_idx] - 1  # The label starts from 1, we need to subtract 1 to make it start from 0

        return img_c, img_d, label
