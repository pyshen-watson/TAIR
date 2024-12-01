from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import Dataset


@dataclass
class CUB200(Dataset):
    root: str  # The root directory of the dataset. Ex. 'data/CUB_200_2011'
    is_train: bool = (
        True  # If True, return the training set, otherwise return the test set.
    )
    is_degraded: bool = False  # If True, return the degraded version of the dataset.

    def __post_init__(self):
        # Set the directories
        self.root = Path(self.root)
        self.images = self._read_file(self.root / "images.txt")
        self.labels = self._read_file(self.root / "image_class_labels.txt", int)

        # Create the split
        train_test_split = self._read_file(
            self.root / "train_test_split.txt", int
        )  # 1 for train and 0 for test
        self.indices = [
            i for i, split in enumerate(train_test_split) if split == self.is_train
        ]
        self.image_dir = (
            self.root / "images"
            if not self.is_degraded
            else self.root / "degraded_images"
        )

    def _read_file(self, filepath, dtype=str) -> list[str | int]:
        """
        The first column of the files is the image ID and the second column is the infomation we want.
        In this function, we read the second column of the file and return a list of the values.
        """
        with open(filepath, "r") as f:
            return [dtype(line.strip().split()[1]) for line in f.readlines()]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image_path = self.image_dir / self.images[real_idx]
        image = Image.open(image_path)
        label = self.labels[real_idx]
        return image, label
