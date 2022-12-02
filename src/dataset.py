import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class MNISTDataModule(LightningDataModule):
    """MNIST data module."""

    def __init__(
        self, data_dir: str = "./", batch_size: int = 128, num_workers: int = 0
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def prepare_data(self):
        """Download MNIST dataset."""
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        """Split MNIST into train, val, test datasets."""
        if stage == "fit" or stage is None:
            train_dataset = datasets.MNIST(
                self.data_dir, train=True, transform=self.transform
            )
            indices = np.arange(len(train_dataset))
            np.random.shuffle(indices)
            self.train_dataset = Subset(train_dataset, indices[:50000])
            self.val_dataset = Subset(train_dataset, indices[50000:])
        if stage == "test" or stage is None:
            self.test_dataset = datasets.MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        """MNIST train dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        """MNIST val dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        """MNIST test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
