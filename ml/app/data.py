from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

from app.config import ModelConfig


class ImageDataModule(LightningDataModule):
    def __init__(
        self,
        train_path: str,
        val_path: str,
        test_path: str,
        batch_size: int,
        img_size: int,
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.img_size = img_size
        self.train_transforms = self._get_train_transforms()
        self.val_transforms = self._get_val_transforms()
        self.test_transforms = self._get_test_transforms()

    def _get_train_transforms(self):
        return T.Compose(
            [
                T.Resize(self.img_size),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.ToTensor(),
                T.Normalize(mean=ModelConfig.IMG_MEAN, std=ModelConfig.IMG_STD),
            ]
        )

    def _get_val_transforms(self):
        return T.Compose(
            [
                T.Resize(self.img_size),
                T.ToTensor(),
                T.Normalize(mean=ModelConfig.IMG_MEAN, std=ModelConfig.IMG_STD),
            ]
        )

    def _get_test_transforms(self):
        return T.Compose(
            [
                T.Resize(self.img_size),
                T.ToTensor(),
                T.Normalize(mean=ModelConfig.IMG_MEAN, std=ModelConfig.IMG_STD),
            ]
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_data = ImageFolder(root=self.train_path, transform=self.train_transforms)
            self.val_data = ImageFolder(root=self.val_path, transform=self.val_transforms)
        if stage == "test" or stage is None:
            self.test_data = ImageFolder(root=self.test_path, transform=self.test_transforms)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            persistent_workers=True,
            pin_memory=True,
            num_workers=ModelConfig.NUM_WORKERS,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            persistent_workers=True,
            pin_memory=True,
            num_workers=ModelConfig.NUM_WORKERS,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            persistent_workers=True,
            pin_memory=True,
            num_workers=ModelConfig.NUM_WORKERS,
        )
