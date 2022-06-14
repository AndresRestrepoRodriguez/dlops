import os
from glob import glob

import pandas as pd
import pytorch_lightning as pl
import torch
from skimage.io import imread
from torch.utils.data import DataLoader, Dataset


class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = imread(self.images[idx])
        return torch.from_numpy(img).unsqueeze(0), self.labels[idx]
        # return torch.from_numpy(img).unsqueeze(1), self.labels[idx]


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, path, batch_size):
        super().__init__()
        self.path = path
        self.batch_size = batch_size

    def generate_df(self, l1, l2):
        return pd.DataFrame({
            'image': l1 + l2,
            'label': [1] * len(l1) + [0] * len(l2)
        })

    def setup(self, stage=None):
        train_3 = glob(os.path.join(self.path, 'train', '3', '*.png'))
        train_no3 = glob(os.path.join(self.path, 'train', 'no3', '*.png'))
        self.train_df = self.generate_df(train_3, train_no3)
        test_3 = glob(os.path.join(self.path, 'test', '3', '*.png'))
        test_no3 = glob(os.path.join(self.path, 'test', 'no3', '*.png'))
        self.test_df = self.generate_df(test_3, test_no3)
        self.train_ds = MNISTDataset(
            self.train_df.image.values,
            self.train_df.label.values
        )
        self.test_ds = MNISTDataset(
            self.test_df.image.values,
            self.test_df.label.values
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            shuffle=True,
            batch_size=self.batch_size
        )

    def val_dataloader(self, batch_size=None, shuffle=False):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size if batch_size is None else batch_size,
            shuffle=shuffle
        )
