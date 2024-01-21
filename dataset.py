from __future__ import annotations
import os
import torch
import struct
import numpy as np

from typing import List
import matplotlib.pyplot as plt

from utils import device

IMG_DATASET_HEADER_SIZE = 4 * 4
LBL_DATASET_HEADER_SIZE = 4 * 2

class ImageDataset:

    class Header:
        def __init__(self, data: bytes) -> None:
            values = struct.unpack('>IIII', data[:IMG_DATASET_HEADER_SIZE])

            self.magic = values[0]
            self.amount = values[1]
            self.rows = values[2]
            self.cols = values[3]

            assert(self.magic == 2051)

    def __init__(self, data: bytes) -> None:
        self.header = ImageDataset.Header(data)

        self.raw_data = np.array(bytearray(data[IMG_DATASET_HEADER_SIZE:]))
        self.raw_data = self.raw_data.reshape(-1, self.header.rows * self.header.cols)

        self.images = torch.from_numpy(self.raw_data).float().to(device())
        self.images = self.images / 255

        assert(self.images.shape[0] == self.header.amount)

    def plot(self, idx):
        plt.imshow(self.raw_data[idx].reshape(-1, self.header.cols), cmap='gray', vmin=0, vmax=255)
        plt.show()


class LabelDataset:
    class Header:
        def __init__(self, data: bytes) -> None:
            values = struct.unpack('>II', data[:LBL_DATASET_HEADER_SIZE])

            self.magic = values[0]
            self.amount = values[1]

            assert(self.magic == 2049)

    def __init__(self, data: bytes) -> None:
        self.header = LabelDataset.Header(data)
        self.labels = []
        for label in bytearray(data[LBL_DATASET_HEADER_SIZE:]):
            data = np.zeros(10)
            data[label] = 1
            self.labels.append(torch.from_numpy(data).to(device()))

        assert(len(self.labels) == self.header.amount)


class Loader:
    @staticmethod
    def load_images(path: str) -> ImageDataset:
        if not os.path.exists(path):
            raise RuntimeError(f'File does not exist: {path}')
        with open(path, 'rb') as f:
            dataset = ImageDataset(f.read())
        return dataset

    @staticmethod
    def load_labels(path: str) -> LabelDataset:
        if not os.path.exists(path):
            raise RuntimeError(f'File does not exist: {path}')
        with open(path, 'rb') as f:
            dataset = LabelDataset(f.read())
        return dataset


class DigitDataset(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.img = Loader.load_images('dataset/train/images/train-images.idx3-ubyte')
        self.lbl = Loader.load_labels('dataset/train/labels/train-labels.idx1-ubyte')

        assert(len(self.img.images) == len(self.lbl.labels))

    def __len__(self):
        return len(self.img.images)

    def __getitem__(self, idx):
        return self.img.images[idx], self.lbl.labels[idx]
