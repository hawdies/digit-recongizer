import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class DigitDataset(Dataset):

    def __init__(self, filepath, trainset: bool) -> None:
        xy = np.loadtxt(filepath, dtype=int, delimiter=",", skiprows=1)
        xy = xy[:32000] if trainset else xy[32000:]
        print(xy.shape)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, 1:]).view(-1, 1, 28, 28) / 255  # [42000, 1, 28, 28]
        self.y_data = torch.from_numpy(xy[:, 0]).long()  # [42000]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class DigitDatasetNoLabel(Dataset):

    def __init__(self, filepath, trainset: bool) -> None:
        xy = np.loadtxt(filepath, dtype=int, delimiter=",", skiprows=1)
        print(xy.shape)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy).view(-1, 1, 28, 28) / 255  # [10000, 1, 28, 28]

    def __getitem__(self, index):
        return self.x_data[index]

    def __len__(self):
        return self.len


if __name__ == '__main__':
    dataset = DigitDataset("../input/train.csv", False)
    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)
