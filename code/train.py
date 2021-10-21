import torch
from torch.nn import Module
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn


def train(epoch: int,
          train_loader: DataLoader,
          model: Module,
          criterion: nn.CrossEntropyLoss,
          optimizer: optim.Optimizer
          ):
    running_loss = 0.
    for batch_idx, data in enumerate(train_loader):
        inputs, target = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print("[%d, %5d] loss: %.3f" % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.
