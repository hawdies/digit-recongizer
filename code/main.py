import torch
from torch.nn import Module
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import json

import train
import test
from load_dataset import DigitDataset, DigitDatasetNoLabel
import save_output
from network_residual import Net


def read_acc():
    acc = 0.
    with open("../input/acc.json", encoding="utf-8") as f:
        d = json.load(f)
        acc = float(d["acc"])
    return acc


def write_acc(acc: float):
    with open("../input/acc.json", mode="w", encoding="utf-8") as f:
        dic = {"acc": acc}
        f.write(json.dumps(dic))


def train_module():
    max_acc = read_acc()
    module = Net()
    module.load_state_dict(torch.load("best_module.pth"))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(module.parameters(), lr=0.01, momentum=0.5)
    train_data = DigitDataset("../input/train.csv", trainset=True)
    train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True, num_workers=2)
    test_data = DigitDataset("../input/train.csv", trainset=False)
    test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=False, num_workers=2)
    list_epoch = []
    list_acc = []
    for epoch in range(10):
        train.train(epoch, train_loader, module, criterion, optimizer)
        acc = test.test(test_loader, module)
        list_epoch.append(epoch)
        list_acc.append(acc)
        if acc > max_acc:
            print("save model")
            max_acc = acc
            torch.save(module.state_dict(), "best_module.pth")
    write_acc(max_acc)
    plt.plot(list_epoch, list_acc)
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.grid()
    plt.show()


def test_realdata():
    module = Net()
    module.load_state_dict(torch.load("best_module.pth"))
    test_data = DigitDatasetNoLabel("../input/test.csv", trainset=False)
    test_loader = DataLoader(dataset=test_data, batch_size=len(test_data), shuffle=False, num_workers=2)
    result = test.test_no_label(test_loader, module)
    save_output.save_output(result, "../output/result.csv")


if __name__ == '__main__':
    # train_module()
    test_realdata()
