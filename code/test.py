import torch
from torch.nn import Module
from torch.utils.data import DataLoader


def test(test_loader: DataLoader, model: Module) -> float:
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print("Accuracy on test set: %.3f %%" % (100 * acc))
    return acc


def test_no_label(test_loader: DataLoader, model: Module):
    result = []
    with torch.no_grad():
        for image in test_loader:
            print(image.shape)
            outputs = model(image)
            print(outputs.shape)
            _, predicted = torch.max(outputs.data, dim=1)
    predicted = predicted.numpy()
    for idx, label in enumerate(predicted):
        result.append([idx + 1, label])
    return result
