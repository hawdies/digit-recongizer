import torch.nn as nn
import torch.nn.functional as F

import inception


class Net(nn.Module):
    def __int__(self):
        super(Net, self).__int__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)

        self.incep1 = inception.InceptionA(in_channels=10)
        self.incep2 = inception.InceptionA(in_channels=20)

        self.mp = nn.MaxPool1d(2)
        self.fc = nn.Linear(1408, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x
