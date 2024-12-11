import torch
torch.backends.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


class LogisticRegression(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(in_feature, out_feature)

    def forward(self, X):
        return self.linear(X)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def extract_feature(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2)
        return torch.flatten(x, 1)


class DVEstimator(LeNet):
    def __init__(self, scalar, X_valued=[], y_valued=[]):
        super(DVEstimator, self).__init__()
        self.null_value = Parameter(torch.zeros(1), requires_grad=True)

        self.X_valued = X_valued
        self.y_valued = y_valued
        self.n_valued = len(y_valued)
        self.scalar = scalar

        self.map_idx_full = np.arange(self.n_valued)

    def forward(self, subsets, ues):
        data_idx = np.any(subsets[:, :self.n_valued], axis=0)
        data = self.X_valued[data_idx]
        logit = super(DVEstimator, self).forward(data)
        if len(logit) != self.n_valued:
            map_idx = np.concatenate([np.where(data_idx)[0], np.where(~data_idx)[0]])
            map_idx = map_idx.argsort()
        else:
            map_idx = self.map_idx_full

        loss = 0.0
        for ss, ue in zip(subsets, ues):
            data_pos = ss[:self.n_valued]
            label = self.y_valued[data_pos]
            diff = logit[map_idx[data_pos], label].sum() - ue
            if ss[-1]:
                diff += self.null_value[0]
            loss += diff**2
        return loss / len(ues)

    def estimate(self, X, y):
        self.eval()
        with torch.no_grad():
            logit = super(DVEstimator, self).forward(X)
            estimates = logit[torch.arange(len(y)), y] - self.null_value[0]
        self.train()
        return estimates.numpy() * self.scalar

