import torch
import torch.nn as nn


class OutlierDetectionModel(nn.Module):
    def __init__(self):
        super(OutlierDetectionModel, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(768, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 1)
                                 )
        for name, param in self.mlp.named_parameters():
            if 'weight' in name:
                torch.nn.init.kaiming_normal_(param)

    def forward(self, x):
        x = self.mlp(x)
        x = torch.sigmoid(x)
        return x
