import torch.nn as nn
import torch

class ContrastiveLossFunc(nn.Module):
    def __init__(self, margin = 11.0):
        super(ContrastiveLossFunc, self).__init__()
        self.margin = margin

    def forward(self, euclidean_distance, label):
        loss = torch.mean(label * torch.pow(euclidean_distance, 2) + (1-label) * torch.pow((torch.clamp(self.margin - euclidean_distance, min=0.0)), 2))
        return loss
