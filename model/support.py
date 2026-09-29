import torch
import torch.nn as nn

class Support(nn.Module):

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, h, x):
        """
        x is [B, d_inner],

        h is [B, d_inner, d_state]
        
        Expand x to [B, d_inner, 1] for broadcasting

        """
        x_expanded = x.unsqueeze(-1)
        distance = torch.abs(h - x_expanded)
        support = torch.exp(-self.alpha * distance)

        return support, distance
