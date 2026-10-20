import torch
import torch.nn as nn


class Energy(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, h, x):
        """
        Calculate relational energy between the current token representation and the existing state.

        Args:
            h: [B, d_inner, d_state]
            x: [B, d_inner]

        Returns:
            energy: [B]

        Formula:
            E = 0.5 * mean((h - x)^2)
        """

        x_expanded = x.unsqueeze(-1)

        distance = h - x_expanded

        energy = 0.5 * distance.pow(2).mean(
            dim=(1, 2)
        )

        return energy


# Note:
# The current energy mechanism calculates a scalar value by averaging squared distances across the feature and state dimensions.
# This is sufficient for the current simple dataset.

# Future experiments may explore vector-wise energy factors to preserve feature-level information for embedding-based inputs.
