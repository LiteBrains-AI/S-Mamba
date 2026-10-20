import torch
import torch.nn as nn


class SLayer(nn.Module):

    def __init__(self, d_inner):
        super().__init__()

        self.energy_proj = nn.Linear(1, d_inner)

    def forward(self, x, energy):
        """
        Combine token representation and scalar energy.

        Args:
            x: [B, d_inner]
            energy: [B]

        Returns:
            combined: [B, d_inner]

        Method:
            1. Project scalar energy into feature space.
            2. Add it to the token representation.

        Note: The current implementation uses a simple additive interaction.
        """

        energy = energy.unsqueeze(-1)

        energy_features = self.energy_proj(energy)

        combined = x + energy_features

        return combined


# Note:
# SLayer combines the token representation with scalar energy before admission gating.
# Energy is projected into the token feature space and added to the token representation.
# The current implementation uses a simple additive interaction. Multiplicative interaction may be explored later.

# The intention is to let energy influence admission decisions.

# Future experiments will explore mechanisms that explicitly increase gate values in response to higher energy, Or based on energy concentration in `h`