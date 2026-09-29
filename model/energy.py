import torch
import torch.nn as nn


class Energy(nn.Module):

    def __init__(self, capacity=1.0, pressure_decay=0.0):
        super().__init__()

        self.capacity = capacity
        self.pressure_decay = pressure_decay

        # Stateful accumulated pressure.
        self.pressure = None

    def reset(self):
        """
        Reset accumulated pressure.

        Call this at the beginning of every new sequence.
        """
        self.pressure = None

    def forward(self, h, x, reset=None):
        """
        Measure relational energy and accumulate it as pressure.

        h:
            [B, d_inner, d_state]

        x:
            [B, d_inner]

        reset:
            [B]
            Previous release decision.

            1 -> previous pressure was released
            0 -> continue accumulating pressure

        Returns:
            energy:
                [B]

            pressure:
                [B]

            release_probability:
                [B]

        Concept:

            distance
                ↓
              energy
                ↓
            accumulate
                ↓
             pressure
                ↓
             capacity
                ↓
        release probability

        """

        # --------------------------------------------------
        # 1. Match x with every state dimension
        # --------------------------------------------------

        x_expanded = x.unsqueeze(-1)

        # --------------------------------------------------
        # 2. Relational distance
        # --------------------------------------------------

        distance = torch.abs(
            h - x_expanded
        )

        # --------------------------------------------------
        # 3. Relational energy
        # --------------------------------------------------

        energy = 0.5 * distance.pow(2).mean(
            dim=(1, 2)
        )

        # --------------------------------------------------
        # 4. Initialize pressure
        # --------------------------------------------------

        if self.pressure is None:
            self.pressure = torch.zeros_like(energy)

        # --------------------------------------------------
        # 5. Reset pressure after previous release
        # --------------------------------------------------

        if reset is not None:
            self.pressure = (
                self.pressure
                * (1.0 - reset.detach())
            )

        # --------------------------------------------------
        # 6. Accumulate current energy
        # --------------------------------------------------

        self.pressure = (
            self.pressure * (1.0 - self.pressure_decay)
            + energy
        )

        # --------------------------------------------------
        # 7. Convert pressure into release probability
        # --------------------------------------------------

        release_probability = torch.sigmoid(
            self.pressure - self.capacity
        )

        return (
            energy,
            self.pressure,
            release_probability
        )
