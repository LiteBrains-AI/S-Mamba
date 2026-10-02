# ============================================================
# S-Mamba
#
# Refactor:
# - Continuous wave input
# - Scalar input -> linear feature projection
# - No additional output GELU gate
# - No additional learned output gate
# - No SSM D skip connection
# - No layer residual/skip connection
# - Support actively controls state admission
# - Energy accumulates relational pressure
# - Energy produces a binary release decision
# - No ReleaseGate
# - Release consolidates h using mean
# - Released state becomes NEW h
# - No soft h/released_h interpolation
#
# ============================================================


import torch
import torch.nn as nn
import torch.nn.functional as F

from .support import Support
from .energy import Energy


# ============================================================
# S-Mamba Block
# ============================================================

class SMambaBlock(nn.Module):

    def __init__(
        self,
        d_model=128,
        d_state=16,
        d_conv=4,
        expand=2,
        support_alpha=1.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        # ----------------------------------------------------
        # 1. Mamba input projection
        # ----------------------------------------------------

        self.in_proj = nn.Linear(
            d_model,
            2 * self.d_inner
        )

        # ----------------------------------------------------
        # 2. Local causal depthwise convolution
        # ----------------------------------------------------

        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1
        )

        # ----------------------------------------------------
        # 3. Generate Δ, B and C
        # ----------------------------------------------------

        self.x_proj = nn.Linear(
            self.d_inner,
            1 + 2 * d_state
        )

        self.dt_proj = nn.Linear(
            1,
            self.d_inner
        )

        # ----------------------------------------------------
        # 4. SSM parameter A
        # ----------------------------------------------------

        self.A_log = nn.Parameter(
            torch.log(
                torch.arange(
                    1,
                    d_state + 1,
                    dtype=torch.float32
                )
                .unsqueeze(0)
                .repeat(self.d_inner, 1)
            )
        )

        # ----------------------------------------------------
        # 5. Output projection
        # ----------------------------------------------------

        self.out_proj = nn.Linear(
            self.d_inner,
            d_model
        )

        # ----------------------------------------------------
        # 6. Support
        # ----------------------------------------------------

        self.support = Support(
            alpha=support_alpha
        )

        # ----------------------------------------------------
        # 7. Energy / pressure
        # ----------------------------------------------------

        self.energy = Energy()

    # ========================================================
    # Straight-through binary decision
    # ========================================================

    @staticmethod
    def binary_release_decision(probability):
        """
        Forward:
            Returns a hard 0/1 release decision.

        Backward:
            Uses the sigmoid probability as the surrogate
            gradient.
        """

        hard = (
            probability >= 0.5
        ).to(probability.dtype)

        return (
            hard - probability
        ).detach() + probability

    # ========================================================
    # State consolidation
    # ========================================================

    @staticmethod
    def consolidate_state(h):
        """
        Thin the accumulated state by averaging over the
        internal state dimension.

        Input:
            h -> [B, d_inner, d_state]

        Output:
            new_h -> [B, d_inner, d_state]

        The state shape is preserved so the next Mamba
        recurrence can continue normally.
        """

        new_h = h.mean(
            dim=-1,
            keepdim=True
        )

        return new_h.expand_as(h)

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        x,
    ):
        """
        Parameters
        ----------
        x:
            [B, L, d_model]

        Returns
        -------
        y:
            [B, L, d_model]
        """

        B, L, _ = x.shape

        # ====================================================
        # 1. Mamba input projection
        # ====================================================

        projected = self.in_proj(x)

        x_part, z = projected.chunk(
            2,
            dim=-1
        )

        # z is retained as part of the original Mamba-style projection structure.

        del z

        # ====================================================
        # 2. Local causal convolution
        # ====================================================

        x_conv = x_part.transpose(1, 2)

        x_conv = self.conv1d(x_conv)

        x_conv = x_conv[:, :, :L]

        x_conv = x_conv.transpose(1, 2)

        x_conv = F.gelu(x_conv)

        # ====================================================
        # 3. Generate SSM parameters
        # ====================================================

        ssm_params = self.x_proj(x_conv)

        dt_raw = ssm_params[..., :1]

        B_t = ssm_params[
            ...,
            1:1 + self.d_state
        ]

        C_t = ssm_params[
            ...,
            1 + self.d_state:
        ]

        dt = F.softplus(
            self.dt_proj(dt_raw)
        )

        # ====================================================
        # 4. State matrix A
        # ====================================================

        A = -torch.exp(
            self.A_log
        )

        # ====================================================
        # 5. Initialize h
        # ====================================================

        h = torch.zeros(
            B,
            self.d_inner,
            self.d_state,
            device=x.device,
            dtype=x.dtype
        )

        # Reset Energy pressure for this sequence.
        self.energy.reset()

        outputs = []

        # ====================================================
        # 6. Sequential recurrence
        # ====================================================

        previous_release = torch.zeros(
            B,
            device=x.device,
            dtype=x.dtype
        )

        for t in range(L):

            # ------------------------------------------------
            # Selective discretization
            # ------------------------------------------------

            dA = torch.exp(
                dt[:, t, :].unsqueeze(-1)
                * A.unsqueeze(0)
            )

            # ------------------------------------------------
            # Input contribution
            # ------------------------------------------------

            dB = (
                dt[:, t, :].unsqueeze(-1)
                * B_t[:, t, :].unsqueeze(1)
            )

            # ------------------------------------------------
            # Existing state
            # ------------------------------------------------

            h_before = h

            # =================================================
            # SUPPORT
            # =================================================

            support, distance = self.support(
                h_before,
                x_conv[:, t, :]
            )

            # ------------------------------------------------
            # Incoming information
            # ------------------------------------------------

            incoming = (
                x_conv[:, t, :].unsqueeze(-1)
                * dB
            )

            supported_incoming = (
                support * incoming
            )

            # ------------------------------------------------
            # Mamba state update
            # ------------------------------------------------

            h = (
                dA * h_before
                + supported_incoming
            )

            # =================================================
            # ENERGY / PRESSURE
            # =================================================

            energy_t, pressure_t, release_probability = (
                self.energy(
                    h_before,
                    x_conv[:, t, :],
                    reset=previous_release
                )
            )

            # =================================================
            # BINARY RELEASE DECISION
            # =================================================

            release_decision = (
                self.binary_release_decision(
                    release_probability
                )
            )

            # =================================================
            # RELEASE / STATE CONSOLIDATION
            #
            # The gate does not destroy h.
            #
            # Release means that the accumulated state is consolidated and becomes the new state.
            # =================================================

            new_h = self.consolidate_state(h)

            # ------------------------------------------------
            # Hard release event
            #
            # Forward:
            #   no release -> keep h
            #   release    -> use new_h
            #
            # Backward:
            #   release_probability provides the surrogate
            #   gradient through the straight-through decision.
            # ------------------------------------------------

            h = (
                h
                + release_decision.view(
                    B, 1, 1
                ) * (
                    new_h - h
                )
            )

            # ------------------------------------------------
            # Save release for next Energy step.
            # ------------------------------------------------

            previous_release = release_decision

            # =================================================
            # READ FROM STATE
            # =================================================

            y_t = torch.sum(
                h
                * C_t[:, t, :].unsqueeze(1),
                dim=-1
            )

            outputs.append(y_t)

        # ====================================================
        # 7. Stack Mamba outputs
        # ====================================================

        y = torch.stack(
            outputs,
            dim=1
        )

        # ====================================================
        # 8. Output projection
        # ====================================================

        y = self.out_proj(y)

        return y


# ============================================================
# S-Mamba
# ============================================================

class SMamba(nn.Module):

    def __init__(
        self,
        d_model=128,
        d_state=16,
        d_conv=4,
        expand=2,
        n_layers=2,
        support_alpha=1.0,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Scalar -> model representation
        # ----------------------------------------------------

        self.input_proj = nn.Linear(
            1,
            d_model
        )

        # ----------------------------------------------------
        # Mamba layers
        # ----------------------------------------------------

        self.layers = nn.ModuleList([
            SMambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                support_alpha=support_alpha,
            )
            for _ in range(n_layers)
        ])

        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        self.norm = nn.LayerNorm(
            d_model
        )

    def forward(
        self,
        x,
    ):

        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x)

        # ====================================================
        # Final normalization
        # ====================================================

        x = self.norm(x)

        return x
