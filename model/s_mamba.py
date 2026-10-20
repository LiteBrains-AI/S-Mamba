# ============================================================
# S-Mamba ADV1
#
# Refactor:
# - Continuous wave input
# - Scalar input -> linear feature projection
# - No additional output GELU gate
# - No additional learned output gate
# - No SSM D skip connection
# - No layer residual/skip connection
# - Support module retained
# - Scalar relational energy retained
# - SLayer combines token representation and energy
# - Learnable admission gate controls incoming information
# - No accumulated pressure
# - No binary release decision
# - No state consolidation
# ============================================================


import torch
import torch.nn as nn
import torch.nn.functional as F

from .support import Support
from .energy import Energy
from .s_layer import SLayer


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
        # 7. Scalar energy
        # ----------------------------------------------------

        self.energy = Energy()

        # ----------------------------------------------------
        # 8. Token-energy interaction
        # ----------------------------------------------------

        self.s_layer = SLayer(
            d_inner=self.d_inner
        )

        # ----------------------------------------------------
        # 9. Admission gate
        # ----------------------------------------------------

        self.admission_gate = nn.Linear(
            self.d_inner,
            self.d_inner
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, x):
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

        # Retain the original projection structure.
        # No additional output gate is used.

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
        # 5. Initialize state
        # ====================================================

        h = torch.zeros(
            B,
            self.d_inner,
            self.d_state,
            device=x.device,
            dtype=x.dtype
        )

        outputs = []

        # ====================================================
        # 6. Sequential recurrence
        # ====================================================

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
            # ENERGY
            # =================================================

            energy_t = self.energy(
                h_before,
                x_conv[:, t, :]
            )

            # =================================================
            # TOKEN-ENERGY INTERACTION
            # =================================================

            combined = self.s_layer(
                x_conv[:, t, :],
                energy_t
            )

            # =================================================
            # ADMISSION GATE
            # =================================================

            gate = torch.sigmoid(
                self.admission_gate(combined)
            )

            # =================================================
            # SUPPORT
            # =================================================

            support, distance = self.support(
                h_before,
                x_conv[:, t, :]
            )

            # =================================================
            # INCOMING INFORMATION
            # =================================================

            incoming = (
                x_conv[:, t, :].unsqueeze(-1)
                * dB
            )

            # Preserve the existing support mechanism while allowing the learned gate to control admission.

            supported_incoming = (
                support
                * incoming
                * gate.unsqueeze(-1)
            )

            # =================================================
            # MAMBA STATE UPDATE
            # =================================================

            h = (
                dA * h_before
                + supported_incoming
            )

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
        # 7. Stack outputs
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

    def forward(self, x):

        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        return x
