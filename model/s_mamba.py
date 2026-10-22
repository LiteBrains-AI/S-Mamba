# ============================================================
# S-Mamba — Final-State Readout
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
# - No timestep output collection or stacking
# - Final recurrent state -> output projection
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
        # 1. Input projection
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
        # 5. Final-state output projection
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
    # Straight-through binary release decision
    # ========================================================

    @staticmethod
    def binary_release_decision(probability):

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

        # h: [B, d_inner, d_state]

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
        inspect_support=False,
        inspect_energy=False,
    ):
        """
        Parameters
        ----------
        x:
            [B, L, d_model]

        Returns
        -------
        y:
            [B, d_model]

        The block reads out only the final recurrent state.
        """

        B, L, _ = x.shape

        # ====================================================
        # 1. Input projection
        # ====================================================

        projected = self.in_proj(x)

        x_part, z = projected.chunk(
            2,
            dim=-1
        )

        # The second projection branch is intentionally unused.
        del z

        # ====================================================
        # 2. Local causal convolution
        # ====================================================

        x_conv = x_part.transpose(1, 2)

        x_conv = self.conv1d(x_conv)

        # Keep the original sequence length.
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

        # Retained for comparison with the original readout.
        # The final-state readout does not currently use C_t.
        C_t = ssm_params[
            ...,
            1 + self.d_state:
        ]

        del C_t

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
        # 5. Initialize recurrent state
        # ====================================================

        h = torch.zeros(
            B,
            self.d_inner,
            self.d_state,
            device=x.device,
            dtype=x.dtype
        )

        # Reset energy pressure for this sequence.
        self.energy.reset()

        # ====================================================
        # 6. Inspection histories
        # ====================================================

        support_history = []
        distance_history = []

        energy_history = []
        pressure_history = []
        release_history = []

        released_piece_history = []
        h_norm_history = []

        # ====================================================
        # 7. Sequential recurrence
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
            # Recurrent state update
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
            # STATE CONSOLIDATION
            # =================================================

            new_h = self.consolidate_state(h)

            # No release -> retain h.
            # Release    -> replace h with consolidated state.
            h = (
                h
                + release_decision.view(B, 1, 1)
                * (new_h - h)
            )

            previous_release = release_decision

            # =================================================
            # INSPECTION
            # =================================================

            if inspect_support:

                support_history.append(
                    support.detach()
                )

                distance_history.append(
                    distance.detach()
                )

            if inspect_energy:

                energy_history.append(
                    energy_t.detach()
                )

                pressure_history.append(
                    pressure_t.detach()
                )

                release_history.append(
                    release_decision.detach()
                )

                released_piece_history.append(
                    new_h.detach()
                )

                h_norm_history.append(
                    h.detach().norm(
                        dim=(1, 2)
                    )
                )

        # ====================================================
        # 8. FINAL-STATE READOUT
        # ====================================================
        #
        # h:       [B, d_inner, d_state]
        # h_final: [B, d_inner]
        # y:       [B, d_model]
        #
        # No outputs list.
        # No torch.stack(outputs).
        # ====================================================

        h_final = h.mean(
            dim=-1
        )

        y = self.out_proj(h_final)

        # ====================================================
        # 9. Inspection output
        # ====================================================

        if inspect_support or inspect_energy:

            result = [y]

            if inspect_support:

                result.extend([
                    torch.stack(
                        support_history,
                        dim=1
                    ),
                    torch.stack(
                        distance_history,
                        dim=1
                    )
                ])

            if inspect_energy:

                result.extend([
                    torch.stack(
                        energy_history,
                        dim=1
                    ),
                    torch.stack(
                        pressure_history,
                        dim=1
                    ),
                    torch.stack(
                        release_history,
                        dim=1
                    ),
                    torch.stack(
                        released_piece_history,
                        dim=1
                    ),
                    torch.stack(
                        h_norm_history,
                        dim=1
                    )
                ])

            return tuple(result)

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
        # S-Mamba blocks
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
        inspect_support=False,
        inspect_energy=False,
    ):
        """
        Parameters
        ----------
        x:
            [B, L, 1]

        Returns
        -------
        y:
            [B, d_model]

        This version produces one final representation per
        input sequence rather than one output per timestep.
        """

        # [B, L, 1] -> [B, L, d_model]
        x = self.input_proj(x)

        all_support = []
        all_distance = []

        all_energy = []
        all_pressure = []
        all_release = []
        all_released_pieces = []
        all_h_norm = []

        for layer in self.layers:

            # Every block expects a sequence.
            # Later blocks receive the previous block's
            # final representation as a length-1 sequence.
            layer_result = layer(
                x,
                inspect_support=inspect_support,
                inspect_energy=inspect_energy,
            )

            # ------------------------------------------------
            # No inspection
            # ------------------------------------------------

            if not (
                inspect_support
                or inspect_energy
            ):

                x = layer_result.unsqueeze(1)

                continue

            # ------------------------------------------------
            # Parse inspection output
            # ------------------------------------------------

            index = 0

            x = layer_result[index].unsqueeze(1)

            index += 1

            if inspect_support:

                all_support.append(
                    layer_result[index]
                )

                all_distance.append(
                    layer_result[index + 1]
                )

                index += 2

            if inspect_energy:

                all_energy.append(
                    layer_result[index]
                )

                all_pressure.append(
                    layer_result[index + 1]
                )

                all_release.append(
                    layer_result[index + 2]
                )

                all_released_pieces.append(
                    layer_result[index + 3]
                )

                all_h_norm.append(
                    layer_result[index + 4]
                )

        # [B, 1, d_model] -> [B, d_model]
        x = x.squeeze(1)

        x = self.norm(x)

        # ----------------------------------------------------
        # Inspection output
        # ----------------------------------------------------

        if inspect_support or inspect_energy:

            result = [x]

            if inspect_support:

                result.extend([
                    all_support,
                    all_distance
                ])

            if inspect_energy:

                result.extend([
                    all_energy,
                    all_pressure,
                    all_release,
                    all_released_pieces,
                    all_h_norm
                ])

            return tuple(result)

        return x

# Note: Released representations `h` could be used to train Jev alike discriminative models