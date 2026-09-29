# ============================================
# S-Mamba V1 — Small Mamba Baseline
# Raw Wave → Mamba → Next Value
# ============================================

import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from data.hard_generator import generate_dataset

# ============================================
# Configuration
# ============================================

DATASET_DIR = Path("dataset")


D_MODEL = 64
D_STATE = 16

NUM_LAYERS = 2

BATCH_SIZE = 40
LEARNING_RATE = 1e-3

EPOCHS = 10

TRAIN_RATIO = 0.8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

# ============================================
# Generate Dataset
# ============================================

OUTPUT_DIR = Path("dataset")

generate_dataset()
print(
    f"\nHard dataset generated in: "
    f"{OUTPUT_DIR}"
)

# ============================================
# Dataset
# ============================================

class WaveDataset(Dataset):

    def __init__(self, dataset_dir):

        self.files = sorted(
            dataset_dir.glob("sample_*.npz")
        )

        if len(self.files) == 0:
            raise RuntimeError(
                f"No samples found in {dataset_dir}"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        data = np.load(self.files[idx])

        x = torch.tensor(
            data["input"],
            dtype=torch.float32
        )

        y = torch.tensor(
            data["target"],
            dtype=torch.float32
        )

        return x, y


# ============================================
# Load Dataset
# ============================================

dataset = WaveDataset(DATASET_DIR)

train_size = int(
    len(dataset) * TRAIN_RATIO
)

test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Total samples :", len(dataset))
print("Train samples :", len(train_dataset))
print("Test samples  :", len(test_dataset))


# ============================================
# Small Mamba Block
# ============================================

class SmallMambaBlock(nn.Module):

    def __init__(
        self,
        d_model,
        d_state
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state

        # ------------------------------------
        # Expand
        # ------------------------------------

        self.d_inner = d_model * 2

        self.in_proj = nn.Linear(
            d_model,
            self.d_inner * 2
        )

        # ------------------------------------
        # Local convolution
        # ------------------------------------

        self.conv = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=4,
            groups=self.d_inner,
            padding=3
        )

        # ------------------------------------
        # SSM parameter projection
        #
        # dt, B, C
        # ------------------------------------

        self.x_proj = nn.Linear(
            self.d_inner,
            1 + 2 * d_state
        )

        self.dt_proj = nn.Linear(
            1,
            self.d_inner
        )

        # ------------------------------------
        # A parameter
        # ------------------------------------

        self.A_log = nn.Parameter(
            torch.randn(
                self.d_inner,
                d_state
            )
        )

        # ------------------------------------
        # Output projection
        # ------------------------------------

        self.out_proj = nn.Linear(
            self.d_inner,
            d_model
        )

    def forward(self, x):

        # x:
        # [B, L, D]

        B, L, D = x.shape

        # ------------------------------------
        # Input projection
        # ------------------------------------

        projected = self.in_proj(x)

        x_part, z = projected.chunk(
            2,
            dim=-1
        )

        # ------------------------------------
        # Depthwise causal convolution
        # ------------------------------------

        x_conv = x_part.transpose(1, 2)

        x_conv = self.conv(x_conv)

        # Remove extra right-side positions
        x_conv = x_conv[:, :, :L]

        x_conv = x_conv.transpose(1, 2)

        x_conv = torch.nn.functional.silu(
            x_conv
        )

        # ------------------------------------
        # SSM parameters
        # ------------------------------------

        params = self.x_proj(x_conv)

        dt_raw = params[:, :, :1]

        B_param = params[
            :, :, 1:1 + self.d_state
        ]

        C_param = params[
            :, :, 1 + self.d_state:
        ]

        # ------------------------------------
        # Delta
        # ------------------------------------

        dt = torch.nn.functional.softplus(
            self.dt_proj(dt_raw)
        )

        # ------------------------------------
        # A
        # ------------------------------------

        A = -torch.exp(
            self.A_log
        )

        # ------------------------------------
        # Recurrent state
        #
        # h:
        # [B, d_inner, d_state]
        # ------------------------------------

        h = torch.zeros(
            B,
            self.d_inner,
            self.d_state,
            device=x.device,
            dtype=x.dtype
        )

        outputs = []

        # ------------------------------------
        # Sequential SSM
        # ------------------------------------

        for t in range(L):

            dt_t = dt[:, t, :]
            B_t = B_param[:, t, :]
            C_t = C_param[:, t, :]

            x_t = x_conv[:, t, :]

            # --------------------------------
            # Discretized A
            # --------------------------------

            dA = torch.exp(
                dt_t.unsqueeze(-1) * A
            )

            # --------------------------------
            # Input contribution
            # --------------------------------

            dB = (
                dt_t.unsqueeze(-1)
                * B_t.unsqueeze(1)
            )

            # --------------------------------
            # State update
            # --------------------------------

            h = (
                dA * h
                +
                dB * x_t.unsqueeze(-1)
            )

            # --------------------------------
            # State readout
            # --------------------------------

            y_t = (
                h
                * C_t.unsqueeze(1)
            ).sum(dim=-1)

            outputs.append(y_t)

        # ------------------------------------
        # [B, L, d_inner]
        # ------------------------------------

        y = torch.stack(
            outputs,
            dim=1
        )

        # ------------------------------------
        # Native Mamba-style gate
        # ------------------------------------

        y = y * torch.nn.functional.silu(z)

        # ------------------------------------
        # Output projection
        # ------------------------------------

        y = self.out_proj(y)

        return y


# ============================================
# Mamba Model
# ============================================

class WaveMamba(nn.Module):

    def __init__(
        self,
        d_model=64,
        d_state=16,
        num_layers=2
    ):
        super().__init__()

        # ------------------------------------
        # Raw scalar → Mamba dimension
        # ------------------------------------

        self.input_projection = nn.Linear(
            1,
            d_model
        )

        # ------------------------------------
        # Mamba layers
        # ------------------------------------

        self.layers = nn.ModuleList([
            SmallMambaBlock(
                d_model=d_model,
                d_state=d_state
            )
            for _ in range(num_layers)
        ])

        # ------------------------------------
        # Prediction head
        # ------------------------------------

        self.output = nn.Linear(
            d_model,
            1
        )

    def forward(self, x):

        # x:
        # [B, L]

        x = x.unsqueeze(-1)

        # [B, L, 1]
        # →
        # [B, L, D]

        x = self.input_projection(x)

        # ------------------------------------
        # Mamba stack
        # ------------------------------------

        for layer in self.layers:

            x = layer(x)

        # ------------------------------------
        # Next-value prediction
        # ------------------------------------

        x = self.output(x)

        return x.squeeze(-1)


# ============================================
# Create Model
# ============================================

model = WaveMamba(
    d_model=D_MODEL,
    d_state=D_STATE,
    num_layers=NUM_LAYERS
).to(DEVICE)

print("\nModel:")
print(model)

print(
    "\nParameters:",
    sum(
        p.numel()
        for p in model.parameters()
    )
)


# ============================================
# Loss + Optimizer
# ============================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================
# Training
# ============================================

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0

    for x, y in train_loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # ------------------------------------
        # Forward
        # ------------------------------------

        prediction = model(x)

        # ------------------------------------
        # Loss
        # ------------------------------------

        loss = criterion(
            prediction,
            y
        )

        # ------------------------------------
        # Backprop
        # ------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    # ========================================
    # Evaluation
    # ========================================

    model.eval()

    test_loss = 0.0

    with torch.no_grad():

        for x, y in test_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            prediction = model(x)

            loss = criterion(
                prediction,
                y
            )

            test_loss += loss.item()

    test_loss /= len(test_loader)


    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train MSE: {train_loss:.6f} | "
        f"Test MSE: {test_loss:.6f}"
    )
