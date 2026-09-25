from pathlib import Path
import numpy as np
from tqdm import tqdm


# =========================
# Configuration
# =========================

NUM_SAMPLES = 10000
SEQ_LEN = 100

AMPLITUDE_RANGE = (0.5, 1.0)
FREQUENCY_RANGE = (1.0, 5.0)

OUTPUT_DIR = Path("dataset")


# =========================
# Signal Generator
# =========================

def generate_signal(seq_len):
    t = np.linspace(0, 1, seq_len, endpoint=False)

    amplitude = np.random.uniform(
        AMPLITUDE_RANGE[0],
        AMPLITUDE_RANGE[1]
    )

    frequency = np.random.uniform(
        FREQUENCY_RANGE[0],
        FREQUENCY_RANGE[1]
    )

    phase = np.random.uniform(0, 2 * np.pi)

    if np.random.rand() < 0.5:
        signal = amplitude * np.sin(
            2 * np.pi * frequency * t + phase
        )
    else:
        signal = amplitude * np.cos(
            2 * np.pi * frequency * t + phase
        )

    return signal.astype(np.float32)


# =========================
# Dataset Generation
# =========================

def generate_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(NUM_SAMPLES), desc="Generating dataset"):
        signal = generate_signal(SEQ_LEN)

        # Next-value prediction
        input_signal = signal[:-1]
        target_signal = signal[1:]

        np.savez_compressed(
            OUTPUT_DIR / f"sample_{i:05d}.npz",
            input=input_signal,
            target=target_signal
        )


# =========================
# Main
# =========================

if __name__ == "__main__":
    generate_dataset()
    print(f"\nDataset generated in: {OUTPUT_DIR}")
