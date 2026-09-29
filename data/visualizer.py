from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# =========================
# Configuration
# =========================

DATA_DIR = Path("dataset")

NUM_SAMPLES = 1

RANDOM_SEED = 10


# =========================
# Visualizer
# =========================

def visualize_dataset():

    files = sorted(DATA_DIR.glob("sample_*.npz"))

    if not files:
        print(f"No dataset files found in: {DATA_DIR}")
        return

    rng = np.random.default_rng(RANDOM_SEED)

    selected_indices = rng.choice(
        len(files),
        size=min(NUM_SAMPLES, len(files)),
        replace=False
    )
    selected = [files[int(index)] for index in selected_indices]

    for file in selected:

        data = np.load(file)

        input_signal = data["input"]
        target_signal = data["target"]

        # Reconstruct complete signal for visualization
        signal = np.concatenate([
            input_signal[:1],
            target_signal
        ])

        plt.figure(figsize=(12, 4))

        plt.plot(
            signal,
            linewidth=1.5
        )

        plt.title(
            f"Waveform — {file.name}"
        )

        plt.xlabel("Timestep")
        plt.ylabel("Amplitude")

        plt.grid(True, alpha=0.25)

        plt.tight_layout()
        plt.show()


# =========================
# Main
# =========================

if __name__ == "__main__":
    visualize_dataset()
