from pathlib import Path
import numpy as np
from tqdm import tqdm


# =========================
# Configuration
# =========================

NUM_SAMPLES = 1000
SEQ_LEN = 100

AMPLITUDE_RANGE = (0.5, 1.0)
FREQUENCY_RANGE = (1.0, 5.0)

NOISE_STD = 0.035

OUTPUT_DIR = Path("dataset")


# =========================
# Signal Generator
# =========================

def generate_signal(seq_len):
    t = np.linspace(0, 1, seq_len, endpoint=False)

    # -------------------------
    # Base parameters
    # -------------------------

    amplitude = np.random.uniform(
        AMPLITUDE_RANGE[0],
        AMPLITUDE_RANGE[1]
    )

    frequency = np.random.uniform(
        FREQUENCY_RANGE[0],
        FREQUENCY_RANGE[1]
    )

    phase = np.random.uniform(
        0,
        2 * np.pi
    )

    # -------------------------
    # Slowly changing frequency
    # -------------------------

    frequency_change = np.random.uniform(
        -2.0,
        2.0
    )

    instantaneous_frequency = (
        frequency
        + frequency_change * t
    )

    # Integrate frequency to obtain phase.
    #
    # phase(t) = 2π * (f*t + 0.5*k*t²)
    #
    # This creates a chirp rather than
    # a stationary frequency.

    dynamic_phase = (
        2 * np.pi * (
            frequency * t
            + 0.5 * frequency_change * t ** 2
        )
        + phase
    )

    # -------------------------
    # Slowly changing amplitude
    # -------------------------

    amplitude_change = np.random.uniform(
        -0.4,
        0.4
    )

    dynamic_amplitude = (
        amplitude
        + amplitude_change * t
    )

    dynamic_amplitude = np.clip(
        dynamic_amplitude,
        0.2,
        1.2
    )

    # -------------------------
    # Main waveform
    # -------------------------

    if np.random.rand() < 0.5:
        signal = dynamic_amplitude * np.sin(
            dynamic_phase
        )
    else:
        signal = dynamic_amplitude * np.cos(
            dynamic_phase
        )

    # -------------------------
    # Second component
    # -------------------------

    if np.random.rand() < 0.5:

        frequency_2 = np.random.uniform(
            1.0,
            6.0
        )

        phase_2 = np.random.uniform(
            0,
            2 * np.pi
        )

        amplitude_2 = np.random.uniform(
            0.1,
            0.35
        )

        signal += amplitude_2 * np.sin(
            2 * np.pi * frequency_2 * t
            + phase_2
        )

    # -------------------------
    # Local structural change
    # -------------------------

    if np.random.rand() < 0.5:

        change_point = np.random.randint(
            seq_len // 3,
            2 * seq_len // 3
        )

        post_phase = np.random.uniform(
            0,
            2 * np.pi
        )

        post_frequency = np.random.uniform(
            1.0,
            6.0
        )

        post_amplitude = np.random.uniform(
            0.5,
            1.0
        )

        second_half_t = t[change_point:]

        signal[change_point:] = (
            post_amplitude
            * np.sin(
                2 * np.pi
                * post_frequency
                * second_half_t
                + post_phase
            )
        )

    # -------------------------
    # Small observation noise
    # -------------------------

    noise = np.random.normal(
        0,
        NOISE_STD,
        size=seq_len
    )

    signal += noise

    return signal.astype(np.float32)


# =========================
# Dataset Generation
# =========================

def generate_dataset():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for i in tqdm(
        range(NUM_SAMPLES),
        desc="Generating hard dataset"
    ):

        signal = generate_signal(
            SEQ_LEN
        )

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

    print(
        f"\nHard dataset generated in: "
        f"{OUTPUT_DIR}"
    )
