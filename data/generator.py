from pathlib import Path

import torch
from torch.utils.data import Dataset
from collections import defaultdict
from tqdm.auto import tqdm


NUM_SAMPLES = 10000
SEQ_LEN = 100
VALUE_RANGE = 30

MIN_NGRAM = 2
MAX_NGRAM = 5

OUTPUT_PATH = Path("dataset/concept_sequences.pt")


def find_repeated_ngrams(
    sequence,
    min_n=MIN_NGRAM,
    max_n=MAX_NGRAM,
):
    seq_len = len(sequence)

    selected_positions = set()
    found_ngrams = []

    for n in range(min_n, max_n + 1):
        occurrences = defaultdict(list)

        for start in range(seq_len - n + 1):
            gram = tuple(sequence[start:start + n])
            occurrences[gram].append(start)

        for gram, starts in occurrences.items():
            if len(starts) < 2:
                continue

            found_ngrams.append({
                "n": n,
                "gram": gram,
                "starts": starts,
            })

            for start in starts:
                for pos in range(start, start + n):
                    selected_positions.add(pos)

    return selected_positions, found_ngrams


def generate_sample(
    seq_len=SEQ_LEN,
    value_range=VALUE_RANGE,
):
    sequence = torch.randint(
        low=0,
        high=value_range,
        size=(seq_len,),
    ).tolist()

    positions, found_ngrams = find_repeated_ngrams(sequence)

    mask = torch.zeros(
        seq_len,
        dtype=torch.long,
    )

    for pos in positions:
        mask[pos] = 1

    return {
        "sequence": torch.tensor(
            sequence,
            dtype=torch.long,
        ),
        "mask": mask,
        "ngrams": found_ngrams,
    }


def generate_dataset():
    samples = []

    for _ in tqdm(
        range(NUM_SAMPLES),
        desc="Generating dataset",
    ):
        samples.append(
            generate_sample()
        )

    return samples


def save_dataset(samples):
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "samples": samples,
            "config": {
                "num_samples": NUM_SAMPLES,
                "seq_len": SEQ_LEN,
                "value_range": VALUE_RANGE,
                "min_ngram": MIN_NGRAM,
                "max_ngram": MAX_NGRAM,
            },
        },
        OUTPUT_PATH,
    )

    print(f"Saved dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    samples = generate_dataset()
    save_dataset(samples)
