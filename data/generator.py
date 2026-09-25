import torch
from torch.utils.data import Dataset
from collections import defaultdict
from tqdm.auto import tqdm


# ============================================================
# Configuration
# ============================================================

NUM_SAMPLES = 10000
SEQ_LEN = 100
VALUE_RANGE = 30

MIN_NGRAM = 2
MAX_NGRAM = 5


# ============================================================
# 1. Find repeated n-grams
# ============================================================

def find_repeated_ngrams(
    sequence,
    min_n=MIN_NGRAM,
    max_n=MAX_NGRAM
):
    """
    Search for repeated contiguous n-grams.

    Searches:
        2-gram → 3-gram → 4-gram → 5-gram

    Returns:
        selected_positions:
            Set of positions participating in at least
            one repeated n-gram.

        found_ngrams:
            Detailed information about every repeated
            n-gram that was found.
    """

    seq_len = len(sequence)

    selected_positions = set()
    found_ngrams = []

    for n in range(min_n, max_n + 1):

        occurrences = defaultdict(list)

        # ----------------------------------------------------
        # Collect every occurrence of each n-gram
        # ----------------------------------------------------

        for start in range(seq_len - n + 1):

            gram = tuple(
                sequence[start:start + n]
            )

            occurrences[gram].append(start)

        # ----------------------------------------------------
        # Keep only n-grams appearing at least twice
        # ----------------------------------------------------

        for gram, starts in occurrences.items():

            if len(starts) < 2:
                continue

            found_ngrams.append({
                "n": n,
                "gram": gram,
                "starts": starts
            })

            # ------------------------------------------------
            # Mark every position belonging to this pattern
            # ------------------------------------------------

            for start in starts:

                for pos in range(
                    start,
                    start + n
                ):
                    selected_positions.add(pos)

    return selected_positions, found_ngrams


# ============================================================
# 2. Generate one sample
# ============================================================

def generate_sample(
    seq_len=SEQ_LEN,
    value_range=VALUE_RANGE
):
    """
    Generate:
        random sequence
        dynamic oracle mask
        repeated n-gram information
    """

    # --------------------------------------------------------
    # Random sequence
    # --------------------------------------------------------

    sequence = torch.randint(
        low=0,
        high=value_range,
        size=(seq_len,)
    ).tolist()

    # --------------------------------------------------------
    # Search 2-gram → 5-gram
    # --------------------------------------------------------

    positions, found_ngrams = find_repeated_ngrams(
        sequence,
        min_n=MIN_NGRAM,
        max_n=MAX_NGRAM
    )

    # --------------------------------------------------------
    # Build dynamic mask
    # --------------------------------------------------------

    mask = torch.zeros(
        seq_len,
        dtype=torch.long
    )

    for pos in positions:
        mask[pos] = 1

    return {
        "sequence": torch.tensor(
            sequence,
            dtype=torch.long
        ),
        "mask": mask,
        "ngrams": found_ngrams
    }


# ============================================================
# 3. Dataset
# ============================================================

class ConceptSequenceDataset(Dataset):

    def __init__(
        self,
        num_samples=NUM_SAMPLES,
        seq_len=SEQ_LEN,
        value_range=VALUE_RANGE
    ):

        self.samples = []

        for _ in tqdm(
            range(num_samples),
            desc="Generating samples"
        ):

            self.samples.append(
                generate_sample(
                    seq_len=seq_len,
                    value_range=value_range
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        return (
            sample["sequence"],
            sample["mask"]
        )


# ============================================================
# 4. Generate dataset
# ============================================================

dataset = ConceptSequenceDataset(
    num_samples=NUM_SAMPLES,
    seq_len=SEQ_LEN,
    value_range=VALUE_RANGE
)


print("\nDataset Created")


print("\nDataset size:", len(dataset))


# ============================================================
# 5. Inspect one sample
# ============================================================

sample = dataset.samples[0]

sequence = sample["sequence"]
mask = sample["mask"]
ngrams = sample["ngrams"]

print("\nSequence:")
print(sequence)

print("\nDynamic mask:")
print(mask)

print("\nMarked positions:")
print(mask.nonzero(as_tuple=True)[0])

print("\nNumber of marked positions:")
print(mask.sum().item())


# ============================================================
# 6. Show actual repeated n-grams
# ============================================================


print("\nRepeated N-Grams")


if len(ngrams) == 0:

    print("No repeated 2–5 grams found.")

else:

    for item in ngrams:

        print(
            f"{item['n']}-gram | "
            f"pattern={item['gram']} | "
            f"starts={item['starts']}"
        )

print("\n✅ Done! Dataset created and verified.")
