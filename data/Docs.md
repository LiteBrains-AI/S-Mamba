# Dataset Generation

S-Mamba currently uses a **synthetic continuous-wave dataset** designed to test long-sequence state behavior rather than semantic language understanding.

## Why Waves?

The current prototype is focused on testing the internal mechanism of S-Mamba:

* State accumulation
* Distance-based Support
* Energy / pressure accumulation
* Release events
* State consolidation
* Long-sequence processing

For this reason, we do not need a language embedding space yet.

The generated waves can be represented directly as numerical arrays, so each sample can be stored and processed as a NumPy array (`.npz`). This keeps the first experiments simple and lets us study the S-Mamba mechanism without introducing an additional embedding model or vocabulary layer.

> **Note:** This does not mean embeddings are unnecessary for future S-Mamba applications. Once semantic/token embeddings are introduced, the distance and Support mechanism will need to operate on the corresponding representation space because distance is an active part of the model.

---

## Dataset Generators

The dataset generation system contains two generators for different stages of development.

### `simple_generator.py`

`simple_generator.py` is intended for **initial testing and quick experiments**.

It was used to:

* Verify that the dataset pipeline
* Quickly generate small datasets
* Test model code
* Debug sequence handling
* Experiment with different sequence lengths

---

### `hard_generator.py`

`hard_generator.py` is intended for the **actual training and evaluation experiments**.

It generates the harder wave sequences used to test whether S-Mamba can maintain useful state behavior over long contexts.

The generated samples are stored as NumPy-compatible arrays, allowing them to be loaded efficiently during training without requiring an embedding pipeline.

The harder dataset is the primary dataset for reported S-Mamba experiments.

---

## `visualizer.py`

`visualizer.py` provides a simple way to inspect the generated dataset visually.

This is useful for checking:

* Whether the generated waves look correct
* Sequence structure
* Input/target relationships
* Whether changes to the generator actually increase difficulty
* Whether a generated dataset contains unexpected patterns

Visual inspection is especially useful before starting a long training run.

---

## Increasing Sequence Length

Developers are encouraged to experiment with longer sequences.

For example:

```text
1,000 → 2,000 → 5,000 → 10,000+
```

Longer sequences are particularly relevant to S-Mamba because the project is interested in how the recurrent state behaves as context becomes increasingly long.

However, **training cost grows substantially with sequence length**.

### ⚠️ Hardware Warning

Long-sequence training can require significant system memory and GPU VRAM.

It is highly recommended to use:

* A CUDA-capable GPU with sufficient VRAM, or
* Google Colab / another GPU-based environment

A normal PC may become extremely slow, run out of memory, or potentially crash when attempting to train very long sequences.

Start with shorter sequences when developing or debugging the model, then increase the sequence length gradually.

> **Do not increase sequence length dramatically without checking available RAM and VRAM first.**

---

## Why This Dataset Is Only Beginning

The wave dataset is intentionally simple.

Its purpose is **not to demonstrate semantic intelligence**. It provides a controlled environment for investigating the internal S-Mamba mechanism before introducing additional complexity.

Future experiments can replace the direct numerical input with *learned* embeddings or other representations.