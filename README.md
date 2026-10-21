# S-Mamba (Adv1)

**S-Mamba ADV1** is an experimental branch of S-Mamba exploring how energy, state-dependent support, and gated admission can influence recurrent state updates.

The idea is simple: instead of allowing incoming information to contribute to the recurrent state without additional control, ADV1 introduces mechanisms that consider the relationship between the incoming representation and the existing state.

Our current goal is to explore how these mechanisms behave and whether they can contribute to S-Mamba's broader vision.

> **Research status: Testing phase!** ADV1 is a work in progress. We're experimenting with its internal mechanisms, understanding their behavior, and exploring what might work next. This branch is part of the research journey, not the final destination. Contributions, ideas, and constructive discussions are welcome!

---

## Why S-Mamba ADV1?

S-Mamba's broader vision is **"Not Overflow Context"** — exploring ways to manage contextual information as a sequence evolves.

Processing long sequences is only one part of that challenge. A model must also preserve important information and relationships as new information arrives.

ADV1 explores one piece of this puzzle: **Can the relationship between incoming information and the existing state help control how new information enters that state?**

To investigate this, ADV1 introduces three mechanisms:

* **Support:** Measures the relationship between the incoming representation and the previous state.
* **Energy:** Calculates a signal from the difference between them.
* **Admission Gate:** Uses an energy-informed representation to regulate the incoming contribution.

These mechanisms provide a starting point for investigating state management. There's still plenty to explore, and we're keeping the implementation experimental while we learn from it.

---

## Architecture Overview

```text
Input Sequence
      ↓
Input Projection
      ↓
Mamba Representation
      ↓
Depthwise Causal Convolution
      ↓
Generate Δ, B, C
      ↓
┌─────────────────────────┐
│      S-Mamba Block      │
│                         │
│  Previous State (h)     │
│          ↓              │
│   Distance / Support    │
│          ↓              │
│        Energy           │
│          ↓              │
│         SLayer          │
│          ↓              │
│     Admission Gate      │
│          ↓              │
│      State Update       │
└─────────────────────────┘
      ↓
Output Projection
      ↓
Output Sequence
```

---

## Core Mechanisms

### 1. Support

Support measures the element-wise distance between the incoming representation and the previous state.

$$
d_t = |h_{t-1} - x_t|
$$

$$
S_t = e^{-\alpha d_t}
$$

Smaller distances produce higher support, while larger distances produce lower support.

Support directly modulates the incoming contribution to the state:

$$
I_t^{supported} = S_t \odot I_t
$$

This lets us investigate whether state-dependent support can help regulate incoming information.

### 2. Energy

Energy is calculated from the squared difference between the previous state and the incoming representation:

$$
E_t = \frac{1}{2}\text{mean}
\left[(h_{t-1}-x_t)^2\right]
$$

The resulting scalar is used as an internal signal.

**Energy is not directly minimized as a loss function.** Instead, it influences the admission mechanism through the SLayer.

### 3. SLayer & Admission Gate

The SLayer projects energy into the internal representation space and combines it with the incoming representation:

$$
c_t = x_t + W_E E_t + b_E
$$

The admission gate then calculates:

$$
g_t = \sigma(W_g c_t + b_g)
$$

The incoming contribution is modulated by both support and admission:

$$
I_t^{admitted} = g_t \odot S_t \odot I_t
$$

The resulting contribution participates in the recurrent state update.

This is the central experiment in ADV1: allowing an energy-informed signal and state-dependent support to influence how incoming information contributes to the state.

---

## Why Is ADV1 Still in the Testing Phase?

S-Mamba's long-term vision is not simply to process long sequences. It's to explore how a model can manage evolving context without losing important information or relationships.

ADV1 is an early step toward investigating that broader question.

### What We Have So Far

* ✅ A working Mamba-style recurrent architecture.
* ✅ State-dependent support.
* ✅ Energy-informed admission.
* ✅ Successful GPU training on the current synthetic task.
* ✅ A baseline for future experiments.

### What We're Still Exploring

**Contextual importance:** Distance doesn't necessarily tell us whether information is important. A representation far from the current state could contain something essential.

**Information retention:** We still need dedicated experiments to evaluate whether important information remains recoverable after many intervening timesteps.

**Context management:** ADV1 regulates incoming state contributions, but it doesn't yet explicitly detect contextual overflow or reorganize accumulated information.

**Mechanism effectiveness:** Further experiments will help us understand how support, energy, and admission interact and where they might be useful.

These are open research questions, not reasons to stop experimenting. The purpose of this branch is to help us investigate them.

---

## Experimental Results

ADV1 was tested against WaveMamba on a small synthetic dataset.

### Setup

| Setting         | Value              |
| --------------- | ------------------ |
| Dataset         | 100 samples        |
| Training / Test | 80 / 20            |
| Sequence length | 1,000              |
| Epochs          | 10                 |
| Device          | CUDA               |
| Environment     | Google Colab       |
| Loss            | Mean Squared Error |

### Results

* [Seq 100](test_baseline/test_seq_100.txt)
* [Seq 1,000](test_baseline/test_seq_1k.txt)
* [Seq 10,000](test_baseline/test_seq_10k.txt)

These results represent **one experiment on a small synthetic dataset**, not a general performance claim. 
The models also differ in parameter count, so additional controlled testing would be needed to isolate the effects of the new mechanisms.

---

## Memory and Hardware

The current implementation uses an explicit recurrent loop over sequence timesteps.

During local CPU testing, long-sequence training encountered substantial memory pressure and process termination. Training subsequently completed using CUDA on Google Colab.

**Google Colab with a GPU is recommended for the current experimental setup.**

Memory requirements depend on sequence length, batch size, model dimensions, and available hardware. The implementation is currently optimized for experimentation rather than production deployment.

---

## Current Limitations

* **Small dataset:** Results come from only 100 synthetic samples.
* **Limited evaluation:** More datasets, random seeds, and controlled comparisons are needed.
* **Context retention:** Dedicated experiments are needed to evaluate long-range information preservation.
* **Distance-based support:** Representational distance does not necessarily indicate contextual importance.
* **Explicit recurrence:** The current implementation can be computationally expensive for long sequences.
* **Experimental mechanisms:** The benefits and limitations of energy-informed admission remain under investigation.

---

## Future Research

Some directions worth exploring include:

* Testing information retention across increasingly long sequences.
* Investigating contextual relationships beyond simple representational distance.
* Comparing admission mechanisms through controlled experiments.
* Exploring methods for reorganizing and preserving accumulated context.
* Evaluating performance across additional datasets and baselines.

These directions will evolve as we learn more from the current implementation.

---

## Project Status

**S-Mamba ADV1 is an ongoing experiment, and we're still learning from it!**

This branch explores how energy and state-dependent support can influence recurrent state updates. It has successfully completed training on our current synthetic task, giving us a foundation to continue experimenting.

Our bigger ambition remains **"Not Overflow Context"** — investigating how models can manage evolving information while preserving what matters.

ADV1 doesn't solve that challenge yet, but it gives us another mechanism to explore along the way.

We're keeping this branch experimental while we investigate its behavior and decide what to try next.

**AI research is a journey of experiments, questions, and discoveries. We're just getting started.**

If you're interested in alternative recurrent architectures, internal state dynamics, or experimental AI research, you're welcome to follow along and share your ideas!
