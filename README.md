# S-Mamba

**S-Mamba** is an experimental Mamba-based architecture that explores a different way of organizing recurrent state over long sequences.

Instead of treating the recurrent state only as something that is continuously updated, S-Mamba introduces a simple internal cycle:

```text
Input
  ↓
Mamba representation
  ↓
Distance / Support
  ↓
State admission
  ↓
Energy
  ↓
Pressure accumulation
  ↓
Release decision
  ↓
State consolidation
  ↓
New state
  ↓
Continue
```

The goal of the current prototype is to investigate whether **state-dependent support, accumulated pressure, and repeated state consolidation** can improve long-context behavior while keeping the model close to a small Mamba-style architecture.

> **Research status:** Experimental prototype. Results should not be interpreted as a general claim that S-Mamba outperforms all Mamba implementations.

---

## Why S-Mamba?

Mamba already provides an efficient recurrent mechanism for processing long sequences while maintaining a fixed-size internal state.

S-Mamba explores an additional question:

> **Can the state itself have an internal accumulation-and-release process?**

The current prototype uses the relationship between the incoming representation and the existing state to produce a distance signal.

That distance is used by Support and Energy:

```text
Incoming information
        ↓
Compare with current state
        ↓
     Distance
       ↙   ↘
  Support   Energy
     ↓        ↓
State      Pressure
admission     ↓
     ↓      Release
     └───────┬──────
             ↓
       Consolidated h
             ↓
        next timestep
```

---

## Overall Pipeline

The complete high-level pipeline is:

```text
Input
  ↓
Input Projection
  ↓
Depthwise Causal Convolution
  ↓
Gate
  ↓
Generate Δ, B, C
  ↓
┌─────────────────┐
│  S-Mamba Block  │
└─────────────────┘
  ↓
State readout
  ↓
Output Projection
  ↓
Output sequence
```

---

## Core Mechanisms

### Support

Support measures the relationship between the incoming representation and the current state.

$$
d_t = |h_{t-1}-x_t|
$$

and:

$$
S_t=e^{-\alpha d_t}
$$

Support then actively modifies the incoming contribution to the state.

This makes distance part of the state-admission mechanism rather than merely an inspection statistic.

---

### Energy and Pressure

Energy is derived from the same relational distance:

$$
E_t=\frac{1}{2}\operatorname{mean}(d_t^2)
$$

Energy is accumulated into pressure over time.

Pressure is then used to determine when a release event should occur.

Importantly, **the current implementation does not minimize Energy**.

Energy is used as an internal pressure signal:

```text
Distance
   ↓
Energy
   ↓
Pressure
   ↓
Release
```

---

### Release and Consolidation

When pressure reaches the release region, S-Mamba performs a hard release decision using a straight-through mechanism.

The current prototype consolidates the state by averaging across the internal state dimension:

$$
new_h =
\operatorname{mean}_{d_{state}}(h)
$$

The state shape is preserved so that recurrence can continue normally.

This is intentionally a simple prototype mechanism. More sophisticated consolidation methods are possible in future versions.

---

## Native Mamba Baseline

The primary comparison is against a **small native Mamba-style baseline with the same general experimental setup**.

This is important:

> **The baseline is one specific Mamba implementation.**

Different Mamba versions, implementations, dimensions, kernels, initialization methods, training procedures, and hardware can produce different results.

They should **not** be interpreted as S-Mamba outperforming every Mamba variant.

---

### Parameter Count

The two models were deliberately kept very close in size.

| Model        | Parameters |
| ------------ | ---------: |
| Native Mamba |     64,387 |
| S-Mamba      |     64,450 |

---

### Test Results

In particular run:

Sequence length 10000.

Device: cuda (On Google Colab)

```text
Normal Mamba
Parameters: 64387
Epoch 01/10 | Train MSE: 0.315914 | Test MSE: 0.233302
Epoch 02/10 | Train MSE: 0.309436 | Test MSE: 0.215854
Epoch 03/10 | Train MSE: 0.248930 | Test MSE: 0.173077
Epoch 04/10 | Train MSE: 0.495278 | Test MSE: 0.154533
Epoch 05/10 | Train MSE: 0.226316 | Test MSE: 0.188908
Epoch 06/10 | Train MSE: 0.262229 | Test MSE: 0.197991
Epoch 07/10 | Train MSE: 0.267705 | Test MSE: 0.196383
Epoch 08/10 | Train MSE: 0.259101 | Test MSE: 0.183721
Epoch 09/10 | Train MSE: 0.229699 | Test MSE: 0.147931
Epoch 10/10 | Train MSE: 0.162363 | Test MSE: 0.092045
```
```text
S-Mamba
S-Mamba parameters: 64450
Epoch 01 | Train MSE: 0.470891 | Test MSE: 0.349138
Epoch 02 | Train MSE: 0.358359 | Test MSE: 0.221367
Epoch 03 | Train MSE: 0.282800 | Test MSE: 0.170128
Epoch 04 | Train MSE: 0.190714 | Test MSE: 0.098630
Epoch 05 | Train MSE: 0.183269 | Test MSE: 0.119663
Epoch 06 | Train MSE: 0.154639 | Test MSE: 0.075211
Epoch 07 | Train MSE: 0.113967 | Test MSE: 0.059817
Epoch 08 | Train MSE: 0.094185 | Test MSE: 0.051473
Epoch 09 | Train MSE: 0.073875 | Test MSE: 0.036857
Epoch 10 | Train MSE: 0.052223 | Test MSE: 0.032958
```

S-Mamba produced approximately **64% lower test MSE** at epoch 10.

However, this is **one run on one synthetic task**. It is evidence that the mechanism is worth investigating, not a general performance claim.

---

### Benchmark Limitation

The baseline used here is a **native Mamba implementation**.

Performance can vary substantially depending on:

* Mamba version
* State size
* Model width
* Number of layers
* Dataset
* Sequence length
* Optimizer
* Hardware
* Output head

Therefore:

> **The reported numbers are implementation-specific experimental results.**

A stronger evaluation would test S-Mamba against multiple Mamba variants, multiple datasets, etc.

---

## Inspection Mode

S-Mamba includes an inspection mode for examining the internal behavior of the model.

The model can expose:

* Support history
* Distance history
* Energy
* Pressure
* Release decisions

This allows researchers to investigate not only the final prediction but also **what the internal state mechanism is doing during a long sequence**.

The inspection outputs can be used to visualize where pressure rises, where releases occur, and how state behavior changes between layers.

This is particularly important for S-Mamba because the research question concerns **internal state dynamics**, not only final accuracy.

---

## Current Limitations

S-Mamba is still an experimental prototype.

### 1. Synthetic dataset

The current results use continuous-wave data.

They do not establish performance on:

* Natural language
* Vision
* Speech
* Real-world time series
* Large-scale benchmarks

---

### 2. No semantic embeddings

The current wave representation can be directly mapped to numerical arrays, so embeddings are intentionally disabled.

Future token/semantic experiments will require an embedding or representation layer.

Because distance is part of the architecture, changing the representation space may also require revisiting the Support and Energy formulation.

---

### 3. Simple consolidation

The current release mechanism uses:

$$
mean(h)
$$

to consolidate the state.

This is intentionally simple and may remove some internal state-specific information.

It should therefore be treated as a baseline consolidation mechanism rather than a final solution.

---

### 4. Implementation speed

The current implementation performs the recurrent sequence loop explicitly.

This is useful for experimentation and inspection but is not yet optimized as a production implementation.

Long training runs can therefore be significantly slower than highly optimized Mamba implementations. (~3X slower)

---

### 5. Limited evaluation

The current results are based on a small synthetic experiment.

More extensive evaluation is required before making broader claims.

---

## Future Research Direction

The current implementation deliberately leaves several questions open.

One of the most interesting is what can be done with the sequence of state-derived representations produced throughout the recurrent process.

Currently:

$$
Y=[y_1,y_2,\ldots,y_L]
$$

is still returned as a sequence.

Future research may investigate whether these evolving representations can become higher-level memory units, adaptive chunks, or inputs to another selective mechanism.

The current project does not prescribe a particular future architecture.

---

## Project Status

S-Mamba is currently a **research prototype**.

The current phase focuses on establishing whether:

$$
\boxed{
\text{Distance}
\rightarrow
\text{Support}
\rightarrow
\text{Energy}
\rightarrow
\text{Pressure}
\rightarrow
\text{Release}
\rightarrow
\text{State Consolidation}
}
$$

can provide useful behavior inside a Mamba-style recurrent architecture.

The results are promising enough to motivate further experiments, but broader conclusions require larger datasets, additional Mamba baselines, and real-world tasks.
