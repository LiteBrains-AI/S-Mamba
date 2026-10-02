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
E_t=\frac{1}{2}\,\text{mean}(d_t^2)
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
new_h = \text{mean}_{d_{state}}(h)
$$

The state shape is preserved so that recurrence can continue normally.

This is intentionally a simple prototype mechanism. More sophisticated consolidation methods are possible in future versions.

---
