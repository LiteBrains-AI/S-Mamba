# S-Mamba Adv1 — Model Overview

This document describes the internal components of the current S-Mamba Adv1 prototype.

The model is built around a Mamba-style recurrent state, with additional mechanisms for **Support, relational Energy, Gating, and soft consolidation**.

The current prototype contains four model-specific components:

```text
model/
├── embeddings.py
├── support.py
├── energy.py
├── s_layer.py
└── s_mamba.py
```

---

## 1. `embeddings.py`

### Status: Currently inactive

The embedding component is reserved for future experiments where the input is represented as discrete tokens or other semantic representations.

The current dataset uses continuous numerical wave values that can already be represented directly as arrays. Therefore, adding an embedding layer at this stage would introduce an additional transformation without helping test the core S-Mamba mechanism.

The intended future pipeline is:

```text
Raw input
   ↓
Embedding / Representation
   ↓
S-Mamba
```

This becomes particularly important because S-Mamba uses **distance as an active signal**.

The Support mechanism currently evaluates the relationship between the incoming representation and the existing state:

$$
d_t = |h_{t-1} - x_t|
$$

Therefore, once embeddings are introduced, the representation space itself becomes important.

A future embedding system may need to be trained or adapted together with the distance-based mechanisms so that the resulting geometry is meaningful for Support and Energy.

---

## 2. `support.py`

Support is the first major modification to the normal Mamba state update.

Instead of allowing the incoming signal to enter the state without considering its relationship with the existing state, S-Mamba measures their distance.

For the current state:

$$
h \in \mathbb{R}^{B\times d_{inner}\times d_{state}}
$$

and incoming representation:

$$
x \in \mathbb{R}^{B\times d_{inner}}
$$

the incoming representation is expanded across the state dimension.

The distance is:

$$
d = |h-x|
$$

The current Support function is:

$$
S=e^{-\alpha d}
$$

where \(\alpha\) controls the sensitivity to distance.

The implementation returns both:

```text
support
distance
```

because distance is not merely an inspection value. It is subsequently used by the Energy mechanism.

### Role of Support

Support is intended to answer:

> **How compatible is the incoming information with the current state structure?**

It is deliberately different from a conventional learned gate.

A conventional gate typically controls signal magnitude directly.

Support instead derives a factor from the **relationship between the incoming information and the current state**.

The current state update therefore becomes:

$$
h_t=A_th_{t-1}+S_t\odot B_tx_t
$$

rather than simply:

$$
h_t=A_th_{t-1}+B_tx_t
$$

This makes Support an active part of state admission.

---

## 3. `energy.py`
...