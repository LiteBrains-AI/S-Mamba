# S-Mamba — Model Overview

This document describes the internal components of the current S-Mamba prototype.

The model is built around a Mamba-style recurrent state, with additional mechanisms for **Support, relational Energy, Pressure, Release, and state consolidation**.

The current prototype contains four model-specific components:

```text
model/
├── embeddings.py
├── support.py
├── energy.py
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

Energy converts the relational distance into an accumulated internal pressure signal.

The current prototype uses:

$$
E_t=\frac{1}{2}\operatorname{mean}(d_t^2)
$$

Energy is **not currently used as an optimization objective**.

S-Mamba does not perform:

$$
\min_h E(h,x)
$$

Instead, Energy acts as an internal measurement of structural pressure.

The pressure evolves over time:

$$
P_t=(1-\lambda)P_{t-1}+E_t
$$

where \(\lambda\) is the pressure decay factor.

For the current prototype, the default decay is zero, meaning pressure accumulates directly.

The pressure is then converted into a release probability:

$$
R_t=\sigma(P_t-C)
$$

where \(C\) is the capacity threshold.

A straight-through binary decision converts this into a hard release event during the forward pass while retaining a surrogate gradient during backpropagation.

### Energy → Pressure → Release

The complete mechanism is:

```text
Distance
   ↓
Energy
   ↓
Accumulated Pressure
   ↓
Release Probability
   ↓
Binary Release Decision
```

Energy therefore acts as a **controller signal**, rather than as a loss function.

---

## 4. `s_mamba.py`

`s_mamba.py` contains the main S-Mamba architecture.

The current block retains the main structure of a Mamba-style selective state-space model:

```text
Input
 ↓
Input projection
 ↓
Depthwise causal convolution
 ↓
Gate
 ↓
Generate Δ, B, C
 ↓
Selective state update
 ↓
Support-modulated admission
 ↓
Energy / Pressure
 ↓
Release
 ↓
State consolidation
 ↓
State readout
 ↓
Output projection
```

## State

The recurrent state has the shape:

$$
h\in\mathbb{R}^{B\times d_{inner}\times d_{state}}
$$

The state is initialized to zero for each sequence.

At every timestep, the existing state is preserved as:

```text
h_before
```

before processing the new input.

---

## Support-Modulated State Update

The incoming signal first interacts with the existing state through Support.

The model calculates:

$$
d_t=|h_{t-1}-x_t|
$$

and:

$$
S_t=e^{-\alpha d_t}
$$

The incoming contribution is then:

$$
I_t=x_t\odot dB_t
$$

and Support modifies this contribution:

$$
I_t'=S_t\odot I_t
$$

The state becomes:

$$
h_t=A_th_{t-1}+I_t'
$$

This makes distance part of the actual state-admission process.

---

## Release and State Consolidation

After the state update, Energy produces pressure and a release decision.

When no release occurs:

$$
h_t'=h_t
$$

When a release occurs, the current prototype consolidates the state using a mean over the internal state dimension:

$$
\tilde h_t=
\operatorname{mean}_{d_{state}}(h_t)
$$

The result is expanded back to the original state shape so that recurrence can continue.

Therefore:

$$
h_t'=
\begin{cases}
h_t,&R_t=0\\
\tilde h_t,&R_t=1
\end{cases}
$$

The consolidated state becomes the state used by subsequent timesteps.

This is intentionally a simple first implementation of state consolidation. It is **not intended to be the final consolidation mechanism**.

---

## Internal State Transition

The important part of S-Mamba is not simply the presence of another gate.

A release creates a state transition:

```text
Accumulated state
       ↓
   pressure
       ↓
   release event
       ↓
  consolidation
       ↓
     new_h
       ↓
future state evolution
```

Therefore, the consequence of an internal release persists into later timesteps.

This gives the model a repeated internal cycle:

$$
state
\rightarrow
interaction
\rightarrow
pressure
\rightarrow
event
\rightarrow
new\ state
\rightarrow
interaction
$$

This is one reason the architecture can be viewed as having an **RL-like internal state-transition structure**, although the current model is not a conventional reinforcement-learning algorithm.

There is no external reward, policy network, or Q-function. The analogy is specifically about **decisions producing state changes that influence subsequent rounds**.

---

## State Readout

After the release decision, the resulting state is read using the generated \(C_t\) parameter:

$$
y_t=\sum_j h_{t,j}C_{t,j}
$$

The timestep outputs are collected:

$$
Y=[y_1,y_2,\ldots,y_L]
$$

and passed through the output projection.

The current implementation therefore still exposes a sequence of state-derived outputs.

This is intentional for the current prototype.

Future versions may investigate how these evolving state representations can be further aggregated, recalled, or processed as higher-level units.

---

## Current Design Philosophy

The current S-Mamba prototype deliberately keeps the mechanism small.

The central chain is:

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
\text{Consolidated State}
}
$$

The objective of the current phase is to investigate whether these mechanisms can alter how a Mamba-style recurrent state absorbs and organizes long sequences.

The prototype does **not** yet attempt to solve every aspect of memory, recall, embeddings, or adaptive chunking.

Those are natural directions for subsequent experiments rather than assumptions built into the current implementation.
