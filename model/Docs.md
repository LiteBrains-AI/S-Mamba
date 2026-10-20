# S-Mamba ADV1 — Architecture Overview

## Overview

S-Mamba is an experimental architecture exploring modifications to Mamba-style selective state-space models.

The current ADV1 branch investigates a learned admission mechanism that combines token representations, relational energy, and support-based state interaction.

Rather than relying on an accumulated pressure signal and discrete release events, ADV1 introduces a trainable gate to control how much incoming information enters the recurrent state.

The primary objective is to investigate how these mechanisms affect state updates, information admission, and sequence modeling.

**S-Mamba ADV1 is an experimental architecture, not a claim of general superiority over Mamba.**

---

## Core Mechanism

The current architecture combines four principal components:

* **Support:** Measures the relationship between the incoming representation and the existing state.
* **Energy:** Calculates a scalar relational measurement.
* **SLayer:** Projects the energy measurement into the token feature space and combines it with the token representation.
* **Admission Gate:** Learns how much incoming information should be admitted into the recurrent state.

The overall mechanism is:

```text
Input
  |
  v
Mamba Representation
  |
  +----------------------------+
  |                            |
  v                            v
Token Representation       Existing State
  |                            |
  |                            v
  |                          Energy
  |                            |
  +-------------+--------------+
                |
                v
              SLayer
                |
                v
         Admission Gate
                |
                v
         Learned Gate Values
                |
                +------------------+
                |                  |
                v                  v
             Support            Incoming
                |                  |
                +--------+---------+
                         |
                         v
                  State Update
                         |
                         v
                    State Readout
                         |
                         v
                  Output Projection
```

The central idea is to retain the existing recurrent state-space computation while introducing an additional mechanism that controls the admission of incoming information.

---
## 1. Support — `support.py`

The Support module measures the relationship between the incoming token representation and the existing recurrent state.

Rather than directly changing the state, Support produces a modulation signal that controls the incoming contribution during the recurrent update.

### Distance Calculation

Given the existing state:

$$
h\in\mathbb{R}^{B\times d_{\text{inner}}\times d_{\text{state}}}
$$

and the current token representation:

$$
x\in\mathbb{R}^{B\times d_{\text{inner}}}
$$

the token representation is expanded along the state dimension.

The absolute distance is calculated as:

$$
\boxed{
d=|h-x|
}
$$

The resulting distance tensor has shape:

$$
d\in\mathbb{R}^{B\times d_{\text{inner}}\times d_{\text{state}}}
$$

### Support Calculation

Support is calculated using an exponential decay function:

$$
\boxed{
S=e^{-\alpha d}
}
$$

where $\alpha$ controls how strongly distance affects the support values.

For the current implementation:

$$
S\in(0,1]
$$

Smaller distances produce higher support values, while larger distances produce lower support values.

Support is calculated element-wise, preserving the feature and state dimensions.

### Role in ADV1

Support modulates the incoming contribution before it is added to the recurrent state.

The incoming contribution is:

$$
I_t=x_t\odot dB_t
$$

Support modifies it as follows:

$$
I_t'=S_t\odot I_t
$$

ADV1 additionally applies the learned admission gate:

$$
\boxed{
I_t''=G_t\odot S_t\odot I_t
}
$$

The recurrent state is then updated using:

$$
\boxed{
h_t=\bar A_t\odot h_{t-1}+I_t''
}
$$

where $\bar A_t$ is the discretized state-transition factor.

### Support and the Admission Gate

Support and the admission gate serve distinct purposes.

* **Support** determines the element-wise modulation derived from the distance between the incoming representation and the existing state.
* **Admission Gate** learns feature-wise admission values using the combined token and energy representation.

Both mechanisms influence the incoming contribution, but neither independently determines the complete recurrent state update.

---

## 2. Energy — `energy.py`

The Energy module measures the relational difference between the current token representation and the existing recurrent state.

For the current implementation, energy is a scalar for each batch element.

Given:

$$
h_{t-1}\in\mathbb{R}^{B\times d_{\text{inner}}\times d_{\text{state}}}
$$

and:

$$
x_t\in\mathbb{R}^{B\times d_{\text{inner}}},
$$

the token representation is expanded along the state dimension.

The relational difference is:

$$
d_t=h_{t-1}-x_t
$$

where the token representation is broadcast across the state dimension.

Energy is calculated as:

$$
\boxed{
E_t=\frac{1}{2}
\operatorname{mean}(d_t^2)
}
$$

The mean is taken across the feature and state dimensions, producing:

$$
E_t\in\mathbb{R}^{B}
$$

### Purpose of Energy

In ADV1, energy is an internal measurement rather than an optimization objective.

The model does not directly minimize:

$$
\min_h E(h,x)
$$

Instead, energy provides an additional signal to the admission mechanism.

The current implementation uses scalar energy because it is sufficient for the present experimental setup.

Future experiments may investigate vector-wise energy factors that preserve more feature-level information, particularly when working with embedding-based inputs.

---

## 3. SLayer — `s_layer.py`

SLayer combines the incoming token representation with the scalar energy measurement.

Because energy is scalar while the token representation contains multiple features, energy is first projected into the token feature space.

The energy projection is:

$$
E_t'=W_E E_t+b_E
$$

where:

$$
E_t'\in\mathbb{R}^{B\times d_{\text{inner}}}
$$

The combined representation is then:

$$
\boxed{
u_t=x_t+E_t'
}
$$

Here:

* \(x_t\) is the incoming token representation.
* \(E_t'\) is the projected energy representation.
* \(u_t\) is the combined representation supplied to the admission gate.

### Why Combine Token and Energy?

The token representation describes the incoming signal, while energy provides a measurement of its relationship with the existing state.

Combining them allows the admission gate to consider both signals when calculating its output.

However, the addition does not guarantee that higher energy produces higher gate values. The learned projection and subsequent gate determine how these signals influence the result.

### Future Experiments

Future experiments may investigate mechanisms that explicitly increase gate values in response to higher energy.

Such behavior is a potential design direction, not a guaranteed property of the current implementation.

---

## 4. Admission Gate

The admission gate is the primary architectural change introduced in ADV1.

Unlike the previous pressure-release mechanism, ADV1 uses a learned gate to modulate the incoming contribution at each timestep.

The combined representation is passed through a trainable linear layer:

$$
a_t=W_Gu_t+b_G
$$

The gate values are obtained using the sigmoid activation:

$$
\boxed{
G_t=\sigma(a_t)
}
$$

where:

$$
G_t\in(0,1)^{B\times d_{\text{inner}}}
$$

The gate is feature-wise. Its values are broadcast across the state dimension when applied to the incoming contribution.

### Interpretation

The gate controls how strongly the incoming contribution participates in the recurrent state update.

Conceptually:

* Lower gate values suppress more of the incoming contribution.
* Higher gate values admit more of the incoming contribution.

These interpretations describe the direct effect of the gate on the incoming term. They do not imply that every feature will learn a particular admission pattern.

The gate is learned through the task's training objective.

### Important Distinction

The admission gate is not a replacement for the entire state-space mechanism.

The underlying selective state update remains responsible for recurrent state evolution. ADV1 introduces an additional learned control over the incoming contribution.

---

## 5. Support-Modulated State Update

Support measures the relationship between the incoming token representation and the existing state.

The distance is calculated as:

$$
d_t=|h_{t-1}-x_t|
$$

Support is then computed as:

$$
\boxed{
S_t=e^{-\alpha d_t}
}
$$

where \(\alpha\) controls the sensitivity of support to the measured distance.

For the current implementation:

$$
S_t\in(0,1]
$$

The incoming contribution is:

$$
I_t=x_t\odot dB_t
$$

where \(dB_t\) represents the discretized input-dependent state contribution.

Support modulates this incoming contribution:

$$
I_t'=S_t\odot I_t
$$

ADV1 additionally applies the learned admission gate:

$$
\boxed{
I_t''=G_t\odot I_t'
}
$$

Combining the two mechanisms gives:

$$
I_t''=G_t\odot S_t\odot I_t
$$

The recurrent state update becomes:

$$
\boxed{
h_t=\bar A_t\odot h_{t-1}
+G_t\odot S_t\odot I_t
}
$$

where:

$$
\bar A_t=e^{\Delta_t A}
$$

is the discretized state-transition factor.

The multiplication involving \(G_t\), \(S_t\), and \(I_t\) uses broadcasting across the state dimension as required by the implementation.

### Why Keep Both Support and the Gate?

Support and the admission gate have different roles.

| Component        | Role                                                                                     |
| ---------------- | ---------------------------------------------------------------------------------------- |
| Support          | Modulates incoming information according to its relationship with the existing state.    |
| Admission Gate   | Learns feature-wise admission values using the combined token and energy representation. |
| State Transition | Evolves the existing recurrent state.                                                    |
| Energy           | Supplies a scalar relational measurement to SLayer.                                      |

Both Support and the admission gate affect the incoming contribution.

This creates a combined admission mechanism without removing the underlying recurrent dynamics.

---

## 6. Main Architecture — `s_mamba.py`

`s_mamba.py` contains the main S-Mamba block and the top-level model.

The architecture retains the principal components of a Mamba-style selective state-space model while introducing the ADV1 admission mechanism.

The current processing sequence is:

```text
Input
  |
  v
Input Projection
  |
  v
Depthwise Causal Convolution
  |
  v
Generate Δ, B, C
  |
  +-----------------------------+
  |                             |
  v                             v
Token Representation        Existing State
  |                             |
  |                             v
  |                           Energy
  |                             |
  +--------------+--------------+
                 |
                 v
               SLayer
                 |
                 v
          Admission Gate
                 |
                 v
       Support-Modulated Admission
                 |
                 v
           Recurrent Update
                 |
                 v
            State Readout
                 |
                 v
          Output Projection
```

### Recurrent State

The recurrent state has the shape:

$$
\boxed{
h\in\mathbb{R}^{B\times d_{\text{inner}}\times d_{\text{state}}}
}
$$

The state is initialized to zero for each sequence.

Before processing the incoming contribution at each timestep, the existing state is retained as:

```python
h_before = h
```

This provides the state used for relational measurements and the subsequent recurrent update.

### State Evolution

At each timestep, the model:

1. Obtains the current token representation and selective state-space parameters.
2. Calculates energy using the existing state and token representation.
3. Combines token features and projected energy through SLayer.
4. Computes the learned admission gate.
5. Calculates the Support-modulated incoming contribution.
6. Applies the admission gate to that contribution.
7. Updates the recurrent state.
8. Reads the resulting state into an output representation.

The updated state is carried into subsequent timesteps.

---

## What Changed from the Previous Prototype?

The previous prototype used an energy based *pressure-release* mechanism.
The previous prototype used accumulated pressure to determine when to release and consolidate the state.

ADV1 instead investigates how a learned, feature-wise gate can control the incoming contribution at every timestep.

The current ADV1 implementation does not use the previous pressure accumulation, release decision, or release-triggered state consolidation in its recurrent update.

---

## State Readout and Output Projection

After updating the recurrent state, the model reads the state using the generated \(C_t\) parameter.

The readout is:

$$
\boxed{
y_t=\sum_j h_{t,:,j}C_{t,j}
}
$$

Here, the summation is over the internal state dimension.

Each timestep produces a representation:

$$
y_t\in\mathbb{R}^{B\times d_{\text{inner}}}
$$

The timestep outputs are collected into a sequence:

$$
Y=\operatorname{stack}(y_1,y_2,\ldots,y_L)
$$

with shape:

$$
Y\in\mathbb{R}^{B\times L\times d_{\text{inner}}}
$$

The output projection then transforms the feature dimension:

$$
\boxed{
Y'=YW_O+b_O
}
$$

producing:

$$
Y'\in\mathbb{R}^{B\times L\times d_{\text{model}}}
$$

The projection is applied independently to each timestep's feature vector. It does not average the sequence dimension.

The recurrent state and the readout representation have different roles: the recurrent state carries information through the timestep updates, while the readout provides the representation exposed to the next processing stage.

This design remains part of the current prototype.

Future experiments may investigate alternative readout or routing mechanisms, but those are not required for the current ADV1 implementation.

---

## Design Philosophy

ADV1 deliberately keeps the admission mechanism relatively small.

The main experimental chain is:

$$
\boxed{
\text{Relational Distance}
\rightarrow
\text{Energy}
\rightarrow
\text{SLayer}
\rightarrow
\text{Admission Gate}
}
$$

In parallel, the Support mechanism modifies the incoming contribution before it enters the recurrent state.

The complete update combines:

$$
\boxed{
\text{State Transition}
+
\text{Support-Modulated Admission}
+
\text{Learned Gate}
}
$$

The objective is to investigate whether a learned admission gate, informed by token representations and relational energy, can improve the behavior of a Mamba-style recurrent state update.

The current prototype does not yet establish that this mechanism improves accuracy, stability, memory retention, or computational efficiency.

Those questions must be answered experimentally.

---

## Current Scope and Future Experiments

The current ADV1 phase focuses on:

* Evaluating the learned admission gate.
* Measuring training stability.
* Comparing results with the previous prototype and an appropriate baseline.
* Understanding the interaction between Support, Energy, and the recurrent state update.
* Measuring the computational cost of the additional parameters and operations.

Potential future directions include:

* Investigating vector-wise energy representations.
* Exploring explicit relationships between energy magnitude and gate values.
* Evaluating alternative admission mechanisms.
* Investigating alternative state readout or routing mechanisms.
* Testing on more complex sequences and embedding-based inputs.

These are future research directions, not capabilities that the current implementation is assumed to possess.

**The central question of ADV1 is whether learned, energy-informed admission can provide a useful alternative to the previous pressure-release mechanism.**
