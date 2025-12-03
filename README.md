# Traffic Engineering with Ising Models (Quantum Dice Challenge)

### Applying ORBIT to Network Congestion Optimisation Using Abilene + Repetita

## Overview

This project explores whether **Ising-model–based optimisation**, as enabled by **ORBIT** from **Quantum Dice**, can be used to solve real-world **traffic engineering (TE)** problems.
We benchmark ORBIT against two classical optimisation baselines:

* **MILP** — throughput maximisation
* **PF/ADMM** — proportional fairness
* **ORBIT** — sampling-based Ising optimisation (external-field Ising model)

The experiments are run on the **Abilene ISP topology** with **Repetita** demand matrices.

> **Note**: For confidentiality reasons, the ORBIT simulator **is not included** in this repository.
> Users must use their own licensed or provided ORBIT installation, or an alternative simulator.

---

## Challenge Motivation

This repository was developed as part of the **Quantum Dice Challenge**, investigating whether **probabilistic hardware and Ising models** can efficiently solve optimisation tasks that traditionally rely on linear/convex solvers.

Traffic engineering is a natural testbed because:

* routing decisions can be encoded as binary variables
* congestion can be translated into an energy function
* ORBIT allows fast exploration of routing configurations via sampling

---

## Requirements

This repository contains:

* helper utilities for loading topologies, demands, and evaluating routing
* MILP and PF/ADMM baseline solvers
* a full demonstration notebook showing ORBIT plugged into a TE pipeline

**Not included:**

* ORBIT simulator
* Repetita datasets

---

## Repository Structure

```
.
├── helpers/
│   ├── load_topology.py
│   ├── load_demands.py
│   ├── te_solvers.py
│   ├── plot_topology.py
│   └── <utilities>
│
├── notebooks/
│   └── demo.ipynb   # Main experiment (user must edit dataset paths here)
│
└── README.md
```

## Setup Instructions

### 1. **Install ORBIT or alternative simulator Separately**

Due to confidentiality and licensing, ORBIT is **not bundled** with this repository.


### 2. **Download Repetita Manually**

Users must download the Repetita dataset:

📎 [https://github.com/heal-research/repetita](https://github.com/heal-research/repetita)

and update the paths directly in:

```
notebooks/demo.ipynb
```

## Methods

### **MILP (Mixed-Integer Linear Programming)**

Maximises total flow while respecting link capacities.

### **PF/ADMM (Proportional-Fair Optimisation)**

Balances flows across demands by maximising:

[
\sum_k \log(f_k)
]

### **ORBIT (Ising Sampling)**

Represents routing as a binary Ising system:

[
E(s) = -h^\top s
]

This project uses a simple **external-field model** where path costs represent normalised congestion.

---

## Results Summary

### Baseline vs Random vs ORBIT (Abilene)

| Method                   | Mean MLU | Best MLU |
| ------------------------ | -------- | -------- |
| Shortest-path            | 3.32     | –        |
| Random (10,000 samples)      | 2.88     | 2.08     |
| **ORBIT (10,000 samples)** | **2.80** | **1.95** |

ORBIT identifies better routing configurations than random search and dramatically improves upon shortest-path routing.

### MILP vs PF/ADMM vs ORBIT

| Method      | Throughput | Fairness | Blocking |
| ----------- | ---------- | -------- | -------- |
| **MILP**    | highest    | lowest   | lowest   |
| **PF/ADMM** | medium     | highest  | high     |
| **ORBIT**   | medium | mid      | high     |

* **MILP** achieves the highest throughput, but does so by favouring a few demands, leading to low fairness.
* **PF/ADMM** redistributes flow more evenly, giving high fairness, but at the cost of reduced throughput and more blocked traffic.
* **ORBIT** lands between the two: its allocations are not as throughput-oriented as MILP nor as fair as PF, reflecting its role as a simple heuristic based on a basic Ising energy.

Overall, ORBIT finds reasonable solutions but does not match either classical baseline in this configuration.

However, despite using only a simple external-field formulation, ORBIT consistently discovers routing configurations that reduce congestion compared to the classical shortest-path baseline and outperform naive random search on the same candidate path space.

This demonstrates that Ising-based optimisation is a viable approach for traffic engineering, even without full problem-specific modelling. The ability to encode TE objectives into an energy function and explore the solution space through sampling provides a flexible framework that can incorporate richer signals (e.g., link-level penalties, pairwise couplings, or learned energy terms).

Overall, these results indicate that ORBIT is clearly applicable in this domain, and with further refinement of the energy model, it has strong potential as a practical heuristic for low-congestion routing in large networks.

## **Future Improvements for ORBIT**

To strengthen ORBIT’s performance and bias it toward better solutions, future work could include:

* **Pairwise couplings** between paths that share congested links
  -> capture interactions and reduce joint overloads.

* **More expressive energy function**
  (include link-capacity penalties, not just per-path costs).

* **Improved annealing / temperature schedule**
  to better explore and concentrate around good solutions.

These additions would make the energy landscape more informative and help ORBIT converge to higher-quality routing assignments.