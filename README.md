# Quantum Graph Coloring with Qiskit

This repository presents a quantum-oriented approach to the classical **Graph Coloring Problem** using **Qiskit**.

The project models graph coloring as a **QUBO (Quadratic Unconstrained Binary Optimization)** problem and converts it into a form compatible with quantum optimization workflows. It also builds the corresponding **QAOA circuit structure** and generates visual outputs for better interpretation of the model.

## Project Overview

Graph coloring is a well-known combinatorial optimization problem where each node in a graph must be assigned a color such that no two connected nodes share the same color.

In this project:

- A random graph with 6 nodes is generated
- The coloring problem is formulated using 4 colors
- The optimization problem is converted into a QUBO model
- The model is solved using Qiskit's exact eigensolver
- A QAOA circuit ansatz is created for the same formulation
- Multiple visualizations are generated, including the graph, adjacency matrix, QUBO structure, and circuit layout

## Why This Is Quantum

This project uses **Qiskit**, IBM's quantum computing framework.

The workflow is quantum-oriented because:

- The problem is formulated using quantum optimization tools from Qiskit
- The graph coloring task is mapped into a **QUBO / Ising-style energy model**
- A **QAOA quantum circuit ansatz** is constructed for the problem

Although the final optimal solution in this version is obtained with an exact eigensolver on classical hardware, the formulation itself follows the same structure used in quantum optimization pipelines.

## Problem Formulation

For each node `i` and color `c`, a binary decision variable is defined:

- `x(i,c) = 1` if node `i` is assigned color `c`
- `x(i,c) = 0` otherwise

The objective function enforces two conditions:

1. Each node must receive exactly one color
2. Adjacent nodes must not share the same color

These rules are encoded as penalty terms inside the QUBO objective.

## 
