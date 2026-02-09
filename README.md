# Rich Learning: Topological Graph Memory for Lifelong RL

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18565288.svg)](https://doi.org/10.5281/zenodo.18565288)
![Energy Efficient](https://img.shields.io/badge/Energy%20Efficiency-High-brightgreen?style=for-the-badge&logo=leaf)
![Architecture](https://img.shields.io/badge/Hidden%20Layers-None-blue?style=for-the-badge)

**Author:** Nasser Towfigh  
**License:** [Apache 2.0](LICENSE)  
**Status:** Published Reference Implementation

## 📖 What is Rich Learning?
**Rich Learning** is a reinforcement learning paradigm focused on the accumulation of persistent knowledge assets rather than transient weight optimization.

In standard Deep RL, an agent "lives paycheck to paycheck"—often overwriting neural weights to learn new tasks (catastrophic forgetting). **Rich Learning** agents accumulate a **Topological Graph Memory**—a navigable map of the policy-state space stored in a graph database (Neo4j). New experiences *add to* the graph; they never degrade existing structure.

## 🚀 Key Innovations
This repository contains the reference implementation of the **Rich Learning** architecture, featuring:

* **Topological Graph Memory:** Knowledge is stored as navigable "Landmarks" and "Transitions" in Neo4j, not just neural weights.
* **Zero Forgetting:** New experiences add nodes and edges; they do not modify existing graph structure.
* **Explainable Plans:** Navigation through named landmarks, fully auditable.
* **C# / .NET 10:** Implemented in pure C# with zero Python dependencies, achieving ~30–50× performance gains in RL inner loops.

## 📊 Experimental Results (from Paper)

We validate on two continual learning benchmarks where standard MLPs catastrophically forget:

| Benchmark | Method | Task A Accuracy | After Task B | Retention |
| :--- | :--- | :--- | :--- | :--- |
| **Split-MNIST** | Bare MLP | 97.9% | 0.0% | 0.0% |
| | EWC (λ=100) | 97.9% | 19.1% | 19.5% |
| | **Topological Memory** | 85.2% | **85.2%** | **100.0%** |
| **Split-Audio** | Bare MLP | — | — | ~0% |
| (FSD50K) | **Topological Memory** | — | — | **100.0%** |

*The topological graph retains 100% of Task A landmarks after training fully on Task B.*

## 📄 Read the Paper
[**Download the Full Research Paper (PDF)**](paper/Rich-Learning-Paper.pdf)  
*Abstract: We introduce Rich Learning, a reinforcement learning paradigm that addresses catastrophic forgetting through topological graph memory...*

## 🛠️ Tech Stack
* **Language:** C# 12 / .NET 10 (Zero Python dependencies)
* **Database:** Neo4j (Graph Persistence)
* **Interfaces:** IGraphMemory, IStateEncoder, IExplorationStrategy, Cartographer

## ⚡ Architectural Note: No Hidden Layers

Unlike traditional Deep Learning approaches that rely on opaque hidden layers and computationally expensive backpropagation, **Rich Learning** operates without hidden layers.

By offloading intelligence into explicit graph topology rather than neural weights, this paradigm achieves:

* **Transparency:** Every decision path is traceable through named landmarks and edges.
* **Energy Efficiency:** Inference is reduced to graph traversal (O(1) per hop) rather than matrix multiplication (O(N²)), resulting in a fraction of the energy consumption typical of Deep Neural Networks.
* **Suitability for Edge:** Ideal for low-power, battery-operated devices where thermal limits and battery life are critical.

## ⚡ Quick Start

```bash
# Prerequisites: .NET 10 SDK, Neo4j 5+

# Clone and build
git clone https://github.com/Minu476/rich-learning.git
cd rich-learning/src/RichLearning
dotnet build

# Run Split-MNIST (downloads real MNIST, no Neo4j needed for MLP baseline)
NEO4J_URI="bolt://localhost:7687" NEO4J_USER="neo4j" NEO4J_PASSWORD="password" \
  dotnet run -- SplitMnist

# Run Split-Audio (FSD50K features — extract first with scripts/extract_fsd50k.py)
dotnet run -- SplitAudio

# C# performance benchmark
dotnet run -- Benchmark

# Interactive graph exploration demo
dotnet run -- Demo
```

## 📂 Project Structure

```
rich-learning/
├── src/RichLearning/
│   ├── Abstractions/          # Interfaces (IGraphMemory, IStateEncoder, ...)
│   ├── Models/                # StateLandmark, StateTransition, SubgoalDirective
│   ├── Memory/                # Neo4jGraphMemory implementation
│   ├── Planning/              # Cartographer (mid-level planner)
│   └── PoC/
│       ├── SplitMnist/        # Catastrophic forgetting on MNIST digits
│       └── SplitAudio/        # Catastrophic forgetting on FSD50K audio
├── data/                      # Pre-extracted features (gitignored for large files)
├── paper/                     # Research paper PDF
├── scripts/                   # Data preparation scripts
└── README.md
```

## 🔌 Extend Rich Learning

### Custom State Encoder
```csharp
public class MyEncoder : IStateEncoder
{
    public int EmbeddingDimension => 128;
    public double[] Encode(double[] raw) => MyTransform(raw);
    public double Distance(double[] a, double[] b) => CosineDistance(a, b);
}
```

### Alternative Graph Backend
Implement `IGraphMemory` for SQLite, Redis, or in-memory graphs.

### New PoC
Add a folder under `PoC/` with a static `RunAsync` method — the pattern is self-documenting.

## 🔗 Citation
If you use this methodology, please cite:

> Towfigh, N. (2026). *Rich Learning: Topological Graph Memory for Lifelong Reinforcement Learning*. GitHub. https://github.com/Minu476/rich-learning

## ⚖️ License & IP
This project is licensed under **Apache License 2.0**.
* **Code:** You are free to use, modify, and distribute this software.
* **Patents:** This license grants an explicit patent grant for the *specific implementation* provided here.
