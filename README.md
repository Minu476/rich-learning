# Rich Learning: Topological Graph Memory for Lifelong RL

**Author:** Nasser Towfigh  
**License:** [Apache 2.0](LICENSE) (Source Code) / [CC BY 4.0](paper/Rich-Learning-Paper.pdf) (Paper)  
**Status:** Published Reference Implementation

## 📖 What is Rich Learning?
**Rich Learning** is a reinforcement learning paradigm focused on the accumulation of persistent knowledge assets rather than transient weight optimization.

In standard Deep RL, an agent "lives paycheck to paycheck"—often overwriting neural weights to learn new tasks (catastrophic forgetting). **Rich Learning** agents accumulate a **Topological Graph Memory**—a navigable map of the policy-state space stored in a graph database (Neo4j). This allows the agent to "spend" previously learned structures to solve novel problems zero-shot.

## 🚀 Key Innovations
This repository contains the reference implementation of the **Rich Learning** architecture, featuring:

### 1. Topological Graph Memory
* **Asset-Based Learning:** Knowledge is stored as distinct nodes ("Landmarks") and edges ("Transitions") in Neo4j.
* **Zero Forgetting:** New experiences add to the graph; they do not degrade existing structures.
* **Loop Detection:** Uses Cypher queries to explicitly detect and break infinite loops in navigation.

### 2. Recursive Meta Hierarchy (RMH)
A self-organizing structure where behavioral patterns are compounded into complex subgraphs. Just as compound interest grows wealth, the hierarchy allows the agent to compose high-level strategies from low-level primitives without retraining.

### 3. Agent Efficiency Tiering
The system identifies "Star" agents and shares their learned topological patterns with "Struggling" agents, amplifying collective intelligence across the swarm.

## 📄 Read the Paper
[**Download the Full Research Paper (PDF)**](paper/Rich-Learning-Paper.pdf)  
*Abstract: We introduce Rich Learning, a reinforcement learning architecture that addresses catastrophic forgetting through topological graph memory...*

## 📊 Results (from Paper)
| Domain | Challenge | Result |
| :--- | :--- | :--- |
| **Warehouse Logistics** | Multi-agent coordination (44 AGVs) | **0% Collision Rate**, 70 deliveries |
| **Atari Games** | Continual learning (Pong → Breakout) | **100% Skill Retention** |
| **Medical Imaging** | Distribution drift across 3 scanners | **100% Diagnostic Retention** |

## 🛠️ Tech Stack
* **Language:** C# 12 / .NET 10 (Zero Python dependencies)
* **Database:** Neo4j (Graph Persistence)
* **Core:** Fugue-Cartographer Engine

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
