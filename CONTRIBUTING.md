# Contributing to Rich Learning

Thank you for your interest in contributing! Rich Learning is a C#/.NET framework for continual reinforcement learning with topological graph memory.

## How to Contribute

### Areas We Need Help

1. **New Domain Encoders** — Implement `IStateEncoder` for robotics, NLP, game AI, finance, etc.
2. **Alternative Graph Backends** — Implement `IGraphMemory` with SQLite, Redis, in-memory, or other databases.
3. **New PoC Experiments** — Demonstrate continual learning on your domain. Real data is strongly preferred.
4. **Performance Optimizations** — SIMD intrinsics, Span<T> usage, GPU acceleration, parallel processing.
5. **Documentation & Tutorials** — Help others understand and use the framework.
6. **Bug Fixes** — Found an issue? Fix it and send a PR.

### Getting Started

1. **Fork** the repo and clone your fork.
2. Install [.NET 10 SDK](https://dotnet.microsoft.com/download) and [Neo4j](https://neo4j.com/download/).
3. Run the tests: `dotnet build && dotnet run -- SplitMnist`
4. Create a feature branch: `git checkout -b feature/my-awesome-encoder`
5. Make your changes with clear commit messages.
6. Submit a Pull Request.

### Code Standards

- **C# 12+** with nullable reference types enabled.
- **async/await** for all I/O operations (Neo4j, file, network).
- **XML doc comments** on all public APIs.
- **Records** for immutable data models.
- **Interfaces** for extensibility points (IStateEncoder, IGraphMemory, etc.).
- No external ML frameworks — pure C# numerical code for core components.

### Adding a New PoC

1. Create a folder under `src/RichLearning/PoC/YourDomain/`
2. Implement a domain-specific `IStateEncoder`
3. Create a demo class with a `RunAsync` method
4. Use **real data** — download datasets or include small test datasets
5. Compare against at least one baseline (bare neural net, EWC, etc.)
6. Wire it up in `Program.cs`

### Pull Request Checklist

- [ ] Code builds without errors: `dotnet build`
- [ ] New code has XML doc comments
- [ ] No new warnings introduced
- [ ] PR description explains what and why
- [ ] Real data used (not simulated/hardcoded) for PoCs

## Code of Conduct

Be respectful, constructive, and welcoming. We're here to advance the science of continual learning together.

## Questions?

Open an issue or start a discussion. We're happy to help you get started!
