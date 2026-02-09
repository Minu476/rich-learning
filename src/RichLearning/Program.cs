using Microsoft.Extensions.Logging;
using RichLearning;
using RichLearning.Abstractions;
using RichLearning.Memory;
using RichLearning.Planning;
using RichLearning.PoC.SplitMnist;
using RichLearning.PoC.SplitAudio;

// ─────────────────────────────────────────────────────────────────────────
//  Rich Learning — Continual RL with Topological Graph Memory
//
//  Usage:
//    dotnet run                           # Interactive demo
//    dotnet run -- SplitMnist             # Split-MNIST catastrophic forgetting PoC
//    dotnet run -- SplitAudio             # Split-Audio with FSD50K features
//    dotnet run -- Benchmark              # C# vs Python speed benchmark
//
//  Environment:
//    NEO4J_URI=bolt://localhost:7687
//    NEO4J_USER=neo4j
//    NEO4J_PASSWORD=password
// ─────────────────────────────────────────────────────────────────────────

var neo4jUri = Environment.GetEnvironmentVariable("NEO4J_URI") ?? "bolt://localhost:7687";
var neo4jUser = Environment.GetEnvironmentVariable("NEO4J_USER") ?? "neo4j";
var neo4jPassword = Environment.GetEnvironmentVariable("NEO4J_PASSWORD") ?? "password";

using var loggerFactory = LoggerFactory.Create(builder =>
{
    builder.SetMinimumLevel(LogLevel.Warning);
    builder.AddConsole();
});

var command = args.Length > 0 ? args[0].ToLowerInvariant() : "demo";

switch (command)
{
    case "splitmnist":
        await RunSplitMnist();
        break;
    case "splitaudio":
        await RunSplitAudio();
        break;
    case "benchmark":
        RunBenchmark();
        break;
    default:
        await RunDemo();
        break;
}

// ── Split-MNIST PoC ──

async Task RunSplitMnist()
{
    var logger = loggerFactory.CreateLogger("SplitMnist");
    await using var memory = new Neo4jGraphMemory(neo4jUri, neo4jUser, neo4jPassword,
        loggerFactory.CreateLogger<Neo4jGraphMemory>());
    await memory.InitialiseSchemaAsync();

    var encoder = new MnistStateEncoder();
    await SplitMnistDemo.RunAsync(memory, encoder, logger);
}

// ── Split-Audio PoC ──

async Task RunSplitAudio()
{
    var logger = loggerFactory.CreateLogger("SplitAudio");
    await using var memory = new Neo4jGraphMemory(neo4jUri, neo4jUser, neo4jPassword,
        loggerFactory.CreateLogger<Neo4jGraphMemory>());
    await memory.InitialiseSchemaAsync();

    var encoder = new DefaultStateEncoder(embeddingDimension: Fsd50kLoader.FeatureDim);

    // Check for --data argument
    string? csvPath = null;
    for (int i = 1; i < args.Length - 1; i++)
        if (args[i] == "--data") csvPath = args[i + 1];

    await SplitAudioDemo.RunAsync(memory, encoder, logger, csvPath);
}

// ── C# vs Python Benchmark ──

void RunBenchmark()
{
    Console.WriteLine("╔══════════════════════════════════════════════════════════════╗");
    Console.WriteLine("║   Rich Learning — C# vs Python Performance Benchmark       ║");
    Console.WriteLine("║   .NET 10 JIT vs CPython: who wins for RL numerics?         ║");
    Console.WriteLine("╚══════════════════════════════════════════════════════════════╝\n");

    var rng = new Random(42);
    const int dim = 784;
    const int iters = 100_000;

    // 1. Vector distance computation
    double[] a = Enumerable.Range(0, dim).Select(_ => rng.NextDouble()).ToArray();
    double[] b = Enumerable.Range(0, dim).Select(_ => rng.NextDouble()).ToArray();

    var sw = System.Diagnostics.Stopwatch.StartNew();
    double sum = 0;
    for (int i = 0; i < iters; i++)
    {
        double dot = 0, nA = 0, nB = 0;
        for (int j = 0; j < dim; j++) { dot += a[j] * b[j]; nA += a[j] * a[j]; nB += b[j] * b[j]; }
        sum += 1.0 - dot / (Math.Sqrt(nA) * Math.Sqrt(nB));
    }
    sw.Stop();
    Console.WriteLine($"  Cosine distance ({dim}-dim, {iters:N0} iters): {sw.ElapsedMilliseconds} ms");
    Console.WriteLine($"    = {(double)iters / sw.Elapsed.TotalSeconds:N0} distances/sec");

    // 2. Dense matrix multiply (forward pass simulation)
    float[,] W = new float[dim, 256];
    float[] x = new float[dim];
    float[] y = new float[256];
    for (int i = 0; i < dim; i++) { x[i] = (float)rng.NextDouble(); for (int j = 0; j < 256; j++) W[i, j] = (float)(rng.NextDouble() * 0.1); }

    sw.Restart();
    for (int iter = 0; iter < 10_000; iter++)
    {
        for (int j = 0; j < 256; j++)
        {
            float s = 0;
            for (int i = 0; i < dim; i++) s += x[i] * W[i, j];
            y[j] = MathF.Max(0, s);
        }
    }
    sw.Stop();
    Console.WriteLine($"\n  Dense matmul ({dim}x256, ReLU, 10K iters): {sw.ElapsedMilliseconds} ms");
    Console.WriteLine($"    = {10000.0 / sw.Elapsed.TotalSeconds:N0} forward passes/sec");

    // 3. MLP training step
    var mlp = new MLP(rng);
    float[] img = new float[784];
    for (int i = 0; i < 784; i++) img[i] = (float)rng.NextDouble();

    sw.Restart();
    for (int i = 0; i < 1000; i++)
        mlp.TrainStep(img, i % 10);
    sw.Stop();
    Console.WriteLine($"\n  MLP train step (784->256->128->10, 1K steps): {sw.ElapsedMilliseconds} ms");
    Console.WriteLine($"    = {1000.0 / sw.Elapsed.TotalSeconds:N0} train steps/sec");

    Console.WriteLine("\n  Compare these numbers against Python/NumPy:");
    Console.WriteLine("  Equivalent Python (no NumPy) is typically 20-50x slower.");
    Console.WriteLine("  Python + NumPy BLAS is competitive for large matmuls,");
    Console.WriteLine("  but C# wins on small-batch RL loops (no Python overhead).\n");
}

// ── Interactive Demo ──

async Task RunDemo()
{
    Console.WriteLine("╔══════════════════════════════════════════════════════════════╗");
    Console.WriteLine("║       Rich Learning — Topological Graph Memory for RL       ║");
    Console.WriteLine("║       .NET 10 • Neo4j • Continual Learning                  ║");
    Console.WriteLine("╚══════════════════════════════════════════════════════════════╝\n");

    await using var memory = new Neo4jGraphMemory(neo4jUri, neo4jUser, neo4jPassword,
        loggerFactory.CreateLogger<Neo4jGraphMemory>());
    await memory.InitialiseSchemaAsync();

    var encoder = new DefaultStateEncoder(embeddingDimension: 8);
    var cartographer = new Cartographer(memory, encoder, loggerFactory.CreateLogger<Cartographer>())
    {
        NoveltyThreshold = 0.4
    };

    var rng2 = new Random(42);
    const int episodes = 3, steps = 15;

    for (int ep = 0; ep < episodes; ep++)
    {
        Console.WriteLine($"\n-- Episode {ep + 1} --");
        double[] state = Enumerable.Range(0, 8).Select(_ => rng2.NextDouble()).ToArray();

        string? prevId = null;
        for (int step = 0; step < steps; step++)
        {
            double reward = rng2.NextDouble() * 2.0 - 1.0;
            string landmarkId = await cartographer.ObserveStateAsync(state, reward);

            if (prevId != null)
                await cartographer.RecordTransitionAsync(prevId, landmarkId, rng2.Next(4), reward, 1, true);
            prevId = landmarkId;

            // Perturb state
            int idx = rng2.Next(8);
            state[idx] += (rng2.NextDouble() - 0.5) * 0.3;
        }
    }

    Console.WriteLine("\n" + await cartographer.GetMapSummaryAsync());

    await memory.AssignClustersAsync();
    var (lm, tr) = await memory.GetGraphStatsAsync();
    Console.WriteLine($"Final graph: {lm} landmarks, {tr} transitions");
    Console.WriteLine("Explore at http://localhost:7474 (Neo4j Browser)");
}
