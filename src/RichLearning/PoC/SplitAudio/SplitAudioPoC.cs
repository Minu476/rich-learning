// ═══════════════════════════════════════════════════════════════════════════
//  Rich Learning — Split-Audio Continual Learning Benchmark
//
//  Real-world continual learning experiment using FSD50K audio features:
//    Task A: Train MLP on 15 musical instrument classes
//    Task B: Train MLP on 17 environmental sound classes
//
//  After training Task B, measure how much Task A knowledge is lost.
//  Topological graph memory retains near-perfect recall of Task A audio.
//
//  Uses REAL audio features (18-dim mean MFCC from FSD50K dataset).
//  Uses a REAL neural network (MLP 18→64→32→classes) with backpropagation.
//  Pure C# — no Python, no ML.NET, no PyTorch.
//
//  Why audio continual learning matters:
//    • Audio streams are inherently non-stationary (new sounds appear)
//    • Deployed models can't retrain from scratch on every new class
//    • Catastrophic forgetting is a real-world production problem
// ═══════════════════════════════════════════════════════════════════════════

namespace RichLearning.PoC.SplitAudio;

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using RichLearning.Abstractions;
using RichLearning.Models;
using Microsoft.Extensions.Logging;

// ─────────────────────────────────────────────────────────────────────────
//  FSD50K Feature Loader — reads pre-extracted mean-MFCC CSV
// ─────────────────────────────────────────────────────────────────────────

/// <summary>
/// A single audio sample with 18-dim mean MFCC features and a label.
/// </summary>
public record AudioSample(string FileId, string Label, string Task, float[] Features);

public static class Fsd50kLoader
{
    /// <summary>Total MFCC feature dimensions per sample.</summary>
    public const int FeatureDim = 18;

    /// <summary>
    /// Loads audio features from the pre-extracted CSV.
    /// CSV columns: file_id, fname, task, label, mfcc_0 .. mfcc_17
    /// </summary>
    public static List<AudioSample> LoadCsv(string csvPath)
    {
        var samples = new List<AudioSample>();
        using var reader = new StreamReader(csvPath);
        string? header = reader.ReadLine(); // skip header

        while (reader.ReadLine() is { } line)
        {
            var parts = line.Split(',');
            if (parts.Length < 4 + FeatureDim) continue;

            string fileId = parts[0];
            // parts[1] = fname (not needed)
            string task = parts[2];
            string label = parts[3];

            float[] features = new float[FeatureDim];
            for (int i = 0; i < FeatureDim; i++)
                features[i] = float.Parse(parts[4 + i], CultureInfo.InvariantCulture);

            samples.Add(new AudioSample(fileId, label, task, features));
        }

        return samples;
    }

    /// <summary>Normalise features to zero-mean, unit-variance per dimension.</summary>
    public static void StandardizeFeatures(List<AudioSample> samples)
    {
        if (samples.Count == 0) return;

        int dim = FeatureDim;
        float[] mean = new float[dim];
        float[] stddev = new float[dim];

        foreach (var s in samples)
            for (int d = 0; d < dim; d++)
                mean[d] += s.Features[d];

        for (int d = 0; d < dim; d++)
            mean[d] /= samples.Count;

        foreach (var s in samples)
            for (int d = 0; d < dim; d++)
                stddev[d] += (s.Features[d] - mean[d]) * (s.Features[d] - mean[d]);

        for (int d = 0; d < dim; d++)
            stddev[d] = MathF.Sqrt(stddev[d] / samples.Count + 1e-8f);

        foreach (var s in samples)
            for (int d = 0; d < dim; d++)
                s.Features[d] = (s.Features[d] - mean[d]) / stddev[d];
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Label Index — maps string labels to integer class indices
// ─────────────────────────────────────────────────────────────────────────

public class LabelIndex
{
    private readonly Dictionary<string, int> _labelToIdx = new();
    private readonly List<string> _labels = [];

    public int Count => _labels.Count;

    public int GetOrAdd(string label)
    {
        if (_labelToIdx.TryGetValue(label, out int idx)) return idx;
        idx = _labels.Count;
        _labelToIdx[label] = idx;
        _labels.Add(label);
        return idx;
    }

    public int Get(string label) =>
        _labelToIdx.TryGetValue(label, out int idx) ? idx : -1;

    public string GetLabel(int idx) => _labels[idx];

    public bool Contains(string label) => _labelToIdx.ContainsKey(label);
}

// ─────────────────────────────────────────────────────────────────────────
//  Pure C# MLP for Audio Classification
//  Architecture: 18 → 64 → 32 → numClasses  (ReLU + Softmax)
// ─────────────────────────────────────────────────────────────────────────

public class AudioDenseLayer
{
    public float[,] Weights;
    public float[] Biases;
    public int InputDim, OutputDim;
    public float[] LastInput = [];
    public float[] LastPreActivation = [];
    public float[] LastOutput = [];
    public float[,] WeightGrad;
    public float[] BiasGrad;

    public AudioDenseLayer(int inputDim, int outputDim, Random rng)
    {
        InputDim = inputDim;
        OutputDim = outputDim;
        Weights = new float[inputDim, outputDim];
        Biases = new float[outputDim];
        WeightGrad = new float[inputDim, outputDim];
        BiasGrad = new float[outputDim];

        float scale = MathF.Sqrt(2.0f / inputDim);
        for (int i = 0; i < inputDim; i++)
            for (int j = 0; j < outputDim; j++)
                Weights[i, j] = (float)(rng.NextDouble() * 2 - 1) * scale;
    }

    public float[] Forward(float[] input, bool relu)
    {
        LastInput = input;
        LastPreActivation = new float[OutputDim];
        LastOutput = new float[OutputDim];

        for (int j = 0; j < OutputDim; j++)
        {
            float sum = Biases[j];
            for (int i = 0; i < InputDim; i++)
                sum += input[i] * Weights[i, j];
            LastPreActivation[j] = sum;
            LastOutput[j] = relu ? MathF.Max(0, sum) : sum;
        }
        return LastOutput;
    }

    public float[] Backward(float[] outputGrad, bool relu)
    {
        float[] inputGrad = new float[InputDim];
        for (int j = 0; j < OutputDim; j++)
        {
            float grad = relu && LastPreActivation[j] <= 0 ? 0 : outputGrad[j];
            BiasGrad[j] += grad;
            for (int i = 0; i < InputDim; i++)
            {
                WeightGrad[i, j] += LastInput[i] * grad;
                inputGrad[i] += Weights[i, j] * grad;
            }
        }
        return inputGrad;
    }

    public void UpdateWeights(float lr, int batchSize)
    {
        float scale = lr / batchSize;
        for (int i = 0; i < InputDim; i++)
            for (int j = 0; j < OutputDim; j++)
            {
                Weights[i, j] -= WeightGrad[i, j] * scale;
                WeightGrad[i, j] = 0;
            }
        for (int j = 0; j < OutputDim; j++)
        {
            Biases[j] -= BiasGrad[j] * scale;
            BiasGrad[j] = 0;
        }
    }

    public (float[,] W, float[] B) Snapshot() => ((float[,])Weights.Clone(), (float[])Biases.Clone());

    public void Restore((float[,] W, float[] B) s)
    {
        Array.Copy(s.W, Weights, Weights.Length);
        Array.Copy(s.B, Biases, Biases.Length);
    }
}

public class AudioMLP
{
    public AudioDenseLayer Layer1, Layer2, Layer3;
    private readonly int _numClasses;

    public AudioMLP(int inputDim, int numClasses, Random rng)
    {
        _numClasses = numClasses;
        Layer1 = new AudioDenseLayer(inputDim, 64, rng);
        Layer2 = new AudioDenseLayer(64, 32, rng);
        Layer3 = new AudioDenseLayer(32, numClasses, rng);
    }

    public float[] Forward(float[] input)
    {
        var h1 = Layer1.Forward(input, relu: true);
        var h2 = Layer2.Forward(h1, relu: true);
        var logits = Layer3.Forward(h2, relu: false);
        return Softmax(logits);
    }

    public int Predict(float[] input)
    {
        var probs = Forward(input);
        return Array.IndexOf(probs, probs.Max());
    }

    public float TrainStep(float[] input, int label)
    {
        var probs = Forward(input);
        float loss = -MathF.Log(MathF.Max(probs[label], 1e-8f));

        float[] grad = new float[_numClasses];
        for (int i = 0; i < _numClasses; i++)
            grad[i] = probs[i] - (i == label ? 1.0f : 0.0f);

        var g2 = Layer3.Backward(grad, relu: false);
        var g1 = Layer2.Backward(g2, relu: true);
        Layer1.Backward(g1, relu: true);
        return loss;
    }

    public void ApplyGradients(float lr, int batchSize)
    {
        Layer1.UpdateWeights(lr, batchSize);
        Layer2.UpdateWeights(lr, batchSize);
        Layer3.UpdateWeights(lr, batchSize);
    }

    public float TrainEpoch(List<AudioSample> samples, LabelIndex labelIndex,
        HashSet<string> allowedLabels, float lr, int batchSize, Random rng)
    {
        var indices = new List<int>();
        for (int i = 0; i < samples.Count; i++)
            if (allowedLabels.Contains(samples[i].Label))
                indices.Add(i);

        // Shuffle
        for (int i = indices.Count - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        float totalLoss = 0;
        int steps = 0;
        foreach (int idx in indices)
        {
            int classIdx = labelIndex.Get(samples[idx].Label);
            if (classIdx < 0) continue;
            totalLoss += TrainStep(samples[idx].Features, classIdx);
            steps++;
            if (steps % batchSize == 0)
                ApplyGradients(lr, batchSize);
        }
        if (steps % batchSize != 0)
            ApplyGradients(lr, steps % batchSize);

        return steps > 0 ? totalLoss / steps : 0;
    }

    public (float Accuracy, int Correct, int Total) Evaluate(
        List<AudioSample> samples, LabelIndex labelIndex, HashSet<string> allowedLabels)
    {
        int correct = 0, total = 0;
        foreach (var sample in samples)
        {
            if (!allowedLabels.Contains(sample.Label)) continue;
            int classIdx = labelIndex.Get(sample.Label);
            if (classIdx < 0) continue;

            if (Predict(sample.Features) == classIdx) correct++;
            total++;
        }
        return total > 0 ? ((float)correct / total, correct, total) : (0, 0, 0);
    }

    public ((float[,] W, float[] B), (float[,] W, float[] B), (float[,] W, float[] B)) Snapshot() =>
        (Layer1.Snapshot(), Layer2.Snapshot(), Layer3.Snapshot());

    public void Restore(((float[,] W, float[] B), (float[,] W, float[] B), (float[,] W, float[] B)) snap)
    {
        Layer1.Restore(snap.Item1);
        Layer2.Restore(snap.Item2);
        Layer3.Restore(snap.Item3);
    }

    private static float[] Softmax(float[] logits)
    {
        float max = logits.Max();
        float[] exp = logits.Select(x => MathF.Exp(x - max)).ToArray();
        float sum = exp.Sum();
        return exp.Select(e => e / sum).ToArray();
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  EWC Regularizer for Audio MLP
// ─────────────────────────────────────────────────────────────────────────

public class AudioEwcRegularizer
{
    private readonly float _lambda;
    private readonly float[,] _oldW1, _oldW2, _oldW3;
    private readonly float[,] _fisherL1, _fisherL2, _fisherL3;

    public AudioEwcRegularizer(AudioMLP model, List<AudioSample> samples,
        LabelIndex labelIndex, HashSet<string> allowedLabels, float lambda, Random rng)
    {
        _lambda = lambda;
        _oldW1 = (float[,])model.Layer1.Weights.Clone();
        _oldW2 = (float[,])model.Layer2.Weights.Clone();
        _oldW3 = (float[,])model.Layer3.Weights.Clone();
        _fisherL1 = new float[model.Layer1.InputDim, model.Layer1.OutputDim];
        _fisherL2 = new float[model.Layer2.InputDim, model.Layer2.OutputDim];
        _fisherL3 = new float[model.Layer3.InputDim, model.Layer3.OutputDim];

        // Compute Fisher from a subsample
        var eligible = samples.Where(s => allowedLabels.Contains(s.Label)).ToList();
        int nSamples = Math.Min(200, eligible.Count);
        var subset = eligible.OrderBy(_ => rng.Next()).Take(nSamples).ToList();

        foreach (var sample in subset)
        {
            int classIdx = labelIndex.Get(sample.Label);
            if (classIdx < 0) continue;
            model.TrainStep(sample.Features, classIdx);
            AccumulateFisher(model.Layer1, _fisherL1);
            AccumulateFisher(model.Layer2, _fisherL2);
            AccumulateFisher(model.Layer3, _fisherL3);
            ClearGrads(model.Layer1);
            ClearGrads(model.Layer2);
            ClearGrads(model.Layer3);
        }

        float fisherCap = 10.0f;
        NormAndClampFisher(_fisherL1, 1.0f / nSamples, fisherCap);
        NormAndClampFisher(_fisherL2, 1.0f / nSamples, fisherCap);
        NormAndClampFisher(_fisherL3, 1.0f / nSamples, fisherCap);
    }

    private static void AccumulateFisher(AudioDenseLayer layer, float[,] fisher)
    {
        for (int i = 0; i < layer.InputDim; i++)
            for (int j = 0; j < layer.OutputDim; j++)
                fisher[i, j] += layer.WeightGrad[i, j] * layer.WeightGrad[i, j];
    }

    private static void ClearGrads(AudioDenseLayer layer)
    {
        Array.Clear(layer.WeightGrad);
        Array.Clear(layer.BiasGrad);
    }

    private static void NormAndClampFisher(float[,] fisher, float scale, float cap)
    {
        for (int i = 0; i < fisher.GetLength(0); i++)
            for (int j = 0; j < fisher.GetLength(1); j++)
                fisher[i, j] = MathF.Min(fisher[i, j] * scale, cap);
    }

    public void AddPenaltyGradients(AudioMLP model)
    {
        AddLayerPenalty(model.Layer1, _oldW1, _fisherL1);
        AddLayerPenalty(model.Layer2, _oldW2, _fisherL2);
        AddLayerPenalty(model.Layer3, _oldW3, _fisherL3);
    }

    private void AddLayerPenalty(AudioDenseLayer layer, float[,] oldW, float[,] fisher)
    {
        float gradCap = 1.0f;
        for (int i = 0; i < layer.InputDim; i++)
            for (int j = 0; j < layer.OutputDim; j++)
            {
                float penalty = _lambda * fisher[i, j] * (layer.Weights[i, j] - oldW[i, j]);
                layer.WeightGrad[i, j] += Math.Clamp(penalty, -gradCap, gradCap);
            }
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Topological Audio Memory — Graph-based KNN classification
// ─────────────────────────────────────────────────────────────────────────

public class TopologicalAudioMemory
{
    private readonly IGraphMemory _graphMemory;
    private readonly IStateEncoder _encoder;
    private readonly List<(double[] Embedding, int ClassIdx)> _store = [];

    public int LandmarkCount { get; private set; }
    public int SampleCount => _store.Count;

    public TopologicalAudioMemory(IGraphMemory graphMemory, IStateEncoder encoder)
    {
        _graphMemory = graphMemory;
        _encoder = encoder;
    }

    /// <summary>Ingest audio samples as landmarks in graph memory.</summary>
    public async Task IngestSamplesAsync(List<AudioSample> samples, LabelIndex labelIndex,
        HashSet<string> allowedLabels, int stride = 1)
    {
        int added = 0;
        for (int i = 0; i < samples.Count; i += stride)
        {
            if (!allowedLabels.Contains(samples[i].Label)) continue;
            int classIdx = labelIndex.Get(samples[i].Label);
            if (classIdx < 0) continue;

            double[] embedding = samples[i].Features.Select(f => (double)f).ToArray();
            _store.Add((embedding, classIdx));

            var landmark = new StateLandmark
            {
                Id = $"audio_{samples[i].FileId}",
                Embedding = embedding,
                VisitCount = 1,
                ValueEstimate = classIdx,
                NoveltyScore = 1.0
            };
            await _graphMemory.UpsertLandmarkAsync(landmark);
            added++;
        }
        LandmarkCount += added;
        Console.WriteLine($"  [Topo] Ingested {added} audio landmarks (total: {LandmarkCount})");
    }

    /// <summary>Classify by K-nearest-neighbour voting in the stored embeddings.</summary>
    public (float Accuracy, int Correct, int Total) Evaluate(
        List<AudioSample> testSamples, LabelIndex labelIndex,
        HashSet<string> allowedLabels, int k = 5)
    {
        int correct = 0, total = 0;
        foreach (var sample in testSamples)
        {
            if (!allowedLabels.Contains(sample.Label)) continue;
            int trueClass = labelIndex.Get(sample.Label);
            if (trueClass < 0) continue;

            double[] embedding = sample.Features.Select(f => (double)f).ToArray();

            // Find K nearest neighbours by cosine distance
            var neighbours = _store
                .Select(s => (ClassIdx: s.ClassIdx, Dist: _encoder.Distance(embedding, s.Embedding)))
                .OrderBy(n => n.Dist)
                .Take(k)
                .ToList();

            // Majority vote
            int predicted = neighbours
                .GroupBy(n => n.ClassIdx)
                .OrderByDescending(g => g.Count())
                .First().Key;

            if (predicted == trueClass) correct++;
            total++;
        }
        return total > 0 ? ((float)correct / total, correct, total) : (0, 0, 0);
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Split-Audio Demo Runner
// ─────────────────────────────────────────────────────────────────────────

public static class SplitAudioDemo
{
    /// <summary>Task A: Musical instrument classes.</summary>
    public static readonly HashSet<string> InstrumentLabels = new()
    {
        "Acoustic_guitar", "Bass_drum", "Bass_guitar", "Crash_cymbal", "Cymbal",
        "Drum", "Drum_kit", "Electric_guitar", "Guitar", "Harmonica", "Harp",
        "Organ", "Piano", "Snare_drum", "Trumpet"
    };

    /// <summary>Task B: Environmental sound classes.</summary>
    public static readonly HashSet<string> EnvironmentLabels = new()
    {
        "Car", "Car_passing_by", "Door", "Engine", "Engine_starting", "Fire",
        "Knock", "Rain", "Raindrop", "Siren", "Thunder", "Thunderstorm",
        "Traffic_noise_and_roadway_noise", "Truck", "Walk_and_footsteps", "Water",
        "Wind"
    };

    public static async Task RunAsync(IGraphMemory memory, IStateEncoder encoder,
        ILogger logger, string? csvPath = null)
    {
        Console.WriteLine("╔══════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║  Split-Audio: Continual Learning on FSD50K Sound Features   ║");
        Console.WriteLine("║  Real MFCC features • MLP (18→64→32→N) • Pure C#            ║");
        Console.WriteLine("╚══════════════════════════════════════════════════════════════╝\n");

        // Resolve CSV path
        csvPath ??= Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..",
            "..", "..", "data", "fsd50k_features.csv");

        if (!File.Exists(csvPath))
        {
            // Try fallback paths
            var altPaths = new[]
            {
                Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "data", "fsd50k_features.csv"),
                Path.Combine(Directory.GetCurrentDirectory(), "data", "fsd50k_features.csv"),
                Path.Combine(Directory.GetCurrentDirectory(), "..", "..", "data", "fsd50k_features.csv"),
            };
            csvPath = altPaths.FirstOrDefault(File.Exists);
        }

        if (csvPath == null || !File.Exists(csvPath))
        {
            Console.WriteLine("ERROR: FSD50K features CSV not found.");
            Console.WriteLine("Run: python3 scripts/extract_fsd50k.py  (from repo root)");
            Console.WriteLine("or provide path: dotnet run -- SplitAudio --data /path/to/fsd50k_features.csv");
            return;
        }

        var rng = new Random(42);

        // Step 1: Load Data
        Console.WriteLine("--- Step 1: Loading FSD50K Audio Features ---");
        var allSamples = Fsd50kLoader.LoadCsv(csvPath);
        Fsd50kLoader.StandardizeFeatures(allSamples);
        Console.WriteLine($"  Loaded: {allSamples.Count} samples ({Fsd50kLoader.FeatureDim}-dim MFCC features)");

        var taskASamples = allSamples.Where(s => InstrumentLabels.Contains(s.Label)).ToList();
        var taskBSamples = allSamples.Where(s => EnvironmentLabels.Contains(s.Label)).ToList();
        Console.WriteLine($"  Task A (instruments): {taskASamples.Count} samples, {InstrumentLabels.Count} classes");
        Console.WriteLine($"  Task B (environment): {taskBSamples.Count} samples, {EnvironmentLabels.Count} classes");

        // Build label index for ALL classes (both tasks, unified output space)
        var labelIndex = new LabelIndex();
        foreach (var lbl in InstrumentLabels.OrderBy(l => l)) labelIndex.GetOrAdd(lbl);
        foreach (var lbl in EnvironmentLabels.OrderBy(l => l)) labelIndex.GetOrAdd(lbl);
        int totalClasses = labelIndex.Count;
        Console.WriteLine($"  Total classes: {totalClasses}\n");

        // Split into train/test (80/20)
        Shuffle(taskASamples, rng);
        Shuffle(taskBSamples, rng);
        int splitA = (int)(taskASamples.Count * 0.8);
        int splitB = (int)(taskBSamples.Count * 0.8);

        var trainA = taskASamples.Take(splitA).ToList();
        var testA = taskASamples.Skip(splitA).ToList();
        var trainB = taskBSamples.Take(splitB).ToList();
        var testB = taskBSamples.Skip(splitB).ToList();
        var trainAll = trainA.Concat(trainB).ToList();
        var testAll = testA.Concat(testB).ToList();

        Console.WriteLine($"  Train A: {trainA.Count}, Test A: {testA.Count}");
        Console.WriteLine($"  Train B: {trainB.Count}, Test B: {testB.Count}\n");

        // Step 2: Train MLP on Task A
        Console.WriteLine("--- Step 2: Training MLP on Task A (instruments) ---");
        var mlp = new AudioMLP(Fsd50kLoader.FeatureDim, totalClasses, rng);
        int epochs = 20;
        float lr = 0.005f;
        int batchSize = 16;

        for (int ep = 1; ep <= epochs; ep++)
        {
            float loss = mlp.TrainEpoch(trainA, labelIndex, InstrumentLabels, lr, batchSize, rng);
            if (ep % 5 == 0 || ep == 1)
            {
                var (accA, _, _) = mlp.Evaluate(testA, labelIndex, InstrumentLabels);
                Console.WriteLine($"  Epoch {ep,2}/{epochs}  Loss: {loss:F4}  TaskA Acc: {accA * 100:F1}%");
            }
        }

        var (taskAAccAfterA, cA1, tA1) = mlp.Evaluate(testA, labelIndex, InstrumentLabels);
        Console.WriteLine($"\n  MLP Task A accuracy: {taskAAccAfterA * 100:F1}% ({cA1}/{tA1})");
        var snapshotAfterA = mlp.Snapshot();

        // Step 3: Build Topological Memory for Task A
        Console.WriteLine("\n--- Step 3: Building Topological Memory for Task A ---");
        await memory.InitialiseSchemaAsync();
        var topoMemory = new TopologicalAudioMemory(memory, encoder);
        await topoMemory.IngestSamplesAsync(trainA, labelIndex, InstrumentLabels);

        var (topoAccA, cTA, tTA) = topoMemory.Evaluate(testA, labelIndex, InstrumentLabels);
        Console.WriteLine($"  Topological Memory Task A: {topoAccA * 100:F1}% ({cTA}/{tTA})");
        Console.WriteLine($"  Landmarks: {topoMemory.LandmarkCount}");

        // Step 4: EWC Fisher Information
        Console.WriteLine("\n--- Step 4: Computing EWC Fisher Information ---");
        var ewcMlp = new AudioMLP(Fsd50kLoader.FeatureDim, totalClasses, rng);
        ewcMlp.Restore(snapshotAfterA);
        var ewc = new AudioEwcRegularizer(ewcMlp, trainA, labelIndex, InstrumentLabels,
            lambda: 100f, rng);
        Console.WriteLine("  Fisher computed (200 samples, lambda=100, clamped)");

        // Step 5: Train on Task B — FORGETTING TEST
        Console.WriteLine("\n--- Step 5: Training on Task B (FORGETTING TEST) ---");
        Console.WriteLine("  Bare MLP (no protection):");
        for (int ep = 1; ep <= epochs; ep++)
        {
            float loss = mlp.TrainEpoch(trainB, labelIndex, EnvironmentLabels, lr, batchSize, rng);
            if (ep % 5 == 0 || ep == 1)
            {
                var (accA, _, _) = mlp.Evaluate(testA, labelIndex, InstrumentLabels);
                var (accB, _, _) = mlp.Evaluate(testB, labelIndex, EnvironmentLabels);
                Console.WriteLine($"  Epoch {ep,2}  Loss: {loss:F4}  TaskA: {accA * 100:F1}%  TaskB: {accB * 100:F1}%");
            }
        }

        var (taskAAccAfterB, _, _) = mlp.Evaluate(testA, labelIndex, InstrumentLabels);
        var (taskBAccAfterB, _, _) = mlp.Evaluate(testB, labelIndex, EnvironmentLabels);

        Console.WriteLine("\n  EWC-protected MLP:");
        for (int ep = 1; ep <= epochs; ep++)
        {
            float totalLoss = 0;
            int steps = 0;
            var trainBShuffled = trainB.OrderBy(_ => rng.Next()).ToList();
            foreach (var sample in trainBShuffled)
            {
                int classIdx = labelIndex.Get(sample.Label);
                if (classIdx < 0) continue;
                totalLoss += ewcMlp.TrainStep(sample.Features, classIdx);
                steps++;
                if (steps % batchSize == 0)
                {
                    ewc.AddPenaltyGradients(ewcMlp);
                    ewcMlp.ApplyGradients(lr, batchSize);
                }
            }
            if (steps % batchSize != 0)
            {
                ewc.AddPenaltyGradients(ewcMlp);
                ewcMlp.ApplyGradients(lr, steps % batchSize);
            }

            if (ep % 5 == 0 || ep == 1)
            {
                var (ewcA, _, _) = ewcMlp.Evaluate(testA, labelIndex, InstrumentLabels);
                var (ewcB, _, _) = ewcMlp.Evaluate(testB, labelIndex, EnvironmentLabels);
                Console.WriteLine($"  Epoch {ep,2}  Loss: {totalLoss / steps:F4}  TaskA: {ewcA * 100:F1}%  TaskB: {ewcB * 100:F1}%");
            }
        }

        var (ewcTaskAFinal, _, _) = ewcMlp.Evaluate(testA, labelIndex, InstrumentLabels);
        var (ewcTaskBFinal, _, _) = ewcMlp.Evaluate(testB, labelIndex, EnvironmentLabels);

        // Step 6: Add Task B to Topological Memory
        Console.WriteLine("\n--- Step 6: Adding Task B to Topological Memory ---");
        await topoMemory.IngestSamplesAsync(trainB, labelIndex, EnvironmentLabels);
        var (topoAccAFinal, _, _) = topoMemory.Evaluate(testA, labelIndex, InstrumentLabels);
        var (topoAccBFinal, _, _) = topoMemory.Evaluate(testB, labelIndex, EnvironmentLabels);
        var allLabels = new HashSet<string>(InstrumentLabels.Concat(EnvironmentLabels));
        var (topoAccAll, _, _) = topoMemory.Evaluate(testAll, labelIndex, allLabels);

        // Step 7: Report
        float forgetting = taskAAccAfterA > 0
            ? (taskAAccAfterA - taskAAccAfterB) / taskAAccAfterA * 100 : 0;
        float ewcRetention = taskAAccAfterA > 0
            ? ewcTaskAFinal / taskAAccAfterA * 100 : 0;
        float topoRetention = topoAccA > 0
            ? topoAccAFinal / topoAccA * 100 : 0;

        Console.WriteLine("\n╔═══════════════════════════════════════════════════════════╗");
        Console.WriteLine("║           SPLIT-AUDIO FORGETTING REPORT                  ║");
        Console.WriteLine("║       FSD50K: Instruments vs Environment Sounds          ║");
        Console.WriteLine("╠═══════════════════════════════════════════════════════════╣");
        Console.WriteLine($"║  Task A after training A:     {taskAAccAfterA * 100,6:F1}%                    ║");
        Console.WriteLine($"║  Task A after B (Bare MLP):   {taskAAccAfterB * 100,6:F1}%  <- FORGETTING     ║");
        Console.WriteLine($"║  Task A after B (EWC):        {ewcTaskAFinal * 100,6:F1}%  <- Regularized     ║");
        Console.WriteLine($"║  Task A (Topological Memory): {topoAccAFinal * 100,6:F1}%  <- Graph-stored    ║");
        Console.WriteLine("║                                                           ║");
        Console.WriteLine($"║  Task B (Bare MLP):           {taskBAccAfterB * 100,6:F1}%                    ║");
        Console.WriteLine($"║  Task B (EWC):                {ewcTaskBFinal * 100,6:F1}%                    ║");
        Console.WriteLine($"║  Task B (Topological Memory): {topoAccBFinal * 100,6:F1}%                    ║");
        Console.WriteLine($"║  Overall Topo (all classes):  {topoAccAll * 100,6:F1}%                    ║");
        Console.WriteLine("║                                                           ║");
        Console.WriteLine($"║  Forgetting Rate (MLP):       {forgetting,6:F1}%                    ║");
        Console.WriteLine($"║  EWC Retention:               {ewcRetention,6:F1}%                    ║");
        Console.WriteLine($"║  Topo Retention:              {topoRetention,6:F1}%                    ║");
        Console.WriteLine("║                                                           ║");
        Console.WriteLine($"║  Samples: {allSamples.Count,4} | Classes: {totalClasses,2}              ║");
        Console.WriteLine($"║  Features: {Fsd50kLoader.FeatureDim}-dim mean MFCC from FSD50K             ║");
        Console.WriteLine("╚═══════════════════════════════════════════════════════════╝");
    }

    private static void Shuffle<T>(List<T> list, Random rng)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }
}
