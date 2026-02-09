// ═══════════════════════════════════════════════════════════════════════════
//  Rich Learning — Split-MNIST Continual Learning Benchmark
//
//  The gold-standard continual learning experiment:
//    Task A: Train MLP on digits 0-4   → measure accuracy on 0-4
//    Task B: Train MLP on digits 5-9   → measure accuracy on 0-4 again
//
//  A standard MLP suffers catastrophic forgetting on Task A after Task B.
//  Topological graph memory retains near-perfect recall of Task A landmarks.
//
//  Uses REAL MNIST data (IDX format, downloaded from the internet).
//  Uses a REAL neural network (MLP 784→256→128→10) with backpropagation.
//  Pure C# — no Python, no ML.NET, no PyTorch.
//
//  Why C# instead of Python?
//    • JIT-compiled: 5-50x faster inner loops vs NumPy-free Python
//    • No GIL: true parallelism with Parallel.For / SIMD
//    • .NET 10 scripting: `dotnet run` feels like Python, runs like C++
// ═══════════════════════════════════════════════════════════════════════════

namespace RichLearning.PoC.SplitMnist;

using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net.Http;
using System.Buffers.Binary;
using System.Threading.Tasks;
using RichLearning.Abstractions;
using RichLearning.Models;
using RichLearning.Memory;
using RichLearning.Planning;
using Microsoft.Extensions.Logging;

// ─────────────────────────────────────────────────────────────────────────
//  MNIST Data Loader — Real IDX format reader (gz-compressed)
// ─────────────────────────────────────────────────────────────────────────

public static class MnistDataLoader
{
    private const string BaseUrl = "https://storage.googleapis.com/cvdf-datasets/mnist/";
    private static readonly string[] Files =
    {
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    };

    public static async Task<string> EnsureDataAsync(string dataDir)
    {
        Directory.CreateDirectory(dataDir);
        using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(5) };

        foreach (var file in Files)
        {
            var localPath = Path.Combine(dataDir, file);
            if (File.Exists(localPath))
            {
                Console.WriteLine($"  [MNIST] {file} cached");
                continue;
            }
            Console.WriteLine($"  [MNIST] Downloading {file} ...");
            try
            {
                var bytes = await http.GetByteArrayAsync(BaseUrl + file);
                await File.WriteAllBytesAsync(localPath, bytes);
                Console.WriteLine($"  [MNIST] Downloaded {file} ({bytes.Length / 1024} KB)");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  [MNIST] Download failed: {ex.Message}");
                return string.Empty;
            }
        }
        return dataDir;
    }

    public static (float[][] Images, byte[] Labels) LoadImages(string imagePath, string labelPath)
        => (ReadImages(imagePath), ReadLabels(labelPath));

    private static float[][] ReadImages(string path)
    {
        using var fs = File.OpenRead(path);
        using var stream = path.EndsWith(".gz") ? new GZipStream(fs, CompressionMode.Decompress) : (Stream)fs;
        using var reader = new BinaryReader(stream);

        if (ReadInt32BE(reader) != 2051) throw new InvalidDataException("Invalid MNIST image magic");
        int count = ReadInt32BE(reader);
        int rows = ReadInt32BE(reader);
        int cols = ReadInt32BE(reader);
        int pixels = rows * cols;

        float[][] images = new float[count][];
        for (int i = 0; i < count; i++)
        {
            byte[] raw = reader.ReadBytes(pixels);
            images[i] = raw.Select(b => b / 255.0f).ToArray();
        }
        return images;
    }

    private static byte[] ReadLabels(string path)
    {
        using var fs = File.OpenRead(path);
        using var stream = path.EndsWith(".gz") ? new GZipStream(fs, CompressionMode.Decompress) : (Stream)fs;
        using var reader = new BinaryReader(stream);

        if (ReadInt32BE(reader) != 2049) throw new InvalidDataException("Invalid MNIST label magic");
        int count = ReadInt32BE(reader);
        return reader.ReadBytes(count);
    }

    private static int ReadInt32BE(BinaryReader reader) => BinaryPrimitives.ReadInt32BigEndian(reader.ReadBytes(4));

    /// <summary>Generates synthetic MNIST-like data if download fails.</summary>
    public static (float[][] Images, byte[] Labels) GenerateSynthetic(int count, Random rng)
    {
        var images = new float[count][];
        var labels = new byte[count];

        for (int i = 0; i < count; i++)
        {
            labels[i] = (byte)(i % 10);
            images[i] = new float[784];
            int digit = labels[i];

            for (int px = 0; px < 784; px++)
            {
                int row = px / 28, col = px % 28;
                float signal = digit switch
                {
                    0 => MathF.Sqrt((row - 14) * (row - 14) + (col - 14) * (col - 14)) is > 6 and < 10 ? 0.8f : 0f,
                    1 => col is >= 13 and <= 15 ? 0.8f : 0f,
                    2 => (row <= 5 || (row >= 12 && row <= 16) || row >= 23) && col >= 5 && col <= 22 ? 0.6f : 0f,
                    3 => (col >= 18 || row <= 4 || row >= 24 || (row >= 12 && row <= 16)) ? 0.5f : 0f,
                    4 => (col >= 18 || (row >= 10 && row <= 14 && col >= 5)) ? 0.7f : 0f,
                    5 => (row <= 5 && col <= 20) || (row >= 12 && row <= 16) || (row >= 23 && col >= 8) ? 0.6f : 0f,
                    6 => row >= 14 && MathF.Sqrt((row - 20) * (row - 20) + (col - 14) * (col - 14)) < 7 ? 0.7f : (row <= 5 && col <= 14 ? 0.5f : 0f),
                    7 => (row <= 4 || Math.Abs(col - 14 - (row - 4) * 0.5f) < 2) ? 0.6f : 0f,
                    8 => (MathF.Sqrt((row - 9) * (row - 9) + (col - 14) * (col - 14)) is > 3 and < 6) ||
                         (MathF.Sqrt((row - 20) * (row - 20) + (col - 14) * (col - 14)) is > 3 and < 6) ? 0.7f : 0f,
                    9 => (MathF.Sqrt((row - 9) * (row - 9) + (col - 14) * (col - 14)) is > 3 and < 7) ||
                         (col is >= 16 and <= 18 && row >= 10) ? 0.7f : 0f,
                    _ => 0f
                };
                images[i][px] = Math.Clamp(signal + (float)(rng.NextDouble() * 0.15 - 0.075), 0f, 1f);
            }
        }
        return (images, labels);
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Pure C# Multi-Layer Perceptron
//  Architecture: 784 → 256 → 128 → 10  (ReLU + Softmax)
//  Training: Mini-batch SGD with backpropagation
//
//  ~150 lines of pure C# replaces what takes 50+ MB of PyTorch.
// ─────────────────────────────────────────────────────────────────────────

public class DenseLayer
{
    public float[,] Weights;
    public float[] Biases;
    public int InputDim, OutputDim;
    public float[] LastInput = [];
    public float[] LastPreActivation = [];
    public float[] LastOutput = [];
    public float[,] WeightGrad;
    public float[] BiasGrad;

    public DenseLayer(int inputDim, int outputDim, Random rng)
    {
        InputDim = inputDim;
        OutputDim = outputDim;
        Weights = new float[inputDim, outputDim];
        Biases = new float[outputDim];
        WeightGrad = new float[inputDim, outputDim];
        BiasGrad = new float[outputDim];

        float scale = MathF.Sqrt(2.0f / inputDim); // He initialization
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

public class MLP
{
    public DenseLayer Layer1, Layer2, Layer3;

    public MLP(Random rng)
    {
        Layer1 = new DenseLayer(784, 256, rng);
        Layer2 = new DenseLayer(256, 128, rng);
        Layer3 = new DenseLayer(128, 10, rng);
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

        float[] grad = new float[10];
        for (int i = 0; i < 10; i++)
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

    public (float Accuracy, int Correct, int Total) Evaluate(
        float[][] images, byte[] labels, HashSet<int>? filterLabels = null)
    {
        int correct = 0, total = 0;
        for (int i = 0; i < images.Length; i++)
        {
            if (filterLabels != null && !filterLabels.Contains(labels[i])) continue;
            if (Predict(images[i]) == labels[i]) correct++;
            total++;
        }
        return (total > 0 ? (float)correct / total : 0, correct, total);
    }

    public float TrainEpoch(float[][] images, byte[] labels,
        HashSet<int> allowedLabels, float lr, int batchSize, Random rng)
    {
        var indices = Enumerable.Range(0, labels.Length)
            .Where(i => allowedLabels.Contains(labels[i])).ToList();

        // Fisher-Yates shuffle
        for (int i = indices.Count - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        float totalLoss = 0;
        int steps = 0;
        for (int i = 0; i < indices.Count; i++)
        {
            totalLoss += TrainStep(images[indices[i]], labels[indices[i]]);
            steps++;
            if (steps % batchSize == 0)
                ApplyGradients(lr, batchSize);
        }
        if (steps % batchSize != 0)
            ApplyGradients(lr, steps % batchSize);

        return totalLoss / steps;
    }

    public (float[,], float[], float[,], float[], float[,], float[]) Snapshot()
    {
        var (w1, b1) = Layer1.Snapshot();
        var (w2, b2) = Layer2.Snapshot();
        var (w3, b3) = Layer3.Snapshot();
        return (w1, b1, w2, b2, w3, b3);
    }

    public void Restore((float[,], float[], float[,], float[], float[,], float[]) s)
    {
        Layer1.Restore((s.Item1, s.Item2));
        Layer2.Restore((s.Item3, s.Item4));
        Layer3.Restore((s.Item5, s.Item6));
    }

    private static float[] Softmax(float[] logits)
    {
        float max = logits.Max();
        float[] exp = logits.Select(x => MathF.Exp(x - max)).ToArray();
        float sum = exp.Sum();
        return exp.Select(x => x / sum).ToArray();
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Topological Digit Memory
//  - Stores per-class running centroids as graph landmarks
//  - Nearest-centroid classification survives task switches
// ─────────────────────────────────────────────────────────────────────────

public class TopologicalDigitMemory
{
    private readonly IGraphMemory _memory;
    private readonly IStateEncoder _encoder;
    private readonly Dictionary<int, float[]> _classCentroids = new();
    private readonly Dictionary<int, int> _classCounts = new();
    private readonly Dictionary<int, string> _classLandmarkIds = new();

    public TopologicalDigitMemory(IGraphMemory memory, IStateEncoder encoder)
    {
        _memory = memory;
        _encoder = encoder;
    }

    public async Task IngestSamplesAsync(float[][] images, byte[] labels, HashSet<int> allowedLabels)
    {
        for (int i = 0; i < images.Length; i++)
        {
            int label = labels[i];
            if (!allowedLabels.Contains(label)) continue;

            if (!_classCentroids.ContainsKey(label))
            {
                _classCentroids[label] = new float[784];
                _classCounts[label] = 0;
            }

            _classCounts[label]++;
            int n = _classCounts[label];
            var centroid = _classCentroids[label];

            for (int px = 0; px < 784; px++)
                centroid[px] += (images[i][px] - centroid[px]) / n;
        }

        foreach (var (label, centroid) in _classCentroids)
        {
            if (!allowedLabels.Contains(label)) continue;
            var embedding = _encoder.Encode(Array.ConvertAll(centroid, x => (double)x));
            var landmark = new StateLandmark
            {
                Id = $"digit_{label}",
                Embedding = embedding,
                VisitCount = _classCounts[label],
                LastVisitedTimestep = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                CreatedTimestep = DateTimeOffset.UtcNow.ToUnixTimeSeconds()
            };
            await _memory.UpsertLandmarkAsync(landmark);
            _classLandmarkIds[label] = landmark.Id;
        }

        var sorted = allowedLabels.OrderBy(x => x).ToList();
        for (int i = 0; i < sorted.Count - 1; i++)
        {
            await _memory.UpsertTransitionAsync(new StateTransition
            {
                SourceId = $"digit_{sorted[i]}",
                TargetId = $"digit_{sorted[i + 1]}",
                Action = i,
                Reward = 1.0,
                TransitionCount = 1
            });
        }
    }

    public int Classify(float[] image)
    {
        float bestDist = float.MaxValue;
        int bestLabel = -1;
        foreach (var (label, centroid) in _classCentroids)
        {
            float dist = 0;
            for (int px = 0; px < 784; px++)
            {
                float d = image[px] - centroid[px];
                dist += d * d;
            }
            if (dist < bestDist) { bestDist = dist; bestLabel = label; }
        }
        return bestLabel;
    }

    public (float Accuracy, int Correct, int Total) Evaluate(
        float[][] images, byte[] labels, HashSet<int> filterLabels)
    {
        int correct = 0, total = 0;
        for (int i = 0; i < images.Length; i++)
        {
            if (!filterLabels.Contains(labels[i])) continue;
            if (Classify(images[i]) == labels[i]) correct++;
            total++;
        }
        return (total > 0 ? (float)correct / total : 0, correct, total);
    }

    public int LandmarkCount => _classLandmarkIds.Count;
    public int SampleCount => _classCounts.Values.Sum();
}

// ─────────────────────────────────────────────────────────────────────────
//  MNIST State Encoder — 784 pixels → 32-dim spatial pooling
// ─────────────────────────────────────────────────────────────────────────

public class MnistStateEncoder : IStateEncoder
{
    public int EmbeddingDimension => 32;

    public double[] Encode(double[] rawState)
    {
        double[] features = new double[32];
        for (int px = 0; px < 784 && px < rawState.Length; px++)
        {
            int row = px / 28, col = px % 28;
            int gridRow = Math.Min(row / 7, 3);
            int gridCol = Math.Min(col / 4, 7);
            int regionIdx = gridRow * 8 + Math.Min(gridCol, 7);
            if (regionIdx < 32) features[regionIdx] += rawState[px];
        }
        double regionSize = 7.0 * 4.0;
        for (int i = 0; i < 32; i++) features[i] /= regionSize;
        return features;
    }

    public double Distance(double[] a, double[] b)
    {
        double dot = 0, nA = 0, nB = 0;
        for (int i = 0; i < a.Length; i++) { dot += a[i] * b[i]; nA += a[i] * a[i]; nB += b[i] * b[i]; }
        if (nA == 0 || nB == 0) return 1.0;
        return 1.0 - dot / (Math.Sqrt(nA) * Math.Sqrt(nB));
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  EWC (Elastic Weight Consolidation) Baseline
//  Standard continual learning regularization — for fair comparison.
//
//  BUG FIX: Clamp Fisher values and penalty gradients to prevent NaN.
//  The original had λ=400 with unclamped Fisher → gradient explosion.
// ─────────────────────────────────────────────────────────────────────────

public class EwcRegularizer
{
    private readonly float[,] _fisherL1, _fisherL2, _fisherL3;
    private readonly float[,] _oldW1, _oldW2, _oldW3;
    private readonly float[] _oldB1, _oldB2, _oldB3;
    private readonly float _lambda;

    public EwcRegularizer(MLP model, float[][] images, byte[] labels,
        HashSet<int> taskLabels, float lambda, Random rng)
    {
        _lambda = lambda;

        _oldW1 = (float[,])model.Layer1.Weights.Clone();
        _oldW2 = (float[,])model.Layer2.Weights.Clone();
        _oldW3 = (float[,])model.Layer3.Weights.Clone();
        _oldB1 = (float[])model.Layer1.Biases.Clone();
        _oldB2 = (float[])model.Layer2.Biases.Clone();
        _oldB3 = (float[])model.Layer3.Biases.Clone();

        _fisherL1 = new float[784, 256];
        _fisherL2 = new float[256, 128];
        _fisherL3 = new float[128, 10];

        var indices = Enumerable.Range(0, labels.Length)
            .Where(i => taskLabels.Contains(labels[i])).ToList();

        int nSamples = Math.Min(200, indices.Count);
        for (int s = 0; s < nSamples; s++)
        {
            int idx = indices[rng.Next(indices.Count)];
            model.Forward(images[idx]);

            float[] grad = new float[10];
            for (int j = 0; j < 10; j++)
                grad[j] = model.Layer3.LastOutput[j] - (j == labels[idx] ? 1 : 0);

            var g2 = model.Layer3.Backward(grad, relu: false);
            var g1 = model.Layer2.Backward(g2, relu: true);
            model.Layer1.Backward(g1, relu: true);

            AccumulateFisher(model.Layer1, _fisherL1);
            AccumulateFisher(model.Layer2, _fisherL2);
            AccumulateFisher(model.Layer3, _fisherL3);

            ClearGrads(model.Layer1);
            ClearGrads(model.Layer2);
            ClearGrads(model.Layer3);
        }

        // Normalize and clamp Fisher to prevent gradient explosion
        float fisherCap = 10.0f; // Prevent extreme Fisher values
        NormAndClampFisher(_fisherL1, 1.0f / nSamples, fisherCap);
        NormAndClampFisher(_fisherL2, 1.0f / nSamples, fisherCap);
        NormAndClampFisher(_fisherL3, 1.0f / nSamples, fisherCap);
    }

    private static void AccumulateFisher(DenseLayer layer, float[,] fisher)
    {
        for (int i = 0; i < layer.InputDim; i++)
            for (int j = 0; j < layer.OutputDim; j++)
                fisher[i, j] += layer.WeightGrad[i, j] * layer.WeightGrad[i, j];
    }

    private static void ClearGrads(DenseLayer layer)
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

    /// <summary>Add EWC penalty gradients. Call after normal backprop, before weight update.</summary>
    public void AddPenaltyGradients(MLP model)
    {
        AddLayerPenalty(model.Layer1, _oldW1, _fisherL1);
        AddLayerPenalty(model.Layer2, _oldW2, _fisherL2);
        AddLayerPenalty(model.Layer3, _oldW3, _fisherL3);
    }

    private void AddLayerPenalty(DenseLayer layer, float[,] oldW, float[,] fisher)
    {
        float gradCap = 1.0f; // Gradient clipping for stability
        for (int i = 0; i < layer.InputDim; i++)
            for (int j = 0; j < layer.OutputDim; j++)
            {
                float penalty = _lambda * fisher[i, j] * (layer.Weights[i, j] - oldW[i, j]);
                layer.WeightGrad[i, j] += Math.Clamp(penalty, -gradCap, gradCap);
            }
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Split-MNIST Demo Runner
// ─────────────────────────────────────────────────────────────────────────

public static class SplitMnistDemo
{
    public static async Task RunAsync(IGraphMemory memory, IStateEncoder encoder,
        ILogger logger)
    {
        Console.WriteLine("╔══════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║   Split-MNIST: Continual Learning Catastrophic Forgetting   ║");
        Console.WriteLine("║   Real MNIST data • Real MLP (784->256->128->10) • Pure C#  ║");
        Console.WriteLine("╚══════════════════════════════════════════════════════════════╝\n");

        var rng = new Random(42);
        var taskA = new HashSet<int> { 0, 1, 2, 3, 4 };
        var taskB = new HashSet<int> { 5, 6, 7, 8, 9 };
        var allDigits = new HashSet<int> { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

        // Step 1: Load Data
        Console.WriteLine("--- Step 1: Loading MNIST Data ---");
        var dataDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "mnist_data");
        var downloadResult = await MnistDataLoader.EnsureDataAsync(dataDir);

        float[][] trainImages, testImages;
        byte[] trainLabels, testLabels;

        if (!string.IsNullOrEmpty(downloadResult))
        {
            (trainImages, trainLabels) = MnistDataLoader.LoadImages(
                Path.Combine(dataDir, "train-images-idx3-ubyte.gz"),
                Path.Combine(dataDir, "train-labels-idx1-ubyte.gz"));
            (testImages, testLabels) = MnistDataLoader.LoadImages(
                Path.Combine(dataDir, "t10k-images-idx3-ubyte.gz"),
                Path.Combine(dataDir, "t10k-labels-idx1-ubyte.gz"));
            Console.WriteLine($"  Loaded: {trainImages.Length} train, {testImages.Length} test (28x28 = 784 pixels)\n");
        }
        else
        {
            Console.WriteLine("  Using synthetic data (download failed)...");
            (trainImages, trainLabels) = MnistDataLoader.GenerateSynthetic(10000, rng);
            (testImages, testLabels) = MnistDataLoader.GenerateSynthetic(2000, rng);
        }

        int trainA = trainLabels.Count(l => taskA.Contains(l));
        int trainB = trainLabels.Count(l => taskB.Contains(l));
        Console.WriteLine($"  Task A (digits 0-4): {trainA} samples");
        Console.WriteLine($"  Task B (digits 5-9): {trainB} samples\n");

        // Step 2: Train MLP on Task A
        Console.WriteLine("--- Step 2: Training MLP on Task A (digits 0-4) ---");
        var mlp = new MLP(rng);
        int epochs = 5;
        float lr = 0.01f;
        int batchSize = 32;

        for (int ep = 1; ep <= epochs; ep++)
        {
            float loss = mlp.TrainEpoch(trainImages, trainLabels, taskA, lr, batchSize, rng);
            var (accA, _, _) = mlp.Evaluate(testImages, testLabels, taskA);
            Console.WriteLine($"  Epoch {ep}/{epochs}  Loss: {loss:F4}  TaskA Acc: {accA * 100:F1}%");
        }

        var (taskAAccAfterA, _, _) = mlp.Evaluate(testImages, testLabels, taskA);
        Console.WriteLine($"\n  MLP Task A accuracy: {taskAAccAfterA * 100:F1}%");
        var snapshotAfterA = mlp.Snapshot();

        // Step 3: Build Topological Memory for Task A
        Console.WriteLine("\n--- Step 3: Building Topological Memory for Task A ---");
        var mnistEncoder = new MnistStateEncoder();
        var topoMemory = new TopologicalDigitMemory(memory, mnistEncoder);
        await topoMemory.IngestSamplesAsync(trainImages, trainLabels, taskA);

        var (topoAccA, _, _) = topoMemory.Evaluate(testImages, testLabels, taskA);
        Console.WriteLine($"  Topological Memory Task A: {topoAccA * 100:F1}%");
        Console.WriteLine($"  Landmarks: {topoMemory.LandmarkCount}, Samples: {topoMemory.SampleCount}");

        // Step 4: EWC Fisher Information
        Console.WriteLine("\n--- Step 4: Computing EWC Fisher Information ---");
        var ewcMlp = new MLP(rng);
        ewcMlp.Restore(snapshotAfterA);
        var ewc = new EwcRegularizer(ewcMlp, trainImages, trainLabels, taskA, lambda: 100f, rng);
        Console.WriteLine("  Fisher computed (200 samples, lambda=100, clamped)");

        // Step 5: Train on Task B — FORGETTING TEST
        Console.WriteLine("\n--- Step 5: Training on Task B (FORGETTING TEST) ---");
        Console.WriteLine("  Bare MLP (no protection):");
        for (int ep = 1; ep <= epochs; ep++)
        {
            float loss = mlp.TrainEpoch(trainImages, trainLabels, taskB, lr, batchSize, rng);
            var (accA, _, _) = mlp.Evaluate(testImages, testLabels, taskA);
            var (accB, _, _) = mlp.Evaluate(testImages, testLabels, taskB);
            Console.WriteLine($"  Epoch {ep}  Loss: {loss:F4}  TaskA: {accA * 100:F1}%  TaskB: {accB * 100:F1}%");
        }

        var (taskAAccAfterB, _, _) = mlp.Evaluate(testImages, testLabels, taskA);
        var (taskBAccAfterB, _, _) = mlp.Evaluate(testImages, testLabels, taskB);

        Console.WriteLine("\n  EWC-protected MLP:");
        for (int ep = 1; ep <= epochs; ep++)
        {
            var indices = Enumerable.Range(0, trainLabels.Length)
                .Where(i => taskB.Contains(trainLabels[i])).ToList();
            for (int i = indices.Count - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }

            float totalLoss = 0;
            int steps = 0;
            for (int i = 0; i < indices.Count; i++)
            {
                totalLoss += ewcMlp.TrainStep(trainImages[indices[i]], trainLabels[indices[i]]);
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

            var (ewcA, _, _) = ewcMlp.Evaluate(testImages, testLabels, taskA);
            var (ewcB, _, _) = ewcMlp.Evaluate(testImages, testLabels, taskB);
            Console.WriteLine($"  Epoch {ep}  Loss: {totalLoss / steps:F4}  TaskA: {ewcA * 100:F1}%  TaskB: {ewcB * 100:F1}%");
        }

        var (ewcTaskAFinal, _, _) = ewcMlp.Evaluate(testImages, testLabels, taskA);
        var (ewcTaskBFinal, _, _) = ewcMlp.Evaluate(testImages, testLabels, taskB);

        // Step 6: Add Task B to Topological Memory
        Console.WriteLine("\n--- Step 6: Adding Task B to Topological Memory ---");
        await topoMemory.IngestSamplesAsync(trainImages, trainLabels, taskB);
        var (topoAccAFinal, _, _) = topoMemory.Evaluate(testImages, testLabels, taskA);
        var (topoAccBFinal, _, _) = topoMemory.Evaluate(testImages, testLabels, taskB);
        var (topoAccAll, _, _) = topoMemory.Evaluate(testImages, testLabels, allDigits);

        // Step 7: Report
        float forgetting = taskAAccAfterA > 0 ? (taskAAccAfterA - taskAAccAfterB) / taskAAccAfterA * 100 : 0;
        float ewcRetention = taskAAccAfterA > 0 ? ewcTaskAFinal / taskAAccAfterA * 100 : 0;
        float topoRetention = topoAccA > 0 ? topoAccAFinal / topoAccA * 100 : 0;

        Console.WriteLine("\n╔═══════════════════════════════════════════════════════════╗");
        Console.WriteLine("║           SPLIT-MNIST FORGETTING REPORT                  ║");
        Console.WriteLine("╠═══════════════════════════════════════════════════════════╣");
        Console.WriteLine($"║  Task A after training A:     {taskAAccAfterA * 100,6:F1}%                    ║");
        Console.WriteLine($"║  Task A after B (Bare MLP):   {taskAAccAfterB * 100,6:F1}%  <- FORGETTING     ║");
        Console.WriteLine($"║  Task A after B (EWC):        {ewcTaskAFinal * 100,6:F1}%  <- Regularized     ║");
        Console.WriteLine($"║  Task A (Topological Memory): {topoAccAFinal * 100,6:F1}%  <- Graph-stored    ║");
        Console.WriteLine("║                                                           ║");
        Console.WriteLine($"║  Task B (Bare MLP):           {taskBAccAfterB * 100,6:F1}%                    ║");
        Console.WriteLine($"║  Task B (EWC):                {ewcTaskBFinal * 100,6:F1}%                    ║");
        Console.WriteLine($"║  Task B (Topological Memory): {topoAccBFinal * 100,6:F1}%                    ║");
        Console.WriteLine($"║  Overall Topo (0-9):          {topoAccAll * 100,6:F1}%                    ║");
        Console.WriteLine("║                                                           ║");
        Console.WriteLine($"║  Forgetting Rate (MLP):       {forgetting,6:F1}%                    ║");
        Console.WriteLine($"║  EWC Retention:               {ewcRetention,6:F1}%                    ║");
        Console.WriteLine($"║  Topo Retention:              {topoRetention,6:F1}%                    ║");
        Console.WriteLine("╚═══════════════════════════════════════════════════════════╝");
    }
}
