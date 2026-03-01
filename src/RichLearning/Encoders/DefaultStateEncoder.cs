using RichLearning.Abstractions;

namespace RichLearning.Encoders;

/// <summary>
/// Default state encoder using cosine distance.
/// Suitable for domains where raw numeric features are already meaningful.
///
/// Domain-specific encoders should override this for better performance:
///   - Chess: ZobristEncoder (hash-based, Hamming distance)
///   - Audio: MfccEncoder (MFCC extraction, cosine distance)
///   - Vision: CnnEncoder (learned embeddings, cosine distance)
///   - LLM: SentenceEncoder (transformer embeddings, cosine distance)
/// </summary>
public sealed class DefaultStateEncoder : IStateEncoder
{
    private readonly int _dimension;

    public DefaultStateEncoder(int dimension)
    {
        if (dimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(dimension), "Dimension must be positive");
        _dimension = dimension;
    }

    public int EmbeddingDimension => _dimension;

    /// <summary>
    /// Identity encoding — returns the raw state unchanged.
    /// Domain encoders should transform raw observations into meaningful embeddings.
    /// </summary>
    public double[] Encode(double[] rawState)
    {
        if (rawState.Length != _dimension)
            throw new ArgumentException(
                $"Expected {_dimension}-dimensional input, got {rawState.Length}");
        return rawState;
    }

    /// <summary>
    /// Cosine distance: 1 - cos_sim(a, b).
    /// Returns 0.0 for identical vectors, ~2.0 for opposite vectors.
    /// </summary>
    public double Distance(double[] a, double[] b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must have equal length");

        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        double denom = Math.Sqrt(normA) * Math.Sqrt(normB);
        if (denom < 1e-12) return 1.0; // degenerate case

        return 1.0 - dot / denom;
    }
}
