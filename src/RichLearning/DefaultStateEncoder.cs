using RichLearning.Abstractions;

namespace RichLearning;

/// <summary>
/// A simple identity/pass-through encoder for prototyping.
/// Uses cosine distance for embedding comparison.
/// </summary>
public sealed class DefaultStateEncoder : IStateEncoder
{
    public int EmbeddingDimension { get; }

    public DefaultStateEncoder(int embeddingDimension = 64)
    {
        EmbeddingDimension = embeddingDimension;
    }

    public double[] Encode(double[] rawState)
    {
        var result = new double[EmbeddingDimension];
        var copyLen = Math.Min(rawState.Length, EmbeddingDimension);
        Array.Copy(rawState, result, copyLen);
        return result;
    }

    public double Distance(double[] a, double[] b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Embedding dimensions must match.");

        double dotProduct = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        if (normA == 0 || normB == 0) return 1.0;
        double cosineSimilarity = dotProduct / (Math.Sqrt(normA) * Math.Sqrt(normB));
        return 1.0 - cosineSimilarity;
    }
}
