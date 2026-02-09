namespace RichLearning.Abstractions;

/// <summary>
/// Encodes raw environment states into dense vector embeddings φ(s) ∈ ℝ^d.
/// 
/// Implementations may use learned encoders, autoencoders, random projections,
/// spatial pooling, or domain-specific feature extractors.
/// </summary>
public interface IStateEncoder
{
    /// <summary>Dimensionality of the output embedding.</summary>
    int EmbeddingDimension { get; }

    /// <summary>
    /// Encode a numeric state observation into a dense vector φ(s) ∈ ℝ^d.
    /// </summary>
    double[] Encode(double[] rawState);

    /// <summary>
    /// Encode a domain-specific state observation into a dense vector.
    /// Default implementation converts to double[] if possible.
    /// </summary>
    double[] Encode(object rawState) => rawState switch
    {
        double[] arr => Encode(arr),
        float[] fArr => Encode(Array.ConvertAll(fArr, x => (double)x)),
        _ => throw new ArgumentException(
            $"Cannot encode {rawState.GetType().Name}. Override Encode(object) for custom types.")
    };

    /// <summary>
    /// Compute distance between two embeddings. Returns 0 for identical vectors.
    /// Default: cosine distance.
    /// </summary>
    double Distance(double[] a, double[] b);
}
