namespace RichLearning.Abstractions;

/// <summary>
/// Encodes raw domain-specific observations into dense vector embeddings φ(s) ∈ ℝ^d.
///
/// Rich Learning separates WHAT to remember (graph structure) from HOW to perceive
/// (encoding). This interface defines the perception boundary.
///
/// Domain implementations:
///   - Image domains: CNN-based, pixel pooling, or pretrained encoders
///   - Audio domains: MFCC feature extraction
///   - Chess domains: Zobrist hash → fixed embedding
///   - LLM/RAG domains: MiniLM-L6-v2, ArcFace, or other sentence/entity encoders
///   - Robotics: proprioception + lidar/vision fusion
///   - Financial: regime-weighted market state features
///
/// The encoder defines the metric space in which novelty gating operates.
/// Changing the encoder changes what counts as "novel" without altering the graph.
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
    /// Override for custom domain types (images, audio buffers, board states, etc.).
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
    /// Default: cosine distance = 1 - cos_sim(a, b).
    /// Override for domain-specific metrics (e.g., Hamming for discrete hashes).
    /// </summary>
    double Distance(double[] a, double[] b);
}
