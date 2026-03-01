using RichLearning.Models;

namespace RichLearning.Learning;

/// <summary>
/// Self-organizing recursive meta-hierarchy.
///
/// Patterns compose into subgraphs (skills), which compose into higher-level
/// strategies, which compose into plans. Grows organically based on detected
/// complexity rather than being pre-specified.
///
/// <remarks>This is the public API stub. Full implementation is provided by
/// the rich-learning-base package.</remarks>
/// </summary>
public sealed class MetaHierarchy
{
    /// <summary>How often to trigger subgraph formation (in observations).</summary>
    public int SubgraphFormationInterval { get; init; } = 50;

    /// <summary>Minimum patterns in a cluster to form a subgraph.</summary>
    public int MinPatternsForCluster { get; init; } = 2;

    /// <summary>Minimum observations for a pattern to be clustering-eligible.</summary>
    public int MinObservationsPerPattern { get; init; } = 3;

    /// <summary>Attenuation factor for neighbor reinforcement.</summary>
    public float NeighborAttenuation { get; init; } = 0.5f;

    /// <summary>Maximum hierarchy depth (safety limit).</summary>
    public int MaxDepth { get; init; } = 10;

    /// <summary>Clustering strategy for subgraph formation.</summary>
    public Func<Pattern, string> ClusterKeyExtractor { get; init; }
        = p => p.ClusterKey.Length > 0
            ? p.ClusterKey
            : (p.StateSignature.Length >= 4 ? p.StateSignature[..4] : p.StateSignature);

    public int LevelCount => 0;
    public int TotalPatternCount => 0;

    /// <summary>Observe a new pattern.</summary>
    /// <remarks>Full implementation in rich-learning-base.</remarks>
    public void Observe(Pattern pattern) { }

    /// <summary>Backward teaching: propagate a reward signal through the hierarchy.</summary>
    /// <remarks>Full implementation in rich-learning-base.</remarks>
    public void TeachBackward(double reward, IReadOnlyList<string> activePatternIds) { }

    /// <summary>Get diagnostic info about the hierarchy.</summary>
    public IReadOnlyDictionary<int, int> GetLevelSizes() => new Dictionary<int, int>();
}
