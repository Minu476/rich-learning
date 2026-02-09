namespace RichLearning.Models;

/// <summary>
/// A snapshot of the graph's state — the agent's situational awareness of the map.
/// </summary>
public sealed record MapSnapshot
{
    /// <summary>Total number of landmark nodes in the graph.</summary>
    public int TotalLandmarks { get; init; }

    /// <summary>Total number of transition edges in the graph.</summary>
    public int TotalTransitions { get; init; }

    /// <summary>Average visit count across all landmarks.</summary>
    public double MeanVisitCount { get; init; }

    /// <summary>Average novelty score across all landmarks.</summary>
    public double MeanNoveltyScore { get; init; }

    /// <summary>Number of distinct cluster regions discovered.</summary>
    public int ClusterCount { get; init; }

    /// <summary>Per-cluster statistics.</summary>
    public IReadOnlyDictionary<int, ClusterStats> Clusters { get; init; }
        = new Dictionary<int, ClusterStats>();

    /// <summary>Frontier (leaf) landmarks.</summary>
    public IReadOnlyList<StateLandmark> FrontierNodes { get; init; } = [];

    /// <summary>Bottleneck (bridge) landmarks.</summary>
    public IReadOnlyList<StateLandmark> BottleneckNodes { get; init; } = [];

    /// <summary>Stale (unvisited) landmarks at risk of forgetting.</summary>
    public IReadOnlyList<StateLandmark> StaleNodes { get; init; } = [];

    /// <summary>Current timestep when this snapshot was taken.</summary>
    public long Timestep { get; init; }
}

/// <summary>
/// Aggregated statistics for a single cluster region.
/// </summary>
public sealed record ClusterStats
{
    public int ClusterId { get; init; }
    public int NodeCount { get; init; }
    public double MeanVisitCount { get; init; }
    public double MeanNoveltyScore { get; init; }
    public double MeanValueEstimate { get; init; }
}
