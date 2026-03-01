namespace RichLearning.Models;

/// <summary>
/// A subgoal directive issued by the Cartographer.
/// Represents the next waypoint the worker should navigate toward.
/// </summary>
public sealed record SubgoalDirective
{
    /// <summary>Target landmark ID to navigate toward.</summary>
    public required string TargetLandmarkId { get; init; }

    /// <summary>Distance from current position (graph hops or embedding distance).</summary>
    public double Distance { get; init; }

    /// <summary>Reason for selecting this subgoal (for explainability).</summary>
    public string? Reason { get; init; }
}

/// <summary>
/// Summary statistics of the topological map at a point in time.
/// </summary>
public sealed record MapSnapshot
{
    public int LandmarkCount { get; init; }
    public int TransitionCount { get; init; }
    public int ClusterCount { get; init; }
    public double MeanNovelty { get; init; }
    public double MeanValueEstimate { get; init; }
    public double MeanPolicyEntropy { get; init; }
    public IReadOnlyList<ClusterStats> Clusters { get; init; } = [];
}

/// <summary>
/// Statistics for a single cluster (community) in the graph.
/// </summary>
public sealed record ClusterStats
{
    public int ClusterId { get; init; }
    public int LandmarkCount { get; init; }
    public double MeanValue { get; init; }
    public double MeanNovelty { get; init; }
}
