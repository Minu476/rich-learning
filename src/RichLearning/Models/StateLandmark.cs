namespace RichLearning.Models;

/// <summary>
/// Represents a landmark node in the topological state-space graph.
/// Landmarks are sparsely sampled states that form the skeleton of the "map."
///
/// From the Rich Learning paper (Definition 3):
///   A Landmark ℓ = (id, e, n, V, ν, σ, c) where:
///     e ∈ ℝ^d  : Dense embedding φ(s) from a state encoder
///     n ∈ ℕ    : Visit count (familiarity)
///     V ∈ ℝ    : Running value estimate
///     ν ∈ [0,1]: Novelty score (decays with visits)
///     σ ∈ [0,1]: Uncertainty (inverse confidence)
///     c ∈ ℕ    : Cluster ID (community assignment)
///
/// Landmarks are append-only: new experiences add new landmarks,
/// never overwrite existing ones. This is the Zero Forgetting guarantee.
/// </summary>
public sealed record StateLandmark
{
    /// <summary>Unique identifier (SHA-256 hash of abstracted state).</summary>
    public required string Id { get; init; }

    /// <summary>Dense vector embedding φ(s) ∈ ℝ^d produced by the state encoder.</summary>
    public required double[] Embedding { get; init; }

    /// <summary>Number of times this landmark has been visited.</summary>
    public int VisitCount { get; set; }

    /// <summary>Running estimate of V(s) for this landmark.</summary>
    public double ValueEstimate { get; set; }

    /// <summary>Novelty score — decays with visits, used for exploration priority.</summary>
    public double NoveltyScore { get; set; } = 1.0;

    /// <summary>
    /// Uncertainty score — inverse of confidence.
    /// High uncertainty = need more exploration from this node.
    /// </summary>
    public double UncertaintyScore { get; set; } = 1.0;

    /// <summary>Cluster / region ID assigned by community detection.</summary>
    public int ClusterId { get; set; }

    /// <summary>
    /// Hierarchical level: 0 = fine-grained, 1+ = abstracted regions.
    /// </summary>
    public int HierarchyLevel { get; set; }

    /// <summary>
    /// For abstract nodes (HierarchyLevel > 0): IDs of child nodes in the level below.
    /// </summary>
    public IReadOnlyList<string> ChildNodeIds { get; init; } = [];

    /// <summary>Timestep at which this landmark was last visited.</summary>
    public long LastVisitedTimestep { get; set; }

    /// <summary>Timestep at which this landmark was created.</summary>
    public long CreatedTimestep { get; init; }

    // ── Policy Summary ──

    /// <summary>
    /// Action counts: action → number of times taken from this node.
    /// Used to compute empirical policy π(a|s) at this landmark.
    /// </summary>
    public Dictionary<int, int> ActionCounts { get; init; } = new();

    /// <summary>
    /// Compute the empirical policy distribution at this landmark.
    /// </summary>
    public IReadOnlyDictionary<int, double> GetPolicyDistribution()
    {
        int total = ActionCounts.Values.Sum();
        if (total == 0) return new Dictionary<int, double>();
        return ActionCounts.ToDictionary(kv => kv.Key, kv => (double)kv.Value / total);
    }

    /// <summary>
    /// Policy entropy H(π) at this landmark — higher = more uniform.
    /// </summary>
    public double PolicyEntropy
    {
        get
        {
            var dist = GetPolicyDistribution();
            if (dist.Count == 0) return 0;
            return -dist.Values
                .Where(p => p > 0)
                .Sum(p => p * Math.Log(p));
        }
    }

    // ── Episodic Traces ──

    /// <summary>Compressed episodic traces passing through this landmark.</summary>
    public List<EpisodicTrace> EpisodicTraces { get; init; } = new();

    /// <summary>Maximum number of episodic traces to retain per node.</summary>
    public const int MaxEpisodicTraces = 10;

    // ── Domain Extension ──

    /// <summary>
    /// Optional domain-specific metadata. Domain implementations can store
    /// additional properties here without modifying the base model.
    /// </summary>
    public Dictionary<string, object> Metadata { get; init; } = new();
}

/// <summary>
/// A compressed episodic trace — a short trajectory snippet anchored at a landmark.
/// </summary>
public sealed record EpisodicTrace
{
    /// <summary>Sequence of (action, reward, nextLandmarkId) tuples.</summary>
    public required IReadOnlyList<(int Action, double Reward, string NextLandmarkId)> Steps { get; init; }

    /// <summary>Cumulative return of this trace.</summary>
    public double Return { get; init; }

    /// <summary>Timestep when this trace was recorded.</summary>
    public long RecordedTimestep { get; init; }
}
