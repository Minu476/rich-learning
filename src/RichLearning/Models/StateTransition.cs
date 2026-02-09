namespace RichLearning.Models;

/// <summary>
/// Represents a directed edge (transition) between two StateLandmarks.
/// Carries action, reward, and traversal statistics.
/// </summary>
public sealed record StateTransition
{
    /// <summary>Source landmark ID.</summary>
    public required string SourceId { get; init; }

    /// <summary>Target landmark ID.</summary>
    public required string TargetId { get; init; }

    /// <summary>The primary action taken to traverse this edge.</summary>
    public required int Action { get; init; }

    /// <summary>
    /// Action counts: action → number of times this action led to this transition.
    /// </summary>
    public Dictionary<int, int> ActionCounts { get; init; } = new();

    /// <summary>
    /// Compute action distribution for this edge: action → probability.
    /// </summary>
    public IReadOnlyDictionary<int, double> GetActionDistribution()
    {
        int total = ActionCounts.Values.Sum();
        if (total == 0)
            return new Dictionary<int, double> { [Action] = 1.0 };
        return ActionCounts.ToDictionary(kv => kv.Key, kv => (double)kv.Value / total);
    }

    /// <summary>Observed reward for this transition (running mean).</summary>
    public double Reward { get; set; }

    /// <summary>Reward variance — for uncertainty estimation.</summary>
    public double RewardVariance { get; set; }

    /// <summary>Number of times this exact transition was executed.</summary>
    public int TransitionCount { get; set; } = 1;

    /// <summary>Empirical success rate of reaching the target landmark via this action.</summary>
    public double SuccessRate { get; set; } = 1.0;

    /// <summary>
    /// Confidence in this edge — grows with TransitionCount, decays with variance.
    /// </summary>
    public double Confidence { get; set; } = 0.5;

    /// <summary>Temporal distance (primitive steps) between the two landmarks.</summary>
    public int TemporalDistance { get; set; } = 1;

    /// <summary>Most recent TD-error observed on this edge (for prioritised replay).</summary>
    public double TdError { get; set; }

    /// <summary>Timestep when this edge was last used for training.</summary>
    public long LastTrainedTimestep { get; set; }

    /// <summary>Whether this is a macro-edge (compressed skill).</summary>
    public bool IsMacroEdge { get; init; }

    /// <summary>For macro-edges: the sequence of intermediate landmark IDs.</summary>
    public IReadOnlyList<string> MacroPath { get; init; } = [];
}
