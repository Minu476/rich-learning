namespace RichLearning.Learning;

/// <summary>
/// Agent efficiency tracker: identifies star/efficient/normal/struggling agents
/// and propagates successful strategies across the swarm.
///
/// <remarks>This is the public API stub. Full implementation is provided by
/// the rich-learning-base package.</remarks>
/// </summary>
public sealed class AgentEfficiencyTracker
{
    /// <summary>Threshold for star status (top percentile).</summary>
    public double StarPercentile { get; init; } = 0.9;

    /// <summary>Threshold for struggling status (bottom percentile).</summary>
    public double StrugglingPercentile { get; init; } = 0.25;

    /// <summary>Number of recent rewards to keep per agent for rolling evaluation.</summary>
    public int RewardWindowSize { get; init; } = 50;

    /// <summary>Teaching weight boost for star agents.</summary>
    public float StarTeachingWeight { get; init; } = 1.5f;

    /// <summary>Record a performance observation for an agent.</summary>
    /// <remarks>Full implementation in rich-learning-base.</remarks>
    public void RecordPerformance(string agentId, double reward) { }

    /// <summary>Classify all agents into tiers based on current performance.</summary>
    /// <remarks>Full implementation in rich-learning-base.</remarks>
    public IReadOnlyDictionary<string, AgentTier> ClassifyAll()
        => new Dictionary<string, AgentTier>();

    /// <summary>Get the teaching weights for all agents.</summary>
    /// <remarks>Full implementation in rich-learning-base.</remarks>
    public IReadOnlyDictionary<string, float> GetTeachingWeights()
        => new Dictionary<string, float>();

    /// <summary>Get agents in a specific tier.</summary>
    /// <remarks>Full implementation in rich-learning-base.</remarks>
    public IReadOnlyList<string> GetAgentsInTier(AgentTier tier) => [];

    /// <summary>Rank all agents (Star/Average/Struggler) and return classification map.</summary>
    /// <remarks>Full implementation in rich-learning-base.</remarks>
    public Dictionary<string, AgentTier> RankAgents() => new();
}

/// <summary>Agent tier classification.</summary>
public enum AgentTier
{
    Star,
    Efficient,
    Normal,
    Struggling
}
