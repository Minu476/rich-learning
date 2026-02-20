namespace RichLearning.Learning;

/// <summary>
/// Agent efficiency tracker: identifies star/efficient/normal/struggling agents
/// and propagates successful strategies across the swarm.
///
/// "Copy the best, help the rest" — high-performers share their learned
/// graph structures with underperformers.
///
/// From the RLCartographer paper:
///   - Star agents: exceptional performance (top 10%)
///   - Efficient agents: above average
///   - Normal agents: around average
///   - Struggling agents: below threshold
///
/// Star agents' teaching weight is amplified (1.5×).
/// Struggling agents receive pattern copies from stars.
///
/// This is domain-agnostic: works for any multi-agent deployment.
/// </summary>
public sealed class AgentEfficiencyTracker
{
    private readonly Dictionary<string, AgentRecord> _agents = new(StringComparer.Ordinal);

    /// <summary>Threshold for star status (top percentile).</summary>
    public double StarPercentile { get; init; } = 0.9;

    /// <summary>Threshold for struggling status (bottom percentile).</summary>
    public double StrugglingPercentile { get; init; } = 0.25;

    /// <summary>Teaching weight boost for star agents.</summary>
    public float StarTeachingWeight { get; init; } = 1.5f;

    /// <summary>Record a performance observation for an agent.</summary>
    public void RecordPerformance(string agentId, double reward)
    {
        if (!_agents.TryGetValue(agentId, out var record))
        {
            record = new AgentRecord(agentId);
            _agents[agentId] = record;
        }

        record.TotalReward += reward;
        record.EpisodeCount++;
        record.AverageReward = record.TotalReward / record.EpisodeCount;
    }

    /// <summary>Classify all agents into tiers based on current performance.</summary>
    public IReadOnlyDictionary<string, AgentTier> ClassifyAll()
    {
        if (_agents.Count == 0)
            return new Dictionary<string, AgentTier>();

        var sorted = _agents.Values.OrderBy(a => a.AverageReward).ToList();
        int starThreshold = (int)(sorted.Count * StarPercentile);
        int strugglingThreshold = (int)(sorted.Count * StrugglingPercentile);

        var result = new Dictionary<string, AgentTier>(StringComparer.Ordinal);
        for (int i = 0; i < sorted.Count; i++)
        {
            var tier = i >= starThreshold ? AgentTier.Star
                     : i < strugglingThreshold ? AgentTier.Struggling
                     : i >= sorted.Count / 2 ? AgentTier.Efficient
                     : AgentTier.Normal;

            sorted[i].Tier = tier;
            result[sorted[i].AgentId] = tier;
        }

        return result;
    }

    /// <summary>Get the teaching weights for all agents (star agents get amplified).</summary>
    public IReadOnlyDictionary<string, float> GetTeachingWeights()
    {
        ClassifyAll(); // ensure tiers are current
        return _agents.ToDictionary(
            kv => kv.Key,
            kv => kv.Value.Tier == AgentTier.Star ? StarTeachingWeight : 1.0f);
    }

    /// <summary>Get agents in a specific tier.</summary>
    public IReadOnlyList<string> GetAgentsInTier(AgentTier tier)
    {
        ClassifyAll();
        return _agents.Values.Where(a => a.Tier == tier).Select(a => a.AgentId).ToList();
    }

    /// <summary>Agent performance record.</summary>
    private sealed class AgentRecord(string agentId)
    {
        public string AgentId { get; } = agentId;
        public double TotalReward { get; set; }
        public int EpisodeCount { get; set; }
        public double AverageReward { get; set; }
        public AgentTier Tier { get; set; } = AgentTier.Normal;
    }
}

/// <summary>Agent tier classification.</summary>
public enum AgentTier
{
    Star,
    Efficient,
    Normal,
    Struggling
}
