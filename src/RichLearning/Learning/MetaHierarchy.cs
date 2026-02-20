using RichLearning.Models;

namespace RichLearning.Learning;

/// <summary>
/// Self-organizing recursive meta-hierarchy.
///
/// This is the "compound interest" engine of Rich Learning.
/// Low-level patterns compose into subgraphs (skills), which compose into
/// higher-level strategies, which compose into even higher-level plans.
///
/// Core spawning rule (from FSDE UNIVERSAL_ACTOR_SPEC.md):
///   if |subgraphs at level N| >= 2: spawn level N+1
///
/// Backward teaching:
///   1. Active patterns reinforced at full strength
///   2. Neighbors in same subgraph at attenuated strength (0.5×)
///   3. Recurse upward through parent levels
///
/// The meta-hierarchy grows organically based on detected complexity
/// rather than being pre-specified. This is domain-agnostic — it works
/// for chess openings, robot navigation skills, LLM reasoning patterns,
/// trading strategies, etc.
///
/// IMPORTANT: The meta-hierarchy does NOT affect the hot path (Query).
/// It is a learning scaffold that discovers structure in the pattern stream.
/// </summary>
public sealed class MetaHierarchy
{
    private readonly Dictionary<int, MetaLevel> _levels = new();
    private int _observationCount;

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

    /// <summary>
    /// Clustering strategy for subgraph formation.
    /// Default: 4-char prefix of state signature (simple but effective for v1).
    /// Domain implementations can override via the constructor.
    /// </summary>
    public Func<Pattern, string> ClusterKeyExtractor { get; init; }
        = p => p.StateSignature.Length >= 4
            ? p.StateSignature[..4]
            : p.StateSignature;

    public int LevelCount => _levels.Count;
    public int TotalPatternCount => _levels.Values.Sum(l => l.PatternCount);

    /// <summary>
    /// Observe a new pattern. Periodically triggers subgraph formation.
    /// </summary>
    public void Observe(Pattern pattern)
    {
        if (!_levels.TryGetValue(1, out var level1))
        {
            level1 = new MetaLevel(1);
            _levels[1] = level1;
        }

        level1.AddOrUpdatePattern(pattern);
        _observationCount++;

        if (_observationCount % SubgraphFormationInterval == 0)
        {
            FormSubgraphs(1);
        }
    }

    /// <summary>
    /// Backward teaching: propagate a reward signal through the hierarchy.
    ///
    /// From DAPSA v2.1:
    ///   1. Active patterns reinforced at full strength
    ///   2. Neighbors in same subgraph at attenuated strength
    ///   3. Recurse upward through meta-levels
    /// </summary>
    public void TeachBackward(double reward, IReadOnlyList<string> activePatternIds)
    {
        if (!_levels.TryGetValue(1, out var level1))
            return;

        var activeSet = new HashSet<string>(activePatternIds);

        // Phase 1: Reinforce active patterns at full strength
        foreach (var patternId in activePatternIds)
        {
            if (level1.TryGetPattern(patternId, out var pattern))
            {
                pattern.Reinforce(reward);
            }
        }

        // Phase 2: Attenuated reinforcement to neighbors in same subgraph
        foreach (var subgraph in level1.Subgraphs)
        {
            bool hasActive = subgraph.Patterns.Keys.Any(pid => activeSet.Contains(pid));
            if (!hasActive) continue;

            foreach (var (pid, pattern) in subgraph.Patterns)
            {
                if (!activeSet.Contains(pid))
                {
                    pattern.Reinforce(reward * NeighborAttenuation);
                }
            }
        }

        // Phase 3: Recurse upward through parent levels
        PropagateUpward(2, reward, activePatternIds);
    }

    /// <summary>
    /// Form subgraphs at a given level by clustering patterns.
    /// When 2+ subgraphs form, spawn the next level.
    /// </summary>
    private void FormSubgraphs(int level)
    {
        if (level > MaxDepth) return;
        if (!_levels.TryGetValue(level, out var metaLevel)) return;

        var eligible = metaLevel.Patterns
            .Where(p => p.TimesSeen >= MinObservationsPerPattern)
            .ToList();

        // Group by cluster key (configurable)
        var groups = eligible
            .GroupBy(ClusterKeyExtractor)
            .Where(g => g.Count() >= MinPatternsForCluster);

        foreach (var group in groups)
        {
            var subgraph = new Subgraph(group.Key, level);
            foreach (var pattern in group)
            {
                subgraph.AddPattern(pattern);
            }
            metaLevel.AddSubgraph(subgraph);
        }

        // Spawning rule: 2+ subgraphs at level N → spawn level N+1
        if (metaLevel.SubgraphCount >= 2 && !_levels.ContainsKey(level + 1))
        {
            var nextLevel = new MetaLevel(level + 1);
            _levels[level + 1] = nextLevel;

            // Promote centroids from current level subgraphs
            foreach (var sg in metaLevel.Subgraphs)
            {
                if (sg.CentroidId is not null && metaLevel.TryGetPattern(sg.CentroidId, out var centroid))
                {
                    nextLevel.AddOrUpdatePattern(centroid);
                }
            }
        }
    }

    private void PropagateUpward(int level, double reward, IReadOnlyList<string> sourcePatternIds)
    {
        if (level > MaxDepth || !_levels.TryGetValue(level, out var metaLevel))
            return;

        foreach (var patternId in sourcePatternIds)
        {
            if (metaLevel.TryGetPattern(patternId, out var pattern))
            {
                pattern.Reinforce(reward * NeighborAttenuation);
            }
        }

        PropagateUpward(level + 1, reward * NeighborAttenuation, sourcePatternIds);
    }

    /// <summary>Get diagnostic info about the hierarchy.</summary>
    public IReadOnlyDictionary<int, int> GetLevelSizes()
    {
        return _levels.ToDictionary(kv => kv.Key, kv => kv.Value.PatternCount);
    }

    /// <summary>Internal level in the meta-hierarchy. Holds patterns and subgraphs.</summary>
    private sealed class MetaLevel
    {
        public int Level { get; }
        private readonly Dictionary<string, Pattern> _patterns = new();
        private readonly List<Subgraph> _subgraphs = new();

        public MetaLevel(int level) => Level = level;

        public void AddOrUpdatePattern(Pattern pattern) =>
            _patterns[pattern.Id] = pattern;

        public bool TryGetPattern(string id, out Pattern pattern) =>
            _patterns.TryGetValue(id, out pattern!);

        public void AddSubgraph(Subgraph sg) => _subgraphs.Add(sg);

        public IReadOnlyList<Pattern> Patterns => _patterns.Values.ToList();
        public IReadOnlyList<Subgraph> Subgraphs => _subgraphs;
        public int SubgraphCount => _subgraphs.Count;
        public int PatternCount => _patterns.Count;
    }
}
