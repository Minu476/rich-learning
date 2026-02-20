namespace RichLearning.Models;

/// <summary>
/// A coherent cluster of co-activating patterns.
///
/// Subgraphs are the building blocks of the Recursive Meta Hierarchy.
/// When 2+ subgraphs form at level N, a new parent level N+1 is spawned.
///
/// Properties:
///   coherence = mean(p.confidence for p in patterns)
///   centroid = pattern with highest confidence
///
/// This is analogous to "skills" or "strategies" — groups of behaviours
/// that tend to co-occur and succeed together.
/// </summary>
public sealed class Subgraph
{
    public string Id { get; }
    public string Name { get; }
    public int Level { get; }

    private readonly Dictionary<string, Pattern> _patterns = new();
    private readonly HashSet<string> _linkedSubgraphIds = new();

    public float Coherence { get; private set; }
    public string? CentroidId { get; private set; }
    public int ActivationCount { get; private set; }

    public Subgraph(string name, int level)
    {
        Name = name;
        Level = level;
        Id = $"SG_{level}_{name[..Math.Min(8, name.Length)]}";
    }

    public void AddPattern(Pattern pattern)
    {
        _patterns[pattern.Id] = pattern;
        UpdateCoherence();
    }

    public void LinkSubgraph(string subgraphId)
    {
        _linkedSubgraphIds.Add(subgraphId);
    }

    public void RecordActivation()
    {
        ActivationCount++;
    }

    public IReadOnlyDictionary<string, Pattern> Patterns => _patterns;
    public IReadOnlyCollection<string> LinkedSubgraphIds => _linkedSubgraphIds;

    /// <summary>
    /// Coherence = mean confidence of contained patterns.
    /// Centroid = pattern with highest confidence.
    /// </summary>
    private void UpdateCoherence()
    {
        if (_patterns.Count == 0)
        {
            Coherence = 0f;
            CentroidId = null;
            return;
        }

        float sum = 0f;
        float maxConf = -1f;
        string? bestId = null;

        foreach (var p in _patterns.Values)
        {
            sum += p.Confidence;
            if (p.Confidence > maxConf)
            {
                maxConf = p.Confidence;
                bestId = p.Id;
            }
        }

        Coherence = sum / _patterns.Count;
        CentroidId = bestId;
    }
}
