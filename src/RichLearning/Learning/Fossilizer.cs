using RichLearning.Models;

namespace RichLearning.Learning;

/// <summary>
/// Fossilizer: converts active knowledge into passive O(1) lookup tables.
///
/// This is the Phi function from DAPSA: Phi: S^A → S^P.
/// When patterns have been observed enough times with high enough success rate,
/// they "fossilize" — transforming expensive System 2 computation into
/// cheap System 1 habit.
///
/// Three strategies:
///   1. FromEpisodes: direct (situationHash, action) pairs
///   2. FromPatterns: extract fossilized patterns above threshold
///   3. FromTrajectory: mine trajectory DAG for high-value state→action mappings
///
/// This is domain-agnostic. The fossilized skill is a simple string→string map.
/// Domain implementations interpret the keys and values:
///   - Chess: Zobrist hash → best move/evaluation
///   - Robotics: pose hash → motor command
///   - LLM: query hash → cached answer
///   - Trading: regime hash → trade decision
/// </summary>
public static class Fossilizer
{
    /// <summary>
    /// Create a FossilizedSkill from raw episodes (situation, action) pairs.
    /// </summary>
    public static FossilizedSkill FromEpisodes(
        string name,
        IEnumerable<(string SituationHash, string Action)> episodes)
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var (situation, action) in episodes)
        {
            map[situation] = action;
        }

        return new FossilizedSkill(name, map);
    }

    /// <summary>
    /// Create a FossilizedSkill from patterns that have crossed the fossilization threshold.
    /// Only includes patterns where IsFossilized = true.
    /// </summary>
    public static FossilizedSkill FromPatterns(string name, IEnumerable<Pattern> patterns)
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var p in patterns.Where(p => p.IsFossilized))
        {
            map[p.StateSignature] = p.Action;
        }

        return new FossilizedSkill(name, map);
    }

    /// <summary>
    /// Create a FossilizedSkill from high-value trajectory nodes.
    /// Extracts (SituationHash → Action) pairs from nodes above the Q-value threshold.
    /// </summary>
    public static FossilizedSkill FromTrajectory(
        string name,
        IEnumerable<TrajectoryNode> nodes,
        double minQValue = 0.5)
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var n in nodes.Where(n => n.QValue >= minQValue))
        {
            // If multiple entries for the same situation, keep the one with higher Q-value
            if (!map.ContainsKey(n.SituationHash) || n.QValue > minQValue)
            {
                map[n.SituationHash] = n.Action;
            }
        }

        return new FossilizedSkill(name, map);
    }
}
