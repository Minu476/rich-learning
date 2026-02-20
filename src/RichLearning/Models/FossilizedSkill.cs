namespace RichLearning.Models;

/// <summary>
/// A fossilized skill: an O(1) lookup table mapping situation hashes to actions.
///
/// This is the Passive Manifold's unit of knowledge (Patent 1: Fossilization).
/// Once a pattern has been observed enough times with sufficient success rate,
/// it "fossilizes" into this immutable lookup table.
///
/// The fossilized skill enables System 1 (cheap, habitual) execution.
/// The Agent retrieves the action in O(1) without any computation.
///
/// From DAPSA: Phi: S^A → S^P (Active → Passive promotion).
/// </summary>
public sealed class FossilizedSkill
{
    public string SkillName { get; }
    public IReadOnlyDictionary<string, string> SituationActionMap { get; }
    public DateTimeOffset CreationTime { get; }
    public double SuccessRate { get; }
    public long UsageCount { get; private set; }

    public FossilizedSkill(
        string skillName,
        IDictionary<string, string> situationActionMap,
        DateTimeOffset? creationTime = null,
        double successRate = 0.0,
        long usageCount = 0)
    {
        if (string.IsNullOrWhiteSpace(skillName))
            throw new ArgumentException("Skill name is required", nameof(skillName));

        SkillName = skillName;
        SituationActionMap = new Dictionary<string, string>(
            situationActionMap, StringComparer.Ordinal);
        CreationTime = creationTime ?? DateTimeOffset.UtcNow;
        SuccessRate = successRate;
        UsageCount = usageCount;
    }

    /// <summary>
    /// O(1) lookup: given a situation hash, return the fossilized action if known.
    /// This is the hot path for System 1 (Passive Mode).
    /// </summary>
    public bool TryLookup(string situationHash, out string? action)
    {
        UsageCount++;
        return SituationActionMap.TryGetValue(situationHash, out action);
    }

    public int Size => SituationActionMap.Count;
}
