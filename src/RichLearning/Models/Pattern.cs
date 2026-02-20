namespace RichLearning.Models;

/// <summary>
/// A learned pattern: an observed (situation, action) pair with confidence tracking.
///
/// Patterns are the atomic units of knowledge in the learning pipeline.
/// They track how often a specific action was taken in a specific state,
/// and whether it succeeded.
///
/// From the canonical spec:
///   confidence = times_succeeded / times_seen
///   fossilized = (times_succeeded >= fossil_threshold) AND (confidence > 0.8)
///
/// Backward reinforcement (teaching):
///   times_succeeded += max(0, int(reward * teaching_weight))
///
/// This is domain-agnostic: any (state_signature, action) pair works.
/// </summary>
public sealed class Pattern
{
    public string Id { get; }
    public string StateSignature { get; }
    public string Action { get; }
    public string Agent { get; }
    public int TimesSeen { get; private set; }
    public int TimesSucceeded { get; private set; }
    public float Confidence { get; private set; }
    public bool IsFossilized { get; private set; }
    public float TeachingWeight { get; set; } = 1.0f;

    /// <summary>Minimum successes before a pattern can fossilize.</summary>
    public int FossilThreshold { get; init; } = 50;

    /// <summary>Minimum confidence for fossilization.</summary>
    public float FossilConfidence { get; init; } = 0.8f;

    public Pattern(string stateSignature, string action, string agent)
    {
        StateSignature = stateSignature;
        Action = action;
        Agent = agent;
        Id = $"P_{stateSignature[..Math.Min(8, stateSignature.Length)]}_{action[..Math.Min(4, action.Length)]}";
    }

    /// <summary>
    /// Record an observation of this pattern.
    /// Updates confidence as success_rate = times_succeeded / times_seen.
    /// Checks fossilization threshold.
    /// </summary>
    public void Observe(bool success)
    {
        TimesSeen++;
        if (success)
            TimesSucceeded++;

        Confidence = TimesSeen > 0 ? (float)TimesSucceeded / TimesSeen : 0f;

        if (TimesSucceeded >= FossilThreshold && Confidence >= FossilConfidence)
            IsFossilized = true;
    }

    /// <summary>
    /// Backward reinforcement: boost this pattern's success count.
    ///
    /// From backward.py: pattern.Reinforce(reward * teaching_weight)
    ///   times_succeeded += max(0, int(reward * teaching_weight))
    ///   times_seen += boost  (maintain ratio stability)
    /// </summary>
    public void Reinforce(double reward)
    {
        int boost = Math.Max(0, (int)(reward * TeachingWeight));
        TimesSucceeded += boost;
        TimesSeen += boost;
        Confidence = TimesSeen > 0 ? (float)TimesSucceeded / TimesSeen : 0f;

        if (TimesSucceeded >= FossilThreshold && Confidence >= FossilConfidence)
            IsFossilized = true;
    }

    public float SuccessRate => TimesSeen > 0 ? (float)TimesSucceeded / TimesSeen : 0f;
}
