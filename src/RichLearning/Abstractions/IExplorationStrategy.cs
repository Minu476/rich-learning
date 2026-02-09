using RichLearning.Models;

namespace RichLearning.Abstractions;

/// <summary>
/// Scores frontier landmarks for exploration priority.
/// Allows customisation of the exploration/exploitation strategy.
/// </summary>
public interface IFrontierScorer
{
    /// <summary>
    /// Compute a priority score for a landmark.
    /// Higher score = higher exploration priority.
    /// </summary>
    double Score(double noveltyScore, int visitCount, int outDegree);
}

/// <summary>
/// Decides whether a new observation is novel enough to create a new landmark.
/// </summary>
public interface INoveltyGate
{
    /// <summary>
    /// Returns true if a new landmark should be created (observation is sufficiently novel).
    /// </summary>
    bool ShouldCreateLandmark(double distanceToNearest);
}

/// <summary>
/// Computes priority for experience replay sampling.
/// Graph-prioritised replay uses graph structure (not just TD-error) to select transitions.
/// </summary>
public interface IPrioritySampler
{
    /// <summary>
    /// Compute replay priority for a transition.
    /// Higher = more likely to be sampled.
    /// </summary>
    double ComputePriority(double tdError, int transitionCount, long staleness);
}

/// <summary>
/// Strategy for escaping detected trajectory loops.
/// </summary>
public interface ILoopEscapeStrategy
{
    /// <summary>
    /// Given cycle node IDs and available frontiers, select an escape target.
    /// Returns null if no escape is possible.
    /// </summary>
    Task<SubgoalDirective?> SelectEscapeTargetAsync(
        IReadOnlyList<string> cycleNodeIds,
        IReadOnlyList<StateLandmark> frontiers,
        IGraphMemory memory,
        string currentLandmarkId);
}
