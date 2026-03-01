using System.Security.Cryptography;
using RichLearning.Abstractions;
using RichLearning.Models;

namespace RichLearning.Planning;

/// <summary>
/// Mid-level planner that maintains and queries the topological state-space map.
///
/// Provides domain-agnostic graph management: novelty gating, transition recording,
/// loop detection, frontier exploration, and shortest-path planning.
///
/// <remarks>This is the public API stub. Full implementation is provided by
/// the rich-learning-base package.</remarks>
/// </summary>
public class Cartographer
{
    private readonly IGraphMemory _memory;
    private readonly IStateEncoder _encoder;
    private readonly INoveltyGate _noveltyGate;
    private readonly ILoopEscapeStrategy? _loopEscape;

    /// <summary>Novelty threshold for the default gate if none provided.</summary>
    public static double DefaultNoveltyThreshold { get; set; } = 0.3;

    public Cartographer(
        IGraphMemory memory,
        IStateEncoder encoder,
        INoveltyGate? noveltyGate = null,
        ILoopEscapeStrategy? loopEscape = null,
        double discountFactor = 0.95)
    {
        _memory = memory;
        _encoder = encoder;
        _noveltyGate = noveltyGate ?? new DefaultNoveltyGate(DefaultNoveltyThreshold);
        _loopEscape = loopEscape;
    }

    /// <summary>Process a new state observation. Returns the landmark ID.</summary>
    /// <remarks>Full implementation in rich-learning-base.</remarks>
    public Task<string> ObserveStateAsync(double[] rawState)
        => throw new NotImplementedException("See rich-learning-base for full implementation.");

    /// <summary>Record a transition after the agent executes an action.</summary>
    /// <remarks>Full implementation in rich-learning-base.</remarks>
    public Task RecordTransitionAsync(
        string fromId, string toId, int action, double reward, bool success = true)
        => throw new NotImplementedException("See rich-learning-base for full implementation.");

    /// <summary>Detect and escape trajectory loops.</summary>
    /// <remarks>Full implementation in rich-learning-base.</remarks>
    public Task<SubgoalDirective?> DetectAndBreakLoopAsync()
        => throw new NotImplementedException("See rich-learning-base for full implementation.");

    /// <summary>Select the next subgoal for exploration.</summary>
    /// <remarks>Full implementation in rich-learning-base.</remarks>
    public Task<SubgoalDirective?> SelectNextSubgoalAsync()
        => throw new NotImplementedException("See rich-learning-base for full implementation.");

    /// <summary>Plan a shortest path between two landmarks.</summary>
    public Task<IReadOnlyList<string>> PlanPathAsync(string fromId, string toId) =>
        _memory.ShortestPathAsync(fromId, toId);

    /// <summary>Get prioritised replay batch for training.</summary>
    /// <remarks>Full implementation in rich-learning-base.</remarks>
    public Task<IReadOnlyList<StateTransition>> GetReplayBatchAsync(int batchSize = 32)
        => throw new NotImplementedException("See rich-learning-base for full implementation.");

    /// <summary>Get a summary of the current map state.</summary>
    /// <remarks>Full implementation in rich-learning-base.</remarks>
    public Task<MapSnapshot> GetMapSummaryAsync()
        => throw new NotImplementedException("See rich-learning-base for full implementation.");

    /// <summary>Clear trajectory history.</summary>
    public void ResetTrajectory() { }

    /// <summary>Current trajectory length.</summary>
    public int TrajectoryLength => 0;

    /// <summary>Current timestep.</summary>
    public long Timestep => 0;

    /// <summary>SHA-256 hash of embedding vector, truncated to 16 hex chars.</summary>
    public static string ComputeStateHash(double[] embedding)
    {
        var bytes = new byte[embedding.Length * sizeof(double)];
        Buffer.BlockCopy(embedding, 0, bytes, 0, bytes.Length);
        var hash = SHA256.HashData(bytes);
        return Convert.ToHexString(hash)[..16];
    }

    /// <summary>Default novelty gate with fixed threshold.</summary>
    private sealed class DefaultNoveltyGate(double threshold) : INoveltyGate
    {
        public bool ShouldCreateLandmark(double distanceToNearest) =>
            distanceToNearest > threshold;
    }
}
