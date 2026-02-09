using System.Security.Cryptography;
using System.Text;
using Microsoft.Extensions.Logging;
using RichLearning.Abstractions;
using RichLearning.Models;

namespace RichLearning.Planning;

/// <summary>
/// Mid-level planner that maintains and queries the topological state-space map.
/// 
/// Responsibilities:
///   - Decides when to create new landmark nodes (novelty gating)
///   - Updates node/edge statistics after each transition
///   - Plans shortest-path routes between landmarks
///   - Detects and breaks trajectory loops
///   - Identifies exploration frontiers
/// </summary>
public sealed class Cartographer
{
    private readonly IGraphMemory _memory;
    private readonly IStateEncoder _encoder;
    private readonly ILogger<Cartographer> _logger;

    /// <summary>Minimum distance to nearest landmark to create a new one.</summary>
    public double NoveltyThreshold { get; set; } = 0.3;

    /// <summary>EMA factor for value estimate updates.</summary>
    public double ValueEmaAlpha { get; set; } = 0.1;

    /// <summary>Novelty decay rate per visit.</summary>
    public double NoveltyDecayRate { get; set; } = 0.05;

    /// <summary>Number of recent landmark IDs for loop detection.</summary>
    public int TrajectoryWindowSize { get; set; } = 20;

    private readonly LinkedList<string> _trajectoryWindow = new();
    private long _timestep;

    public Cartographer(
        IGraphMemory memory,
        IStateEncoder encoder,
        ILogger<Cartographer> logger)
    {
        _memory = memory;
        _encoder = encoder;
        _logger = logger;
    }

    // ── Observe a State ──

    /// <summary>
    /// Process a new state observation. Creates a new landmark if novel enough,
    /// or updates the nearest existing one. Returns the landmark ID.
    /// </summary>
    public async Task<string> ObserveStateAsync(double[] rawState, double reward = 0.0)
    {
        _timestep++;
        var embedding = _encoder.Encode(rawState);
        var nearest = await _memory.NearestNeighbourAsync(embedding, _encoder);

        string landmarkId;

        if (nearest is null || nearest.Value.Distance > NoveltyThreshold)
        {
            // Novel enough → create a new landmark.
            landmarkId = ComputeStateHash(embedding);
            var landmark = new StateLandmark
            {
                Id = landmarkId,
                Embedding = embedding,
                VisitCount = 1,
                ValueEstimate = reward,
                NoveltyScore = 1.0,
                ClusterId = 0,
                LastVisitedTimestep = _timestep,
                CreatedTimestep = _timestep
            };
            await _memory.UpsertLandmarkAsync(landmark);
            _logger.LogInformation("Created landmark {Id} (dist={Dist:F3})",
                landmarkId, nearest?.Distance ?? double.NaN);
        }
        else
        {
            // Close to existing → update.
            var existing = nearest.Value.Landmark;
            landmarkId = existing.Id;
            existing.VisitCount++;
            existing.LastVisitedTimestep = _timestep;
            existing.NoveltyScore *= (1.0 - NoveltyDecayRate);
            existing.ValueEstimate += ValueEmaAlpha * (reward - existing.ValueEstimate);
            await _memory.UpsertLandmarkAsync(existing);
        }

        _trajectoryWindow.AddLast(landmarkId);
        while (_trajectoryWindow.Count > TrajectoryWindowSize)
            _trajectoryWindow.RemoveFirst();

        return landmarkId;
    }

    // ── Record a Transition ──

    /// <summary>
    /// Record a transition after the worker executes an action.
    /// </summary>
    public async Task RecordTransitionAsync(
        string fromId, string toId,
        int action, double reward, int primitiveSteps, bool success)
    {
        var existingEdges = await _memory.GetOutgoingTransitionsAsync(fromId);
        var existing = existingEdges.FirstOrDefault(e => e.TargetId == toId && e.Action == action);

        StateTransition transition;
        if (existing is not null)
        {
            int newCount = existing.TransitionCount + 1;
            transition = existing with
            {
                Reward = existing.Reward + (reward - existing.Reward) / newCount,
                TransitionCount = newCount,
                SuccessRate = existing.SuccessRate + (Convert.ToDouble(success) - existing.SuccessRate) / newCount,
                TemporalDistance = (existing.TemporalDistance + primitiveSteps) / 2,
                LastTrainedTimestep = _timestep
            };
        }
        else
        {
            transition = new StateTransition
            {
                SourceId = fromId,
                TargetId = toId,
                Action = action,
                Reward = reward,
                TransitionCount = 1,
                SuccessRate = success ? 1.0 : 0.0,
                TemporalDistance = primitiveSteps,
                TdError = Math.Abs(reward),
                LastTrainedTimestep = _timestep
            };
        }

        await _memory.UpsertTransitionAsync(transition);
    }

    // ── Loop Detection ──

    /// <summary>
    /// Check if the trajectory contains a loop; if so, redirect to a frontier.
    /// </summary>
    public async Task<SubgoalDirective?> DetectAndBreakLoopAsync()
    {
        var recentIds = _trajectoryWindow.ToList();
        if (recentIds.Count < 4) return null;

        var cycleNodes = await _memory.DetectCycleInTrajectoryAsync(recentIds);
        if (cycleNodes.Count == 0) return null;

        _logger.LogWarning("Loop detected: {Count} nodes", cycleNodes.Count);

        var frontiers = await _memory.GetFrontierLandmarksAsync(5);
        var escape = frontiers.FirstOrDefault(f => !cycleNodes.Contains(f.Id));
        if (escape is null) return null;

        var currentId = recentIds[^1];
        var path = await _memory.ShortestPathAsync(currentId, escape.Id);

        return new SubgoalDirective
        {
            TargetLandmarkId = escape.Id,
            Reason = $"Loop escape: cycle of {cycleNodes.Count} nodes",
            PlannedPath = path
        };
    }

    // ── Exploration: Select Next Subgoal ──

    /// <summary>
    /// Select the next subgoal. Priority: (1) escape loops, (2) target frontiers.
    /// </summary>
    public async Task<SubgoalDirective?> SelectNextSubgoalAsync(string currentLandmarkId)
    {
        var loopEscape = await DetectAndBreakLoopAsync();
        if (loopEscape is not null) return loopEscape;

        var frontiers = await _memory.GetFrontierLandmarksAsync(1);
        if (frontiers.Count == 0) return null;

        var target = frontiers[0];
        var path = await _memory.ShortestPathAsync(currentLandmarkId, target.Id);

        return new SubgoalDirective
        {
            TargetLandmarkId = target.Id,
            Reason = $"Frontier (novelty={target.NoveltyScore:F3}, visits={target.VisitCount})",
            PlannedPath = path
        };
    }

    // ── Experience Replay ──

    public async Task<IReadOnlyList<StateTransition>> GetReplayBatchAsync(int batchSize = 32)
        => await _memory.PrioritisedSampleAsync(batchSize, _timestep);

    // ── Diagnostics ──

    public async Task<string> GetMapSummaryAsync()
    {
        var (landmarks, transitions) = await _memory.GetGraphStatsAsync();
        var frontiers = await _memory.GetFrontierLandmarksAsync(3);

        var sb = new StringBuilder();
        sb.AppendLine($"=== Topological Map (t={_timestep}) ===");
        sb.AppendLine($"  Landmarks  : {landmarks}");
        sb.AppendLine($"  Transitions: {transitions}");
        sb.AppendLine($"  Trajectory : {_trajectoryWindow.Count}/{TrajectoryWindowSize}");
        foreach (var f in frontiers)
            sb.AppendLine($"  Frontier: {f.Id} (novelty={f.NoveltyScore:F3}, visits={f.VisitCount})");
        return sb.ToString();
    }

    // ── Helpers ──

    private static string ComputeStateHash(double[] embedding)
    {
        var bytes = new byte[embedding.Length * sizeof(double)];
        Buffer.BlockCopy(embedding, 0, bytes, 0, bytes.Length);
        var hash = SHA256.HashData(bytes);
        return Convert.ToHexString(hash)[..16];
    }
}
