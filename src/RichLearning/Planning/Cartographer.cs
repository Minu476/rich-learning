using System.Security.Cryptography;
using System.Text;
using RichLearning.Abstractions;
using RichLearning.Models;

namespace RichLearning.Planning;

/// <summary>
/// Mid-level planner that maintains and queries the topological state-space map.
///
/// The Cartographer is the central component of Rich Learning. It provides
/// domain-agnostic graph management operations:
///
///   - ObserveState: novelty gating → create new landmark or update existing
///   - RecordTransition: record (state, action, reward, next_state) edge
///   - DetectAndBreakLoop: cycle detection with escape-to-frontier
///   - SelectNextSubgoal: exploration strategy (loops + frontiers)
///   - PlanPath: shortest-path planning to a target landmark
///   - GetMapSummary: diagnostic reporting
///
/// Domain implementations extend Cartographer by injecting domain-specific:
///   - IStateEncoder: how to transform observations into embeddings
///   - INoveltyGate: what counts as "novel enough" to create a new landmark
///   - ILoopEscapeStrategy: how to escape detected loops
///   - IGraphMemory: where to persist the graph (Neo4j, LiteDB, in-memory)
///
/// The Cartographer never makes domain-specific decisions. It IS the "map."
/// The "territory" (domain) is abstracted through the interfaces.
/// </summary>
public class Cartographer
{
    private readonly IGraphMemory _memory;
    private readonly IStateEncoder _encoder;
    private readonly INoveltyGate _noveltyGate;
    private readonly ILoopEscapeStrategy? _loopEscape;
    private readonly List<string> _trajectory = new();
    private long _timestep;

    /// <summary>Novelty threshold for the default gate if none provided.</summary>
    public static double DefaultNoveltyThreshold { get; set; } = 0.3;

    public Cartographer(
        IGraphMemory memory,
        IStateEncoder encoder,
        INoveltyGate? noveltyGate = null,
        ILoopEscapeStrategy? loopEscape = null)
    {
        _memory = memory;
        _encoder = encoder;
        _noveltyGate = noveltyGate ?? new DefaultNoveltyGate(DefaultNoveltyThreshold);
        _loopEscape = loopEscape;
    }

    /// <summary>
    /// Process a new state observation. Creates a new landmark if novel enough,
    /// or updates the nearest existing one. Returns the landmark ID.
    /// </summary>
    public async Task<string> ObserveStateAsync(double[] rawState)
    {
        _timestep++;
        var embedding = _encoder.Encode(rawState);
        var nearest = await _memory.NearestNeighbourAsync(embedding, _encoder);

        if (nearest is null || _noveltyGate.ShouldCreateLandmark(nearest.Value.Distance))
        {
            // Novel state — create new landmark
            var id = ComputeStateHash(embedding);
            var landmark = new StateLandmark
            {
                Id = id,
                Embedding = embedding,
                VisitCount = 1,
                NoveltyScore = 1.0,
                UncertaintyScore = 1.0,
                HierarchyLevel = 0,
                CreatedTimestep = _timestep,
                LastVisitedTimestep = _timestep,
            };
            await _memory.UpsertLandmarkAsync(landmark);
            _trajectory.Add(id);
            return id;
        }
        else
        {
            // Known state — update existing landmark
            var lm = nearest.Value.Landmark;
            lm.VisitCount++;
            lm.LastVisitedTimestep = _timestep;
            lm.NoveltyScore = 1.0 / (1 + Math.Log(1 + lm.VisitCount));
            lm.UncertaintyScore = 1.0 / Math.Sqrt(1 + lm.VisitCount);
            await _memory.UpsertLandmarkAsync(lm);
            _trajectory.Add(lm.Id);
            return lm.Id;
        }
    }

    /// <summary>
    /// Record a transition after the agent executes an action.
    /// </summary>
    public async Task RecordTransitionAsync(
        string fromId, string toId, int action, double reward, bool success = true)
    {
        var existing = await _memory.GetOutgoingTransitionsAsync(fromId);
        var match = existing.FirstOrDefault(t => t.TargetId == toId && t.Action == action);

        if (match is not null)
        {
            // Update existing transition
            int prevCount = match.TransitionCount;
            match.TransitionCount = prevCount + 1;
            match.Reward = (match.Reward * prevCount + reward) / (prevCount + 1);
            match.SuccessRate = (match.SuccessRate * prevCount + (success ? 1.0 : 0.0)) / (prevCount + 1);
            match.Confidence = 1.0 - 1.0 / Math.Sqrt(1 + match.TransitionCount);
            match.LastTrainedTimestep = _timestep;

            // Update action counts
            match.ActionCounts.TryGetValue(action, out var count);
            match.ActionCounts[action] = count + 1;

            // Update TD-error
            var fromLm = await _memory.GetLandmarkAsync(fromId);
            var toLm = await _memory.GetLandmarkAsync(toId);
            if (fromLm is not null && toLm is not null)
            {
                double gamma = 0.99;
                double target = reward + gamma * toLm.ValueEstimate;
                match.TdError = target - fromLm.ValueEstimate;
            }

            await _memory.UpsertTransitionAsync(match);
        }
        else
        {
            // Create new transition
            var transition = new StateTransition
            {
                SourceId = fromId,
                TargetId = toId,
                Action = action,
                Reward = reward,
                SuccessRate = success ? 1.0 : 0.0,
                Confidence = 0.5,
                TransitionCount = 1,
                LastTrainedTimestep = _timestep,
                ActionCounts = new Dictionary<int, int> { [action] = 1 },
            };
            await _memory.UpsertTransitionAsync(transition);
        }

        // Update source landmark action counts
        var sourceLm = await _memory.GetLandmarkAsync(fromId);
        if (sourceLm is not null)
        {
            sourceLm.ActionCounts.TryGetValue(action, out var ac);
            sourceLm.ActionCounts[action] = ac + 1;
            await _memory.UpsertLandmarkAsync(sourceLm);
        }
    }

    /// <summary>
    /// Check if the trajectory contains a loop; if so, redirect to a frontier.
    /// </summary>
    public async Task<SubgoalDirective?> DetectAndBreakLoopAsync()
    {
        if (_trajectory.Count < 4) return null;

        var recentWindow = _trajectory.TakeLast(10).ToList();
        var cycle = await _memory.DetectCycleInTrajectoryAsync(recentWindow);

        if (cycle.Count == 0) return null;

        var frontiers = await _memory.GetFrontierLandmarksAsync(5);
        if (frontiers.Count == 0) return null;

        if (_loopEscape is not null)
        {
            return await _loopEscape.SelectEscapeTargetAsync(
                cycle, frontiers, _memory, _trajectory[^1]);
        }

        // Default: select the frontier with highest exploration priority
        return new SubgoalDirective
        {
            TargetLandmarkId = frontiers[0].Id,
            Reason = $"Loop escape: cycle of {cycle.Count} nodes detected"
        };
    }

    /// <summary>
    /// Select the next subgoal. Priority: (1) escape loops, (2) target frontiers.
    /// </summary>
    public async Task<SubgoalDirective?> SelectNextSubgoalAsync()
    {
        var loopEscape = await DetectAndBreakLoopAsync();
        if (loopEscape is not null) return loopEscape;

        var frontiers = await _memory.GetFrontierLandmarksAsync(3);
        if (frontiers.Count == 0) return null;

        return new SubgoalDirective
        {
            TargetLandmarkId = frontiers[0].Id,
            Reason = "Frontier exploration"
        };
    }

    /// <summary>Plan a shortest path between two landmarks.</summary>
    public Task<IReadOnlyList<string>> PlanPathAsync(string fromId, string toId) =>
        _memory.ShortestPathAsync(fromId, toId);

    /// <summary>Get prioritised replay batch for training.</summary>
    public Task<IReadOnlyList<StateTransition>> GetReplayBatchAsync(int batchSize = 32) =>
        _memory.PrioritisedSampleAsync(batchSize, _timestep);

    /// <summary>Get a summary of the current map state.</summary>
    public async Task<MapSnapshot> GetMapSummaryAsync()
    {
        var (lmCount, trCount) = await _memory.GetGraphStatsAsync();
        var allLandmarks = await _memory.GetAllLandmarksAsync(hierarchyLevel: 0);

        return new MapSnapshot
        {
            LandmarkCount = lmCount,
            TransitionCount = trCount,
            ClusterCount = allLandmarks.Select(l => l.ClusterId).Distinct().Count(),
            MeanNovelty = allLandmarks.Count > 0 ? allLandmarks.Average(l => l.NoveltyScore) : 0,
            MeanValueEstimate = allLandmarks.Count > 0 ? allLandmarks.Average(l => l.ValueEstimate) : 0,
            MeanPolicyEntropy = allLandmarks.Count > 0 ? allLandmarks.Average(l => l.PolicyEntropy) : 0,
        };
    }

    /// <summary>Clear trajectory history (e.g., for new episode).</summary>
    public void ResetTrajectory() => _trajectory.Clear();

    /// <summary>Current trajectory length.</summary>
    public int TrajectoryLength => _trajectory.Count;

    /// <summary>Current timestep.</summary>
    public long Timestep => _timestep;

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
