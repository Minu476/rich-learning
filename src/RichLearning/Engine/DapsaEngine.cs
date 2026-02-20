using System.Collections.Concurrent;
using RichLearning.Abstractions;
using RichLearning.Learning;
using RichLearning.Memory;
using RichLearning.Models;

namespace RichLearning.Engine;

/// <summary>
/// The unified Rich Learning DAPSA engine.
///
/// This is the central orchestrator that implements the complete DAPSA lifecycle:
///
///   1. PERCEIVE: Encode raw observation → embedding
///   2. QUERY PASSIVE: O(1) lookup in fossilized knowledge (System 1)
///   3. CHECK CONSONANCE: If passive hit, is it still valid? (Wake-on-Surprise)
///   4. QUERY ACTIVE: Deep evaluation if passive missed or dissonant (System 2)
///   5. LEARN: Record trajectory, backward reinforce, update patterns
///   6. FOSSILIZE: Promote high-confidence patterns to passive (Phi: S^A → S^P)
///
/// Domain implementations derive from this class and:
///   - Provide domain-specific IStateEncoder
///   - Provide domain-specific IConsonanceChecker
///   - Override EvaluateActive() for domain-specific System 2 computation
///   - Configure fossilization thresholds and discount factors
///
/// This engine is the "source of truth" base class. All domain projects
/// (chess, LLM, trading, robotics, etc.) should derive from it.
/// </summary>
public class DapsaEngine<TObservation, TAction>
{
    private readonly IGraphMemory _graphMemory;
    private readonly IStateEncoder _encoder;
    private readonly IConsonanceChecker<TObservation, TAction>? _consonanceChecker;
    private readonly IActionEncoder<TAction> _actionEncoder;
    private readonly Planning.Cartographer _cartographer;
    private readonly MetaHierarchy _metaHierarchy;
    private readonly AgentEfficiencyTracker _efficiencyTracker;

    private readonly ConcurrentDictionary<string, TrajectoryDag> _trajectories = new();
    private readonly ConcurrentDictionary<string, string?> _lastTrajectoryNodeIds = new();
    private long _sessionCount;
    private long _passiveHits;
    private long _activeMisses;

    /// <summary>Temporal discount factor γ for backward reinforcement.</summary>
    public double DiscountFactor { get; init; } = 0.95;

    /// <summary>Minimum Q-value for fossilization from trajectory.</summary>
    public double FossilizationQThreshold { get; init; } = 0.5;

    /// <summary>All fossilized skills accumulated over the engine's lifetime.</summary>
    private readonly ConcurrentDictionary<string, FossilizedSkill> _fossilVault = new(StringComparer.Ordinal);

    public DapsaEngine(
        IGraphMemory graphMemory,
        IStateEncoder encoder,
        IConsonanceChecker<TObservation, TAction>? consonanceChecker = null,
        IActionEncoder<TAction>? actionEncoder = null,
        INoveltyGate? noveltyGate = null,
        ILoopEscapeStrategy? loopEscape = null)
    {
        _graphMemory = graphMemory;
        _encoder = encoder;
        _consonanceChecker = consonanceChecker;
        _actionEncoder = actionEncoder ?? new DefaultActionEncoder<TAction>();
        _cartographer = new Planning.Cartographer(graphMemory, encoder, noveltyGate, loopEscape);
        _metaHierarchy = new MetaHierarchy();
        _efficiencyTracker = new AgentEfficiencyTracker();
    }

    /// <summary>
    /// The complete DAPSA query cycle.
    ///
    /// 1. Check Passive Manifold (fossilized knowledge)
    /// 2. If miss or dissonant, fall through to Active evaluation
    /// 3. Record in trajectory for learning
    ///
    /// Returns a DapsaResult containing the evaluation, source, and confidence.
    /// </summary>
    public async Task<DapsaResult<TAction>> QueryAsync(
        TObservation observation,
        double[] rawState,
        Func<TObservation, Task<(TAction Action, double QValue, int Depth)>> activeEvaluator,
        string sessionId = "default")
    {
        Interlocked.Increment(ref _sessionCount);

        // Step 1: Encode state
        var embedding = _encoder.Encode(rawState);
        var stateHash = Planning.Cartographer.ComputeStateHash(rawState);

        // Step 2: Check fossilized knowledge (Passive Manifold / System 1)
        foreach (var skill in _fossilVault.Values)
        {
            if (skill.TryLookup(stateHash, out var fossilActionStr) && fossilActionStr is not null)
            {
                var fossilAction = _actionEncoder.Decode(fossilActionStr);

                // Step 3: Consonance check — is the fossil still valid?
                if (_consonanceChecker is not null)
                {
                    var consonance = _consonanceChecker.Check(observation, fossilAction);
                    if (consonance.IsDissonant)
                    {
                        // Surprise! The passive prediction is stale.
                        // Fall through to active evaluation.
                        break;
                    }
                }

                Interlocked.Increment(ref _passiveHits);
                return new DapsaResult<TAction>
                {
                    Source = DapsaSource.Passive,
                    Action = fossilAction,
                    RawAction = fossilActionStr,
                    Confidence = 1.0,
                    QValue = 0,
                };
            }
        }

        // Step 4: Active evaluation (System 2 — expensive)
        Interlocked.Increment(ref _activeMisses);
        var (action, qValue, depth) = await activeEvaluator(observation);

        // Step 5: Record in trajectory
        var landmarkId = await _cartographer.ObserveStateAsync(rawState);
        var trajectory = _trajectories.GetOrAdd(sessionId, _ => new TrajectoryDag());
        _lastTrajectoryNodeIds.TryGetValue(sessionId, out var lastNodeId);

        var (trajNode, isLoop) = trajectory.Append(
            stateHash, _actionEncoder.Encode(action), depth, qValue, lastNodeId);
        _lastTrajectoryNodeIds[sessionId] = trajNode.Id;

        // Step 6: Record pattern for meta-hierarchy
        var pattern = new Pattern(stateHash, _actionEncoder.Encode(action), "engine");
        pattern.Observe(success: qValue > 0);
        _metaHierarchy.Observe(pattern);

        return new DapsaResult<TAction>
        {
            Source = DapsaSource.Active,
            Action = action,
            RawAction = _actionEncoder.Encode(action),
            Confidence = 1.0 / (1 + Math.Exp(-qValue)), // sigmoid confidence
            QValue = qValue,
            Depth = depth,
            IsLoop = isLoop,
            LandmarkId = landmarkId,
        };
    }

    /// <summary>
    /// Signal the end of an episode/game/session.
    /// Triggers backward reinforcement and fossilization check.
    /// </summary>
    public async Task EndEpisodeAsync(double terminalReward, string agentId = "default", string sessionId = "default")
    {
        if (!_trajectories.TryGetValue(sessionId, out var trajectory)) return;

        // Backward reinforcement through the trajectory
        var leaves = trajectory.GetLeafNodes();
        foreach (var leaf in leaves)
        {
            trajectory.BackwardReinforce(leaf.Id, terminalReward, DiscountFactor);
        }

        // Teach the meta-hierarchy
        var highValueNodes = trajectory.GetHighValueNodes(FossilizationQThreshold);
        if (highValueNodes.Count > 0)
        {
            var patternIds = highValueNodes.Select(n =>
                $"P_{n.SituationHash[..Math.Min(8, n.SituationHash.Length)]}_{n.Action[..Math.Min(4, n.Action.Length)]}").ToList();
            _metaHierarchy.TeachBackward(terminalReward, patternIds);
        }

        // Track agent efficiency
        _efficiencyTracker.RecordPerformance(agentId, terminalReward);

        // Attempt fossilization from trajectory
        if (highValueNodes.Count > 0)
        {
            var skill = Fossilizer.FromTrajectory(
                $"session_{Interlocked.Read(ref _sessionCount)}", highValueNodes, FossilizationQThreshold);

            if (skill.Size > 0)
            {
                _fossilVault[skill.SkillName] = skill;
            }
        }

        // Reset trajectory for next session
        trajectory.Clear();
        _lastTrajectoryNodeIds.TryRemove(sessionId, out _);
    }

    /// <summary>Start a new episode/game/session.</summary>
    public void NewSession(string sessionId = "default")
    {
        if (_trajectories.TryGetValue(sessionId, out var trajectory))
        {
            trajectory.Clear();
        }
        _lastTrajectoryNodeIds.TryRemove(sessionId, out _);
    }

    // ── Diagnostics ──

    /// <summary>Get the passive hit rate (% of queries served by fossils).</summary>
    public double PassiveHitRate
    {
        get
        {
            long hits = Interlocked.Read(ref _passiveHits);
            long misses = Interlocked.Read(ref _activeMisses);
            return (hits + misses) > 0 ? (double)hits / (hits + misses) : 0.0;
        }
    }

    /// <summary>Total fossilized knowledge entries across all skills.</summary>
    public int FossilCount => _fossilVault.Values.Sum(s => s.Size);

    /// <summary>Number of hierarchy levels discovered.</summary>
    public int HierarchyLevels => _metaHierarchy.LevelCount;

    /// <summary>Total patterns tracked in the meta-hierarchy.</summary>
    public int PatternCount => _metaHierarchy.TotalPatternCount;

    /// <summary>Get the underlying cartographer for direct graph operations.</summary>
    public Planning.Cartographer Cartographer => _cartographer;

    /// <summary>Get the meta-hierarchy for inspection.</summary>
    public MetaHierarchy MetaHierarchy => _metaHierarchy;

    /// <summary>Get the efficiency tracker for multi-agent setups.</summary>
    public AgentEfficiencyTracker EfficiencyTracker => _efficiencyTracker;

    /// <summary>Get a summary of the topological map.</summary>
    public Task<MapSnapshot> GetMapSummaryAsync() => _cartographer.GetMapSummaryAsync();
}

/// <summary>
/// Result of a DAPSA query cycle.
/// </summary>
public sealed record DapsaResult<TAction>
{
    /// <summary>Whether the result came from Passive (fossil) or Active (search).</summary>
    public required DapsaSource Source { get; init; }

    /// <summary>The domain-specific action.</summary>
    public required TAction Action { get; init; }

    /// <summary>Raw string representation of the action (for fossilization).</summary>
    public string? RawAction { get; init; }

    /// <summary>Confidence in the result [0, 1].</summary>
    public double Confidence { get; init; }

    /// <summary>Q-value from evaluation.</summary>
    public double QValue { get; init; }

    /// <summary>Search depth (for active evaluations).</summary>
    public int Depth { get; init; }

    /// <summary>Whether a loop was detected at this state.</summary>
    public bool IsLoop { get; init; }

    /// <summary>The landmark ID in the topological graph.</summary>
    public string? LandmarkId { get; init; }
}

/// <summary>
/// Source of a DAPSA evaluation result.
/// </summary>
public enum DapsaSource
{
    /// <summary>Result from Passive Manifold (System 1, fossil hit).</summary>
    Passive,

    /// <summary>Result from Active Manifold (System 2, deep evaluation).</summary>
    Active,

    /// <summary>Not found in either manifold.</summary>
    NotFound
}
