using RichLearning.Abstractions;
using RichLearning.Learning;
using RichLearning.Memory;
using RichLearning.Models;

namespace RichLearning.Engine;

/// <summary>
/// The unified Rich Learning DAPSA engine.
///
/// Orchestrates the DAPSA lifecycle: Perceive → Query Passive → Check Consonance
/// → Query Active → Learn → Fossilize.
///
/// <remarks>
/// This is the public API stub. Full implementation is provided by
/// the rich-learning-base package.
/// </remarks>
/// </summary>
public class DapsaEngine<TObservation, TAction>
{
    private readonly IGraphMemory _graphMemory;
    private readonly IStateEncoder _encoder;
    private readonly IConsonanceChecker<TObservation>? _consonanceChecker;
    private readonly Planning.Cartographer _cartographer;
    private readonly MetaHierarchy _metaHierarchy;
    private readonly AgentEfficiencyTracker _efficiencyTracker;

    /// <summary>Temporal discount factor γ for backward reinforcement.</summary>
    public double DiscountFactor { get; init; } = 0.95;

    /// <summary>Minimum Q-value for fossilization from trajectory.</summary>
    public double FossilizationQThreshold { get; init; } = 0.5;

    /// <summary>
    /// Maximum cosine distance for the semantic radius fossil lookup.
    /// Landmarks within this radius are treated as "close enough" for a passive recall.
    /// 0.15 ≈ 82° cosine angle — catches similar paraphrases without false positives.
    /// </summary>
    public double FossilLookupRadius { get; init; } = 0.15;

    public DapsaEngine(
        IGraphMemory graphMemory,
        IStateEncoder encoder,
        IConsonanceChecker<TObservation>? consonanceChecker = null,
        INoveltyGate? noveltyGate = null,
        ILoopEscapeStrategy? loopEscape = null,
        double discountFactor = 0.95)
    {
        DiscountFactor = discountFactor;
        _graphMemory = graphMemory;
        _encoder = encoder;
        _consonanceChecker = consonanceChecker;
        _cartographer = new Planning.Cartographer(graphMemory, encoder, noveltyGate, loopEscape, discountFactor);
        _metaHierarchy = new MetaHierarchy();
        _efficiencyTracker = new AgentEfficiencyTracker();
    }

    /// <summary>
    /// Execute the complete DAPSA query cycle for a single observation.
    /// </summary>
    /// <remarks>Full implementation in rich-learning-base.</remarks>
    public Task<DapsaResult<TAction>> QueryAsync(
        TObservation observation,
        double[] rawState,
        Func<TObservation, Task<(TAction Action, double QValue, int Depth)>> activeEvaluator)
        => throw new NotImplementedException("See rich-learning-base for full DAPSA lifecycle implementation.");

    /// <summary>
    /// Signal the end of an episode. Triggers backward reinforcement and fossilization.
    /// </summary>
    /// <remarks>Full implementation in rich-learning-base.</remarks>
    public Task EndEpisodeAsync(double terminalReward, string agentId = "default")
        => throw new NotImplementedException("See rich-learning-base for full DAPSA lifecycle implementation.");

    /// <summary>Start a new episode/game/session.</summary>
    /// <remarks>Full implementation in rich-learning-base.</remarks>
    public void NewSession()
        => throw new NotImplementedException("See rich-learning-base for full DAPSA lifecycle implementation.");

    // ── Diagnostics ──

    /// <summary>Passive hit rate (% of queries served by fossils).</summary>
    public double PassiveHitRate => 0.0;

    /// <summary>Total fossilized knowledge entries across all skills.</summary>
    public int FossilCount => 0;

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

    /// <summary>
    /// Triggers offline graph coarsening. Runs community detection (label propagation)
    /// and builds macro-landmarks and macro-edges for faster high-level planning.
    /// </summary>
    /// <remarks>Full implementation in rich-learning-base.</remarks>
    public async Task<Planning.MetaLevelReport> SynthesizeMetaGraphAsync()
    {
        var builder = new Planning.MetaLevelBuilder(_graphMemory);
        return await builder.BuildAllLevelsAsync();
    }
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
