namespace RichLearning.Abstractions;

/// <summary>
/// Fractal Actor lifecycle interface (Patent 2: Distributed Fractal Actor).
///
/// Every actor in Rich Learning follows the same lifecycle regardless of domain
/// or hierarchy level. Actors are recursively composable — a Manager spawns
/// Cartographers, which spawn Workers, each at the same interface.
///
/// Lifecycle state machine:
///   Idle → Discovering → Executing → Learning → Idle
///                                              → Fossilized (pattern matched)
///                                              → Error (on failure)
///
/// The fractal property means:
///   - A warehouse AGV actor has the same interface as a fleet supervisor
///   - A chess piece actor has the same interface as a position evaluator
///   - A code method actor has the same interface as a file supervisor
///
/// This enables the Recursive Meta Hierarchy to manage actors at any scale.
/// </summary>
public interface IFractalActor
{
    /// <summary>Unique identifier for this actor instance.</summary>
    string ActorId { get; }

    /// <summary>Current lifecycle state.</summary>
    ActorLifecycleState State { get; }

    /// <summary>Hierarchy level (0 = leaf worker, 1+ = supervisors).</summary>
    int HierarchyLevel { get; }

    /// <summary>
    /// Execute one perception-action-learning cycle.
    /// Returns the outcome of this cycle for upstream aggregation.
    /// </summary>
    Task<ActorCycleResult> ExecuteCycleAsync(CancellationToken ct = default);

    /// <summary>
    /// Receive a backward teaching signal from the hierarchy.
    /// Propagates reward through this actor's patterns and children.
    /// </summary>
    void ReceiveBackwardSignal(double reward);
}

/// <summary>
/// DAPSA lifecycle states following the Fractal Actor specification.
/// </summary>
public enum ActorLifecycleState
{
    /// <summary>Waiting for next perception cycle.</summary>
    Idle,

    /// <summary>Phase 1: Perceiving environment, gathering data.</summary>
    Discovering,

    /// <summary>Phase 2: Choosing action (consulting Passive then Active).</summary>
    Deciding,

    /// <summary>Phase 3: Executing chosen action.</summary>
    Executing,

    /// <summary>Phase 4: Recording outcome, updating patterns and trajectory.</summary>
    Learning,

    /// <summary>Action was served from Passive Manifold (System 1 fossil hit).</summary>
    Fossilized,

    /// <summary>Terminal success state.</summary>
    Complete,

    /// <summary>Unrecoverable error.</summary>
    Error
}

/// <summary>
/// Result of a single actor perception-action-learning cycle.
/// </summary>
public sealed record ActorCycleResult
{
    /// <summary>Was the action served from passive (fossil) or active (search)?</summary>
    public required bool WasFossilHit { get; init; }

    /// <summary>The action chosen (domain-specific string representation).</summary>
    public required string Action { get; init; }

    /// <summary>The landmark ID where the action was taken.</summary>
    public required string LandmarkId { get; init; }

    /// <summary>Observed reward from the environment.</summary>
    public double Reward { get; init; }

    /// <summary>Time elapsed for this cycle (for energy tracking).</summary>
    public TimeSpan Elapsed { get; init; }

    /// <summary>Consonance error that triggered the mode decision.</summary>
    public double ConsonanceError { get; init; }
}
