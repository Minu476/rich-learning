namespace RichLearning.Abstractions;

/// <summary>
/// The Passive Manifold (S^P) — "The Librarian."
///
/// This is System 1: cheap, O(1), every-tick evaluation.
/// Stores fossilized knowledge — habits that have been validated through
/// repeated successful execution in the Active Manifold.
///
/// Properties:
///   - Immutable (or atomically rebuilt): zero contention under concurrent reads
///   - O(1) lookup: situation hash → action/evaluation
///   - Energy-efficient: no computation beyond hash lookup
///   - Domain-agnostic: stores (key → value) pairs regardless of domain
///
/// Domain implementations:
///   - Chess: FrozenDictionary&lt;ulong, FossilEntry&gt; (Zobrist → evaluation)
///   - Robotics: FrozenDictionary&lt;string, NavigationAction&gt; (state hash → manoeuvre)
///   - LLM/RAG: FrozenDictionary&lt;string, string&gt; (query hash → answer)
///   - Trading: FrozenDictionary&lt;string, TradeAction&gt; (regime hash → position)
/// </summary>
public interface IPassiveManifold<TKey, TValue> where TKey : notnull
{
    /// <summary>
    /// O(1) lookup: given a situation key, return the fossilized response if known.
    /// This is the hot path for System 1 (Passive Mode).
    /// </summary>
    bool TryLookup(TKey situationKey, out TValue? value);

    /// <summary>Total number of fossilized entries.</summary>
    int Count { get; }

    /// <summary>
    /// Atomically replace the entire passive manifold with a new snapshot.
    /// Called by the Fossilizer during the freeze cycle (Phi: S^A → S^P).
    /// </summary>
    void Rebuild(IReadOnlyDictionary<TKey, TValue> newEntries);
}

/// <summary>
/// The Active Manifold (S^A) — "The Detective."
///
/// This is System 2: expensive, on-demand deep evaluation.
/// Stores recently evaluated states that are still being verified.
/// Thread-safe and mutable — multiple search threads can write concurrently.
///
/// Properties:
///   - Lock-free or low-contention concurrent writes
///   - Mutable: evaluations are updated with new observations
///   - Contains trajectory tracking for backward reinforcement
///   - Entries may be promoted to Passive (fossilization) or discarded
///
/// Domain implementations decide the specific value type:
///   - Chess: ConcurrentDictionary&lt;ulong, FossilEntry&gt;
///   - Robotics: ConcurrentDictionary&lt;string, NavigationState&gt;
///   - Generic: ConcurrentDictionary&lt;string, ActiveEntry&gt;
/// </summary>
public interface IActiveManifold<TKey, TValue> where TKey : notnull
{
    /// <summary>Look up an entry in active memory.</summary>
    bool TryLookup(TKey key, out TValue? value);

    /// <summary>Add or update an entry in active memory.</summary>
    void AddOrUpdate(TKey key, TValue value);

    /// <summary>Remove a specific entry (e.g., after fossilization).</summary>
    bool TryRemove(TKey key);

    /// <summary>Total entries currently in active memory.</summary>
    int Count { get; }

    /// <summary>Get all active entries (for fossilization candidate scanning).</summary>
    IReadOnlyCollection<KeyValuePair<TKey, TValue>> GetAllEntries();

    /// <summary>Clear all active entries (e.g., on forced refresh).</summary>
    void Clear();
}

/// <summary>
/// Consonance check: determines whether the predicted outcome matches reality.
///
/// This is the "Wake-on-Surprise" trigger from Patent 3 (H-DPU / Scout-Solver).
/// When consonance error exceeds a threshold, the system switches from
/// cheap Passive mode to expensive Active mode (the Scout activates).
///
/// Consonance Error = |predicted − actual|
///
/// Domain implementations:
///   - Chess: |passive_eval − search_eval| > threshold_centipawns
///   - Robotics: |predicted_position − actual_position| > threshold_meters
///   - LLM: cosine_distance(predicted_answer, actual_answer) > threshold
///   - Trading: |predicted_regime − actual_regime| > threshold
public interface IConsonanceChecker<TObservation, TAction>
{
    /// <summary>
    /// Evaluate whether the current observation is surprising enough
    /// to warrant activating the expensive Active Manifold (System 2).
    /// </summary>
    /// <param name="observation">The new observation from the environment.</param>
    /// <param name="predictedAction">The action predicted by the Passive Manifold (fossil).</param>
    /// <returns>
    /// A ConsonanceResult containing the error magnitude and the
    /// decision (passive/active mode).
    /// </returns>
    ConsonanceResult Check(TObservation observation, TAction predictedAction);
}

/// <summary>
/// Result of a consonance check.
/// </summary>
public readonly record struct ConsonanceResult(
    double Error,
    bool IsDissonant,
    string? Reason = null);
