using RichLearning.Models;

namespace RichLearning.Abstractions;

/// <summary>
/// Interface for graph-backed memory that stores and retrieves
/// landmarks (nodes) and transitions (edges) forming a topological map
/// of the state space.
/// 
/// Implementations may use Neo4j, in-memory graphs, SQLite, etc.
/// </summary>
public interface IGraphMemory : IAsyncDisposable
{
    /// <summary>Ensure indexes, constraints, and schema exist.</summary>
    Task InitialiseSchemaAsync();

    // ── Node Operations ──

    /// <summary>Add or update a landmark node.</summary>
    Task UpsertLandmarkAsync(StateLandmark landmark);

    /// <summary>Get a landmark by ID. Returns null if not found.</summary>
    Task<StateLandmark?> GetLandmarkAsync(string id);

    /// <summary>Get all landmarks.</summary>
    Task<IReadOnlyList<StateLandmark>> GetAllLandmarksAsync();

    /// <summary>
    /// Find the nearest landmark to the given embedding.
    /// Returns null if the graph is empty.
    /// </summary>
    Task<(StateLandmark Landmark, double Distance)?> NearestNeighbourAsync(
        double[] embedding, IStateEncoder encoder);

    // ── Edge Operations ──

    /// <summary>Add or update a transition edge.</summary>
    Task UpsertTransitionAsync(StateTransition transition);

    /// <summary>Get all outgoing transitions from a landmark.</summary>
    Task<IReadOnlyList<StateTransition>> GetOutgoingTransitionsAsync(string landmarkId);

    // ── Graph Queries ──

    /// <summary>Find shortest path between two landmarks. Returns ordered list of IDs.</summary>
    Task<IReadOnlyList<string>> ShortestPathAsync(string fromId, string toId);

    /// <summary>Detect cycles in a trajectory (list of recent landmark IDs).</summary>
    Task<IReadOnlyList<string>> DetectCycleInTrajectoryAsync(IReadOnlyList<string> recentIds);

    /// <summary>
    /// Get frontier landmarks ranked by exploration priority.
    /// Frontiers are nodes with high novelty relative to their visit count and connectivity.
    /// </summary>
    Task<IReadOnlyList<StateLandmark>> GetFrontierLandmarksAsync(int topK = 5);

    /// <summary>
    /// Sample transitions prioritised by learning value (TD-error, staleness, etc.).
    /// </summary>
    Task<IReadOnlyList<StateTransition>> PrioritisedSampleAsync(int batchSize, long currentTimestep);

    // ── Graph Maintenance ──

    /// <summary>Assign cluster IDs using community detection.</summary>
    Task AssignClustersAsync(int rounds = 5);

    /// <summary>Get basic graph statistics: (landmarkCount, transitionCount).</summary>
    Task<(int Landmarks, int Transitions)> GetGraphStatsAsync();
}
