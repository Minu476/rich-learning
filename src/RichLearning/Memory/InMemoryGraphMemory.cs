using RichLearning.Abstractions;
using RichLearning.Models;

namespace RichLearning.Memory;

/// <summary>
/// In-memory implementation of IGraphMemory.
/// Zero external dependencies — suitable for PoCs, testing, and embedded use.
///
/// For production deployments, use:
///   - Neo4jGraphMemory (server, Cypher-native, billions of nodes)
///   - LiteDbGraphMemory (embedded, persistent, zero-setup)
///
/// This implementation provides all core operations:
///   - O(n) nearest-neighbour (linear scan; use ANN indexes for scale)
///   - BFS shortest path
///   - Hash-based cycle detection
///   - Label propagation clustering
///   - Prioritised replay sampling
/// </summary>
public sealed class InMemoryGraphMemory : IGraphMemory
{
    private readonly Dictionary<string, StateLandmark> _landmarks = new(StringComparer.Ordinal);
    private readonly Dictionary<string, List<StateTransition>> _outgoing = new(StringComparer.Ordinal);
    private readonly Dictionary<string, StateTransition> _edges = new(StringComparer.Ordinal);

    public Task InitialiseSchemaAsync() => Task.CompletedTask;

    // ── Node Operations ──

    public Task UpsertLandmarkAsync(StateLandmark landmark)
    {
        _landmarks[landmark.Id] = landmark;
        return Task.CompletedTask;
    }

    public Task<StateLandmark?> GetLandmarkAsync(string id)
    {
        _landmarks.TryGetValue(id, out var lm);
        return Task.FromResult(lm);
    }

    public Task<IReadOnlyList<StateLandmark>> GetAllLandmarksAsync(int? hierarchyLevel = null)
    {
        IReadOnlyList<StateLandmark> result = hierarchyLevel.HasValue
            ? _landmarks.Values.Where(l => l.HierarchyLevel == hierarchyLevel.Value).ToList()
            : _landmarks.Values.ToList();
        return Task.FromResult(result);
    }

    public Task<(StateLandmark Landmark, double Distance)?> NearestNeighbourAsync(
        double[] embedding, IStateEncoder encoder)
    {
        if (_landmarks.Count == 0)
            return Task.FromResult<(StateLandmark, double)?>(null);

        StateLandmark? best = null;
        double bestDist = double.MaxValue;

        foreach (var lm in _landmarks.Values)
        {
            if (lm.HierarchyLevel > 0) continue; // only compare base-level landmarks

            var d = encoder.Distance(embedding, lm.Embedding);
            if (d < bestDist)
            {
                bestDist = d;
                best = lm;
            }
        }

        return best is null
            ? Task.FromResult<(StateLandmark, double)?>(null)
            : Task.FromResult<(StateLandmark, double)?>((best, bestDist));
    }

    // ── Edge Operations ──

    public Task UpsertTransitionAsync(StateTransition transition)
    {
        var edgeKey = $"{transition.SourceId}→{transition.TargetId}→{transition.Action}";
        _edges[edgeKey] = transition;

        if (!_outgoing.TryGetValue(transition.SourceId, out var list))
        {
            list = [];
            _outgoing[transition.SourceId] = list;
        }

        // Replace existing or add new
        var idx = list.FindIndex(t =>
            t.TargetId == transition.TargetId && t.Action == transition.Action);
        if (idx >= 0)
            list[idx] = transition;
        else
            list.Add(transition);

        return Task.CompletedTask;
    }

    public Task<IReadOnlyList<StateTransition>> GetOutgoingTransitionsAsync(string landmarkId)
    {
        _outgoing.TryGetValue(landmarkId, out var list);
        IReadOnlyList<StateTransition> result = list ?? [];
        return Task.FromResult(result);
    }

    // ── Graph Queries ──

    /// <summary>BFS shortest path between two landmarks.</summary>
    public Task<IReadOnlyList<string>> ShortestPathAsync(string fromId, string toId)
    {
        if (fromId == toId)
            return Task.FromResult<IReadOnlyList<string>>(new[] { fromId });

        var queue = new Queue<string>();
        var visited = new HashSet<string>(StringComparer.Ordinal);
        var parent = new Dictionary<string, string>(StringComparer.Ordinal);

        queue.Enqueue(fromId);
        visited.Add(fromId);

        while (queue.Count > 0)
        {
            var current = queue.Dequeue();

            if (!_outgoing.TryGetValue(current, out var transitions))
                continue;

            foreach (var t in transitions)
            {
                if (visited.Add(t.TargetId))
                {
                    parent[t.TargetId] = current;
                    if (t.TargetId == toId)
                    {
                        // Reconstruct path
                        var path = new List<string>();
                        var c = toId;
                        while (c != fromId)
                        {
                            path.Add(c);
                            c = parent[c];
                        }
                        path.Add(fromId);
                        path.Reverse();
                        return Task.FromResult<IReadOnlyList<string>>(path);
                    }

                    queue.Enqueue(t.TargetId);
                }
            }
        }

        return Task.FromResult<IReadOnlyList<string>>([]); // no path found
    }

    /// <summary>Detect cycles in a trajectory by checking for repeated IDs.</summary>
    public Task<IReadOnlyList<string>> DetectCycleInTrajectoryAsync(IReadOnlyList<string> recentIds)
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var cycle = new List<string>();

        foreach (var id in recentIds)
        {
            if (!seen.Add(id))
            {
                // Found a cycle — collect all IDs from the first occurrence
                bool collecting = false;
                foreach (var id2 in recentIds)
                {
                    if (id2 == id) collecting = true;
                    if (collecting) cycle.Add(id2);
                }
                break;
            }
        }

        return Task.FromResult<IReadOnlyList<string>>(cycle);
    }

    /// <summary>Frontier landmarks ranked by exploration priority.</summary>
    public Task<IReadOnlyList<StateLandmark>> GetFrontierLandmarksAsync(int topK = 5)
    {
        IReadOnlyList<StateLandmark> result = _landmarks.Values
            .Where(l => l.HierarchyLevel == 0)
            .OrderByDescending(l => l.NoveltyScore / Math.Max(1, l.VisitCount))
            .ThenBy(l => _outgoing.TryGetValue(l.Id, out var outs) ? outs.Count : 0)
            .Take(topK)
            .ToList();
        return Task.FromResult(result);
    }

    /// <summary>Prioritised replay sampling weighted by TD-error and staleness.</summary>
    public Task<IReadOnlyList<StateTransition>> PrioritisedSampleAsync(int batchSize, long currentTimestep)
    {
        IReadOnlyList<StateTransition> result = _edges.Values
            .OrderByDescending(t =>
                Math.Abs(t.TdError) + 0.01 * (currentTimestep - t.LastTrainedTimestep))
            .Take(batchSize)
            .ToList();
        return Task.FromResult(result);
    }

    // ── Graph Maintenance ──

    /// <summary>Label propagation community detection.</summary>
    public Task AssignClustersAsync(int rounds = 5)
    {
        // Initialise each landmark with its own cluster
        int clusterCounter = 0;
        foreach (var lm in _landmarks.Values)
        {
            lm.ClusterId = clusterCounter++;
        }

        for (int round = 0; round < rounds; round++)
        {
            foreach (var lm in _landmarks.Values)
            {
                if (!_outgoing.TryGetValue(lm.Id, out var transitions) || transitions.Count == 0)
                    continue;

                // Count neighbor cluster labels
                var labelCounts = new Dictionary<int, int>();
                foreach (var t in transitions)
                {
                    if (_landmarks.TryGetValue(t.TargetId, out var neighbor))
                    {
                        labelCounts.TryGetValue(neighbor.ClusterId, out var count);
                        labelCounts[neighbor.ClusterId] = count + 1;
                    }
                }

                // Adopt the most common neighbor label
                if (labelCounts.Count > 0)
                {
                    lm.ClusterId = labelCounts.MaxBy(kv => kv.Value).Key;
                }
            }
        }

        return Task.CompletedTask;
    }

    public Task<(int Landmarks, int Transitions)> GetGraphStatsAsync()
    {
        return Task.FromResult((_landmarks.Count, _edges.Count));
    }

    // ── Pruning / Decay ──

    public Task<bool> RemoveLandmarkAsync(string id)
    {
        if (!_landmarks.Remove(id))
            return Task.FromResult(false);

        // Remove all outgoing transitions
        if (_outgoing.Remove(id, out var outList))
        {
            foreach (var t in outList)
                _edges.Remove($"{t.SourceId}→{t.TargetId}→{t.Action}");
        }

        // Remove all incoming transitions (edges targeting this landmark)
        var incomingKeys = _edges
            .Where(kv => kv.Value.TargetId == id)
            .Select(kv => kv.Key)
            .ToList();

        foreach (var key in incomingKeys)
        {
            var trans = _edges[key];
            _edges.Remove(key);

            if (_outgoing.TryGetValue(trans.SourceId, out var srcList))
                srcList.RemoveAll(t => t.TargetId == id);
        }

        return Task.FromResult(true);
    }

    public Task<bool> RemoveTransitionAsync(string sourceId, string targetId, int action)
    {
        var edgeKey = $"{sourceId}→{targetId}→{action}";
        if (!_edges.Remove(edgeKey))
            return Task.FromResult(false);

        if (_outgoing.TryGetValue(sourceId, out var list))
            list.RemoveAll(t => t.TargetId == targetId && t.Action == action);

        return Task.FromResult(true);
    }

    public ValueTask DisposeAsync() => ValueTask.CompletedTask;
}
