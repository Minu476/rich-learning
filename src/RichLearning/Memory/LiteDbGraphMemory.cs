using LiteDB;
using Microsoft.Extensions.Logging;
using RichLearning.Abstractions;
using RichLearning.Models;

namespace RichLearning.Memory;

/// <summary>
/// LiteDB-backed topological graph memory — lightweight embedded alternative to Neo4j.
/// 
/// Ideal for PoC / training mode:
///   • Zero infrastructure (no Docker, no server)
///   • Single-file database (~1 MB for typical PoC runs)
///   • Full IGraphMemory contract with graph queries implemented in-process
///   • Comparable accuracy — only differs in I/O latency, not semantics
/// 
/// Trade-offs vs Neo4j:
///   ✓ No setup, instant start
///   ✓ Portable (single .db file)
///   ✗ No native graph traversal engine (BFS/DFS done in C#)
///   ✗ Not suitable for very large graphs (>100K nodes)
/// </summary>
public sealed class LiteDbGraphMemory : IGraphMemory
{
    private readonly LiteDatabase _db;
    private readonly ILiteCollection<LandmarkDoc> _landmarks;
    private readonly ILiteCollection<TransitionDoc> _transitions;
    private readonly ILogger<LiteDbGraphMemory> _logger;

    public LiteDbGraphMemory(string dbPath, ILogger<LiteDbGraphMemory> logger)
    {
        _db = new LiteDatabase(dbPath);
        _landmarks = _db.GetCollection<LandmarkDoc>("landmarks");
        _transitions = _db.GetCollection<TransitionDoc>("transitions");
        _logger = logger;
    }

    // ── Schema ──

    public Task InitialiseSchemaAsync()
    {
        _landmarks.EnsureIndex(x => x.LandmarkId, unique: true);
        _landmarks.EnsureIndex(x => x.ClusterId);
        _transitions.EnsureIndex(x => x.SourceId);
        _transitions.EnsureIndex(x => x.TargetId);
        _transitions.EnsureIndex(x => x.CompositeKey, unique: true);
        _logger.LogInformation("LiteDB schema initialised.");
        return Task.CompletedTask;
    }

    // ── Node Operations ──

    public Task UpsertLandmarkAsync(StateLandmark landmark)
    {
        var doc = new LandmarkDoc
        {
            LandmarkId = landmark.Id,
            Embedding = landmark.Embedding,
            VisitCount = landmark.VisitCount,
            ValueEstimate = landmark.ValueEstimate,
            NoveltyScore = landmark.NoveltyScore,
            UncertaintyScore = landmark.UncertaintyScore,
            ClusterId = landmark.ClusterId,
            HierarchyLevel = landmark.HierarchyLevel,
            ChildNodeIds = landmark.ChildNodeIds.ToList(),
            ActionCounts = landmark.ActionCounts,
            LastVisitedTimestep = landmark.LastVisitedTimestep,
            CreatedTimestep = landmark.CreatedTimestep
        };

        var existing = _landmarks.FindOne(x => x.LandmarkId == landmark.Id);
        if (existing != null)
        {
            doc.Id = existing.Id;
            _landmarks.Update(doc);
        }
        else
        {
            _landmarks.Insert(doc);
        }

        return Task.CompletedTask;
    }

    public Task<StateLandmark?> GetLandmarkAsync(string id)
    {
        var doc = _landmarks.FindOne(x => x.LandmarkId == id);
        return Task.FromResult(doc != null ? MapToLandmark(doc) : (StateLandmark?)null);
    }

    public Task<IReadOnlyList<StateLandmark>> GetAllLandmarksAsync()
    {
        var all = _landmarks.FindAll().Select(MapToLandmark).ToList();
        return Task.FromResult<IReadOnlyList<StateLandmark>>(all);
    }

    public async Task<(StateLandmark Landmark, double Distance)?> NearestNeighbourAsync(
        double[] embedding, IStateEncoder encoder)
    {
        var all = await GetAllLandmarksAsync();
        if (all.Count == 0) return null;

        StateLandmark? best = null;
        double bestDist = double.MaxValue;

        foreach (var lm in all)
        {
            double dist = encoder.Distance(embedding, lm.Embedding);
            if (dist < bestDist)
            {
                bestDist = dist;
                best = lm;
            }
        }

        return best is null ? null : (best, bestDist);
    }

    // ── Edge Operations ──

    public Task UpsertTransitionAsync(StateTransition transition)
    {
        string key = $"{transition.SourceId}|{transition.Action}|{transition.TargetId}";
        var doc = new TransitionDoc
        {
            CompositeKey = key,
            SourceId = transition.SourceId,
            TargetId = transition.TargetId,
            Action = transition.Action,
            Reward = transition.Reward,
            TransitionCount = transition.TransitionCount,
            SuccessRate = transition.SuccessRate,
            TemporalDistance = transition.TemporalDistance,
            TdError = transition.TdError,
            LastTrainedTimestep = transition.LastTrainedTimestep
        };

        var existing = _transitions.FindOne(x => x.CompositeKey == key);
        if (existing != null)
        {
            doc.Id = existing.Id;
            _transitions.Update(doc);
        }
        else
        {
            _transitions.Insert(doc);
        }

        return Task.CompletedTask;
    }

    public Task<IReadOnlyList<StateTransition>> GetOutgoingTransitionsAsync(string landmarkId)
    {
        var list = _transitions
            .Find(x => x.SourceId == landmarkId)
            .Select(MapToTransition)
            .ToList();
        return Task.FromResult<IReadOnlyList<StateTransition>>(list);
    }

    // ── Graph Queries ──

    public Task<IReadOnlyList<string>> ShortestPathAsync(string fromId, string toId)
    {
        // BFS shortest path
        var adjacency = BuildAdjacencyList();
        var queue = new Queue<string>();
        var visited = new Dictionary<string, string?>(); // node → parent

        queue.Enqueue(fromId);
        visited[fromId] = null;

        while (queue.Count > 0)
        {
            var current = queue.Dequeue();
            if (current == toId)
            {
                // Reconstruct path
                var path = new List<string>();
                string? node = toId;
                while (node != null)
                {
                    path.Add(node);
                    node = visited[node];
                }
                path.Reverse();
                return Task.FromResult<IReadOnlyList<string>>(path);
            }

            if (adjacency.TryGetValue(current, out var neighbours))
            {
                foreach (var neighbour in neighbours)
                {
                    if (!visited.ContainsKey(neighbour))
                    {
                        visited[neighbour] = current;
                        queue.Enqueue(neighbour);
                    }
                }
            }
        }

        return Task.FromResult<IReadOnlyList<string>>(Array.Empty<string>());
    }

    public Task<IReadOnlyList<string>> DetectCycleInTrajectoryAsync(IReadOnlyList<string> recentIds)
    {
        // Local detection: repeated IDs indicate a loop
        var seen = new HashSet<string>();
        var cycleNodes = new HashSet<string>();
        foreach (var id in recentIds)
        {
            if (!seen.Add(id))
                cycleNodes.Add(id);
        }

        if (cycleNodes.Count == 0 && recentIds.Count >= 3)
        {
            // Try graph-based cycle detection via DFS
            var adjacency = BuildAdjacencyList();
            foreach (var startId in recentIds)
            {
                if (!adjacency.ContainsKey(startId)) continue;
                var cyclePath = FindCycleDfs(adjacency, startId, maxDepth: 6);
                if (cyclePath.Count > 0)
                    return Task.FromResult<IReadOnlyList<string>>(cyclePath);
            }
        }

        return Task.FromResult<IReadOnlyList<string>>(cycleNodes.ToList());
    }

    public Task<IReadOnlyList<StateLandmark>> GetFrontierLandmarksAsync(int topK = 5)
    {
        var allLandmarks = _landmarks.FindAll().ToList();
        var outDegrees = new Dictionary<string, int>();

        foreach (var t in _transitions.FindAll())
        {
            outDegrees.TryGetValue(t.SourceId, out int count);
            outDegrees[t.SourceId] = count + 1;
        }

        var ranked = allLandmarks
            .Select(l =>
            {
                outDegrees.TryGetValue(l.LandmarkId, out int outDeg);
                double frontierScore = (l.NoveltyScore / (1.0 + l.VisitCount))
                                       * (1.0 / (1.0 + outDeg));
                return (Doc: l, Score: frontierScore);
            })
            .OrderByDescending(x => x.Score)
            .Take(topK)
            .Select(x => MapToLandmark(x.Doc))
            .ToList();

        return Task.FromResult<IReadOnlyList<StateLandmark>>(ranked);
    }

    public Task<IReadOnlyList<StateTransition>> PrioritisedSampleAsync(
        int batchSize, long currentTimestep)
    {
        var allTransitions = _transitions.FindAll().ToList();

        var ranked = allTransitions
            .Select(t =>
            {
                double priorityBase = (Math.Abs(t.TdError) + 0.01) / (t.TransitionCount + 1.0);
                double staleness = currentTimestep - t.LastTrainedTimestep + 1;
                return (Doc: t, Priority: priorityBase * staleness);
            })
            .OrderByDescending(x => x.Priority)
            .Take(batchSize)
            .Select(x => MapToTransition(x.Doc))
            .ToList();

        return Task.FromResult<IReadOnlyList<StateTransition>>(ranked);
    }

    // ── Graph Maintenance ──

    public Task AssignClustersAsync(int rounds = 5)
    {
        // Label propagation: each node starts as its own cluster
        var allLandmarks = _landmarks.FindAll().ToList();
        var clusterMap = new Dictionary<string, int>();

        for (int i = 0; i < allLandmarks.Count; i++)
            clusterMap[allLandmarks[i].LandmarkId] = i;

        var adjacency = BuildUndirectedAdjacencyList();

        for (int r = 0; r < rounds; r++)
        {
            foreach (var lm in allLandmarks)
            {
                if (!adjacency.TryGetValue(lm.LandmarkId, out var neighbours)) continue;
                int minCluster = clusterMap[lm.LandmarkId];
                foreach (var n in neighbours)
                {
                    if (clusterMap.TryGetValue(n, out int nc) && nc < minCluster)
                        minCluster = nc;
                }
                clusterMap[lm.LandmarkId] = minCluster;
            }
        }

        // Write back
        foreach (var lm in allLandmarks)
        {
            lm.ClusterId = clusterMap[lm.LandmarkId];
            _landmarks.Update(lm);
        }

        _logger.LogInformation("Cluster assignment completed ({Rounds} rounds).", rounds);
        return Task.CompletedTask;
    }

    public Task<(int Landmarks, int Transitions)> GetGraphStatsAsync()
    {
        int landmarks = _landmarks.Count();
        int transitions = _transitions.Count();
        return Task.FromResult((landmarks, transitions));
    }

    // ── Helpers ──

    private Dictionary<string, List<string>> BuildAdjacencyList()
    {
        var adj = new Dictionary<string, List<string>>();
        foreach (var t in _transitions.FindAll())
        {
            if (!adj.TryGetValue(t.SourceId, out var list))
            {
                list = new List<string>();
                adj[t.SourceId] = list;
            }
            if (!list.Contains(t.TargetId))
                list.Add(t.TargetId);
        }
        return adj;
    }

    private Dictionary<string, List<string>> BuildUndirectedAdjacencyList()
    {
        var adj = new Dictionary<string, List<string>>();

        void AddEdge(string a, string b)
        {
            if (!adj.TryGetValue(a, out var list))
            {
                list = new List<string>();
                adj[a] = list;
            }
            if (!list.Contains(b)) list.Add(b);
        }

        foreach (var t in _transitions.FindAll())
        {
            AddEdge(t.SourceId, t.TargetId);
            AddEdge(t.TargetId, t.SourceId);
        }
        return adj;
    }

    private static List<string> FindCycleDfs(
        Dictionary<string, List<string>> adj, string start, int maxDepth)
    {
        var stack = new Stack<(string Node, List<string> Path)>();
        stack.Push((start, new List<string> { start }));

        while (stack.Count > 0)
        {
            var (node, path) = stack.Pop();
            if (path.Count > maxDepth + 1) continue;

            if (!adj.TryGetValue(node, out var neighbours)) continue;
            foreach (var next in neighbours)
            {
                if (next == start && path.Count >= 3)
                    return path.Distinct().ToList();
                if (path.Count <= maxDepth && !path.Contains(next))
                    stack.Push((next, [.. path, next]));
            }
        }
        return [];
    }

    private static StateLandmark MapToLandmark(LandmarkDoc doc) => new()
    {
        Id = doc.LandmarkId,
        Embedding = doc.Embedding,
        VisitCount = doc.VisitCount,
        ValueEstimate = doc.ValueEstimate,
        NoveltyScore = doc.NoveltyScore,
        UncertaintyScore = doc.UncertaintyScore,
        ClusterId = doc.ClusterId,
        HierarchyLevel = doc.HierarchyLevel,
        ChildNodeIds = doc.ChildNodeIds,
        LastVisitedTimestep = doc.LastVisitedTimestep,
        CreatedTimestep = doc.CreatedTimestep,
        ActionCounts = doc.ActionCounts ?? new()
    };

    private static StateTransition MapToTransition(TransitionDoc doc) => new()
    {
        SourceId = doc.SourceId,
        TargetId = doc.TargetId,
        Action = doc.Action,
        Reward = doc.Reward,
        TransitionCount = doc.TransitionCount,
        SuccessRate = doc.SuccessRate,
        TemporalDistance = doc.TemporalDistance,
        TdError = doc.TdError,
        LastTrainedTimestep = doc.LastTrainedTimestep
    };

    public ValueTask DisposeAsync()
    {
        _db.Dispose();
        return ValueTask.CompletedTask;
    }

    // ── LiteDB Document Models ──

    internal class LandmarkDoc
    {
        public int Id { get; set; }
        public string LandmarkId { get; set; } = string.Empty;
        public double[] Embedding { get; set; } = [];
        public int VisitCount { get; set; }
        public double ValueEstimate { get; set; }
        public double NoveltyScore { get; set; } = 1.0;
        public double UncertaintyScore { get; set; } = 1.0;
        public int ClusterId { get; set; }
        public int HierarchyLevel { get; set; }
        public List<string> ChildNodeIds { get; set; } = [];
        public Dictionary<int, int> ActionCounts { get; set; } = new();
        public long LastVisitedTimestep { get; set; }
        public long CreatedTimestep { get; set; }
    }

    internal class TransitionDoc
    {
        public int Id { get; set; }
        public string CompositeKey { get; set; } = string.Empty;
        public string SourceId { get; set; } = string.Empty;
        public string TargetId { get; set; } = string.Empty;
        public int Action { get; set; }
        public double Reward { get; set; }
        public int TransitionCount { get; set; } = 1;
        public double SuccessRate { get; set; } = 1.0;
        public int TemporalDistance { get; set; } = 1;
        public double TdError { get; set; }
        public long LastTrainedTimestep { get; set; }
    }
}
