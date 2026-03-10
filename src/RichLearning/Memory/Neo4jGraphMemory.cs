using Microsoft.Extensions.Logging;
using Neo4j.Driver;
using RichLearning.Abstractions;
using RichLearning.Models;

namespace RichLearning.Memory;

/// <summary>
/// Neo4j-backed topological graph memory.
/// Provides CRUD operations, nearest-neighbor queries, cycle detection,
/// frontier discovery, and graph-prioritised sampling over the state-space map.
/// 
/// This is the reference implementation of <see cref="IGraphMemory"/>.
/// For testing or lightweight use, see <see cref="InMemoryGraphMemory"/>.
/// </summary>
public sealed class Neo4jGraphMemory : IGraphMemory
{
    private readonly IDriver _driver;
    private readonly ILogger<Neo4jGraphMemory> _logger;
    private readonly string? _database;

    public Neo4jGraphMemory(
        string uri,
        string user,
        string password,
        ILogger<Neo4jGraphMemory> logger,
        string? database = null)
    {
        _driver = GraphDatabase.Driver(uri, AuthTokens.Basic(user, password));
        _logger = logger;
        _database = string.IsNullOrWhiteSpace(database) ? null : database;
    }

    private IAsyncSession GetSession() => string.IsNullOrWhiteSpace(_database)
        ? _driver.AsyncSession()
        : _driver.AsyncSession(o => o.WithDatabase(_database));

    // ── Schema ──

    public async Task InitialiseSchemaAsync()
    {
        await using var session = GetSession();
        await session.ExecuteWriteAsync(async tx =>
        {
            await tx.RunAsync(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Landmark) REQUIRE l.id IS UNIQUE");
            await tx.RunAsync(
                "CREATE INDEX IF NOT EXISTS FOR (l:Landmark) ON (l.clusterId)");
        });
        _logger.LogInformation("Neo4j schema initialised.");
    }

    // ── Node Operations ──

    public async Task UpsertLandmarkAsync(StateLandmark landmark)
    {
        await using var session = GetSession();
        var actionCountsJson = GraphMemorySerialization.SerializeActionCounts(landmark.ActionCounts);
        var metadataJson = GraphMemorySerialization.SerializeMetadata(landmark.Metadata);
        var episodicTracesJson = GraphMemorySerialization.SerializeEpisodicTraces(landmark.EpisodicTraces);

        await session.ExecuteWriteAsync(async tx =>
        {
            await tx.RunAsync(
                """
                MERGE (l:Landmark {id: $id})
                SET l.embedding        = $embedding,
                    l.visitCount        = $visitCount,
                    l.valueEstimate     = $valueEstimate,
                    l.noveltyScore      = $noveltyScore,
                    l.uncertaintyScore  = $uncertaintyScore,
                    l.clusterId         = $clusterId,
                    l.hierarchyLevel    = $hierarchyLevel,
                    l.childNodeIds      = $childNodeIds,
                    l.actionCountsJson  = $actionCountsJson,
                    l.lastVisited       = $lastVisited,
                    l.createdTimestep   = $createdTimestep,
                    l.metadataJson      = $metadataJson,
                    l.episodicTracesJson = $episodicTracesJson
                """,
                new
                {
                    id = landmark.Id,
                    embedding = landmark.Embedding.ToList(),
                    visitCount = landmark.VisitCount,
                    valueEstimate = landmark.ValueEstimate,
                    noveltyScore = landmark.NoveltyScore,
                    uncertaintyScore = landmark.UncertaintyScore,
                    clusterId = landmark.ClusterId,
                    hierarchyLevel = landmark.HierarchyLevel,
                    childNodeIds = landmark.ChildNodeIds.ToList(),
                    actionCountsJson,
                    lastVisited = landmark.LastVisitedTimestep,
                    createdTimestep = landmark.CreatedTimestep,
                    metadataJson,
                    episodicTracesJson
                });
        });
    }

    public async Task<StateLandmark?> GetLandmarkAsync(string id)
    {
        await using var session = GetSession();
        return await session.ExecuteReadAsync(async tx =>
        {
            var cursor = await tx.RunAsync(
                "MATCH (l:Landmark {id: $id}) RETURN l", new { id });
            if (await cursor.FetchAsync())
            {
                var node = cursor.Current["l"].As<INode>();
                return MapNodeToLandmark(node);
            }
            return null;
        });
    }

    public async Task<IReadOnlyList<StateLandmark>> GetAllLandmarksAsync(int? hierarchyLevel = null)
    {
        await using var session = GetSession();
        return await session.ExecuteReadAsync(async tx =>
        {
            var query = hierarchyLevel.HasValue
                ? "MATCH (l:Landmark) WHERE l.hierarchyLevel = $hl RETURN l"
                : "MATCH (l:Landmark) RETURN l";

            var p = hierarchyLevel.HasValue ? new { hl = hierarchyLevel.Value } : null;
            var cursor = await tx.RunAsync(query, p);
            var list = new List<StateLandmark>();
            while (await cursor.FetchAsync())
            {
                var node = cursor.Current["l"].As<INode>();
                list.Add(MapNodeToLandmark(node));
            }
            return (IReadOnlyList<StateLandmark>)list;
        });
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

    public async Task UpsertTransitionAsync(StateTransition transition)
    {
        await using var session = GetSession();
        await session.ExecuteWriteAsync(async tx =>
        {
            await tx.RunAsync(
                """
                MATCH (src:Landmark {id: $srcId})
                MATCH (tgt:Landmark {id: $tgtId})
                MERGE (src)-[t:TRANSITION {action: $action}]->(tgt)
                SET t.reward           = $reward,
                    t.rewardVariance   = $rewardVariance,
                    t.transitionCount  = $transitionCount,
                    t.successRate      = $successRate,
                    t.confidence       = $confidence,
                    t.temporalDistance = $temporalDistance,
                    t.tdError          = $tdError,
                    t.lastTrained      = $lastTrained,
                    t.actionCountsJson = $actionCountsJson,
                    t.isMacroEdge      = $isMacroEdge,
                    t.macroPathJson    = $macroPathJson
                """,
                new
                {
                    srcId = transition.SourceId,
                    tgtId = transition.TargetId,
                    action = transition.Action,
                    reward = transition.Reward,
                    rewardVariance = transition.RewardVariance,
                    transitionCount = transition.TransitionCount,
                    successRate = transition.SuccessRate,
                    confidence = transition.Confidence,
                    temporalDistance = transition.TemporalDistance,
                    tdError = transition.TdError,
                    lastTrained = transition.LastTrainedTimestep,
                    actionCountsJson = GraphMemorySerialization.SerializeActionCounts(transition.ActionCounts),
                    isMacroEdge = transition.IsMacroEdge,
                    macroPathJson = GraphMemorySerialization.SerializeMacroPath(transition.MacroPath)
                });
        });
    }

    public async Task<IReadOnlyList<StateTransition>> GetOutgoingTransitionsAsync(string landmarkId)
    {
        await using var session = GetSession();
        return await session.ExecuteReadAsync(async tx =>
        {
            var cursor = await tx.RunAsync(
                """
                MATCH (src:Landmark {id: $id})-[t:TRANSITION]->(tgt:Landmark)
                RETURN t.action AS action, t.reward AS reward, t.transitionCount AS cnt,
                      coalesce(t.rewardVariance, 0.0) AS rv,
                       t.successRate AS sr, t.temporalDistance AS td, t.tdError AS tde,
                      coalesce(t.confidence, 0.5) AS conf,
                      t.lastTrained AS lt, tgt.id AS tgtId,
                      coalesce(t.actionCountsJson, '') AS actionCountsJson,
                      coalesce(t.isMacroEdge, false) AS isMacroEdge,
                      coalesce(t.macroPathJson, '') AS macroPathJson
                """,
                new { id = landmarkId });

            var list = new List<StateTransition>();
            while (await cursor.FetchAsync())
            {
                var r = cursor.Current;
                list.Add(new StateTransition
                {
                    SourceId = landmarkId,
                    TargetId = r["tgtId"].As<string>(),
                    Action = r["action"].As<int>(),
                    Reward = r["reward"].As<double>(),
                    RewardVariance = r["rv"].As<double>(),
                    TransitionCount = r["cnt"].As<int>(),
                    SuccessRate = r["sr"].As<double>(),
                    Confidence = r["conf"].As<double>(),
                    TemporalDistance = r["td"].As<int>(),
                    TdError = r["tde"].As<double>(),
                    LastTrainedTimestep = r["lt"].As<long>(),
                    ActionCounts = GraphMemorySerialization.DeserializeActionCounts(r["actionCountsJson"].As<string>()),
                    IsMacroEdge = r["isMacroEdge"].As<bool>(),
                    MacroPath = GraphMemorySerialization.DeserializeMacroPath(r["macroPathJson"].As<string>())
                });
            }
            return (IReadOnlyList<StateTransition>)list;
        });
    }

    // ── Graph Queries ──

    public async Task<IReadOnlyList<string>> ShortestPathAsync(string fromId, string toId)
    {
        await using var session = GetSession();
        return await session.ExecuteReadAsync(async tx =>
        {
            var cursor = await tx.RunAsync(
                """
                MATCH path = shortestPath(
                    (src:Landmark {id: $fromId})-[:TRANSITION*1..20]->(tgt:Landmark {id: $toId})
                )
                RETURN [n IN nodes(path) | n.id] AS pathIds
                """,
                new { fromId, toId });

            if (await cursor.FetchAsync())
                return cursor.Current["pathIds"].As<List<string>>().AsReadOnly()
                    as IReadOnlyList<string>;
            return (IReadOnlyList<string>)Array.Empty<string>();
        });
    }

    public async Task<IReadOnlyList<string>> DetectCycleInTrajectoryAsync(IReadOnlyList<string> recentIds)
    {
        // Local detection: repeated IDs indicate a loop.
        var seen = new HashSet<string>();
        var cycleNodes = new HashSet<string>();
        foreach (var id in recentIds)
        {
            if (!seen.Add(id))
                cycleNodes.Add(id);
        }

        if (cycleNodes.Count == 0 && recentIds.Count >= 3)
        {
            await using var session = GetSession();
            var result = await session.ExecuteReadAsync(async tx =>
            {
                var cursor = await tx.RunAsync(
                    """
                    UNWIND $ids AS startId
                    MATCH path = (s:Landmark {id: startId})-[:TRANSITION*2..6]->(s)
                    WITH nodes(path) AS cycle_nodes LIMIT 1
                    UNWIND cycle_nodes AS n
                    RETURN DISTINCT n.id AS nodeId
                    """,
                    new { ids = recentIds.ToList() });

                var ids = new List<string>();
                while (await cursor.FetchAsync())
                    ids.Add(cursor.Current["nodeId"].As<string>());
                return ids;
            });
            return result;
        }

        return cycleNodes.ToList();
    }

    public async Task<IReadOnlyList<StateLandmark>> GetFrontierLandmarksAsync(int topK = 5)
    {
        await using var session = GetSession();
        return await session.ExecuteReadAsync(async tx =>
        {
            var cursor = await tx.RunAsync(
                """
                MATCH (l:Landmark)
                OPTIONAL MATCH (l)-[t:TRANSITION]->()
                WITH l, count(t) AS outDegree
                WITH l, (l.noveltyScore / (1.0 + l.visitCount)) * (1.0 / (1.0 + outDegree)) AS frontierScore
                ORDER BY frontierScore DESC
                LIMIT $topK
                RETURN l
                """,
                new { topK });

            var list = new List<StateLandmark>();
            while (await cursor.FetchAsync())
            {
                var node = cursor.Current["l"].As<INode>();
                list.Add(MapNodeToLandmark(node));
            }
            return (IReadOnlyList<StateLandmark>)list;
        });
    }

    public async Task<IReadOnlyList<StateTransition>> PrioritisedSampleAsync(
        int batchSize, long currentTimestep)
    {
        await using var session = GetSession();
        return await session.ExecuteReadAsync(async tx =>
        {
            var cursor = await tx.RunAsync(
                """
                MATCH (src:Landmark)-[t:TRANSITION]->(tgt:Landmark)
                WITH src, t, tgt,
                     (abs(t.tdError) + 0.01) / (toFloat(t.transitionCount) + 1.0) AS priorityBase,
                     toFloat($currentTs - t.lastTrained + 1) AS staleness
                WITH src, t, tgt,
                     priorityBase * staleness AS priority
                ORDER BY priority DESC
                LIMIT $batchSize
                RETURN src.id AS srcId, tgt.id AS tgtId, t.action AS action,
                       t.reward AS reward, t.transitionCount AS cnt,
                      coalesce(t.rewardVariance, 0.0) AS rv,
                       t.successRate AS sr, t.temporalDistance AS td,
                      t.tdError AS tde, t.lastTrained AS lt,
                      coalesce(t.confidence, 0.5) AS conf,
                      coalesce(t.actionCountsJson, '') AS actionCountsJson,
                      coalesce(t.isMacroEdge, false) AS isMacroEdge,
                      coalesce(t.macroPathJson, '') AS macroPathJson
                """,
                new { batchSize, currentTs = currentTimestep });

            var list = new List<StateTransition>();
            while (await cursor.FetchAsync())
            {
                var r = cursor.Current;
                list.Add(new StateTransition
                {
                    SourceId = r["srcId"].As<string>(),
                    TargetId = r["tgtId"].As<string>(),
                    Action = r["action"].As<int>(),
                    Reward = r["reward"].As<double>(),
                    RewardVariance = r["rv"].As<double>(),
                    TransitionCount = r["cnt"].As<int>(),
                    SuccessRate = r["sr"].As<double>(),
                    Confidence = r["conf"].As<double>(),
                    TemporalDistance = r["td"].As<int>(),
                    TdError = r["tde"].As<double>(),
                    LastTrainedTimestep = r["lt"].As<long>(),
                    ActionCounts = GraphMemorySerialization.DeserializeActionCounts(r["actionCountsJson"].As<string>()),
                    IsMacroEdge = r["isMacroEdge"].As<bool>(),
                    MacroPath = GraphMemorySerialization.DeserializeMacroPath(r["macroPathJson"].As<string>())
                });
            }
            return (IReadOnlyList<StateTransition>)list;
        });
    }

    // ── Graph Maintenance ──

    public async Task AssignClustersAsync(int rounds = 5)
    {
        await using var session = GetSession();
        await session.ExecuteWriteAsync(async tx =>
        {
            await tx.RunAsync("MATCH (l:Landmark) SET l.clusterId = id(l)");
            for (int i = 0; i < rounds; i++)
            {
                await tx.RunAsync(
                    """
                    MATCH (l:Landmark)-[:TRANSITION]-(neighbour:Landmark)
                    WITH l, min(neighbour.clusterId) AS minCluster
                    WHERE minCluster < l.clusterId
                    SET l.clusterId = minCluster
                    """);
            }
        });
        _logger.LogInformation("Cluster assignment completed ({Rounds} rounds).", rounds);
    }

    public async Task<(int Landmarks, int Transitions)> GetGraphStatsAsync()
    {
        await using var session = GetSession();
        return await session.ExecuteReadAsync(async tx =>
        {
            var cursor = await tx.RunAsync(
                """
                MATCH (l:Landmark)
                OPTIONAL MATCH ()-[t:TRANSITION]->()
                RETURN count(DISTINCT l) AS landmarks, count(DISTINCT t) AS transitions
                """);
            await cursor.FetchAsync();
            return (
                cursor.Current["landmarks"].As<int>(),
                cursor.Current["transitions"].As<int>()
            );
        });
    }

    // ── Pruning / Decay ──

    public async Task<bool> RemoveLandmarkAsync(string id)
    {
        await using var session = GetSession();
        return await session.ExecuteWriteAsync(async tx =>
        {
            var cursor = await tx.RunAsync(
                """
                MATCH (l:Landmark {id: $id})
                OPTIONAL MATCH (l)-[r:TRANSITION]-()
                DELETE r, l
                RETURN count(l) AS deleted
                """,
                new { id });
            await cursor.FetchAsync();
            return cursor.Current["deleted"].As<int>() > 0;
        });
    }

    public async Task<bool> RemoveTransitionAsync(string sourceId, string targetId, int action)
    {
        await using var session = GetSession();
        return await session.ExecuteWriteAsync(async tx =>
        {
            var cursor = await tx.RunAsync(
                """
                MATCH (s:Landmark {id: $sourceId})-[r:TRANSITION {action: $action}]->(t:Landmark {id: $targetId})
                DELETE r
                RETURN count(r) AS deleted
                """,
                new { sourceId, targetId, action });
            await cursor.FetchAsync();
            return cursor.Current["deleted"].As<int>() > 0;
        });
    }

    // ── Helpers ──

    private static StateLandmark MapNodeToLandmark(INode node)
    {
        var actionCounts = node.Properties.ContainsKey("actionCountsJson")
            ? GraphMemorySerialization.DeserializeActionCounts(node["actionCountsJson"].As<string>())
            : new Dictionary<int, int>();

        var childIds = node.Properties.ContainsKey("childNodeIds")
            ? node["childNodeIds"].As<List<string>>()
            : new List<string>();

        var metadata = node.Properties.ContainsKey("metadataJson")
            ? GraphMemorySerialization.DeserializeMetadata(node["metadataJson"].As<string>())
            : new Dictionary<string, object>();
        var episodicTraces = node.Properties.ContainsKey("episodicTracesJson")
            ? GraphMemorySerialization.DeserializeEpisodicTraces(node["episodicTracesJson"].As<string>())
            : new List<EpisodicTrace>();

        return new StateLandmark
        {
            Id = node["id"].As<string>(),
            Embedding = node["embedding"].As<List<double>>().ToArray(),
            VisitCount = node["visitCount"].As<int>(),
            ValueEstimate = node["valueEstimate"].As<double>(),
            NoveltyScore = node["noveltyScore"].As<double>(),
            UncertaintyScore = node.Properties.ContainsKey("uncertaintyScore")
                ? node["uncertaintyScore"].As<double>() : 1.0,
            ClusterId = node["clusterId"].As<int>(),
            HierarchyLevel = node.Properties.ContainsKey("hierarchyLevel")
                ? node["hierarchyLevel"].As<int>() : 0,
            ChildNodeIds = childIds,
            LastVisitedTimestep = node["lastVisited"].As<long>(),
            CreatedTimestep = node["createdTimestep"].As<long>(),
            ActionCounts = actionCounts,
            EpisodicTraces = episodicTraces,
            Metadata = metadata
        };
    }

    public async ValueTask DisposeAsync()
    {
        _driver.Dispose();
        await Task.CompletedTask;
    }
}
