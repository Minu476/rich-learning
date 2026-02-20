using RichLearning.Abstractions;
using RichLearning.Models;

namespace RichLearning.Planning;

/// <summary>
/// Builds hierarchical meta-level abstractions of the topological graph.
///
/// Meta-levels create progressively coarser views of the state space:
///   Level 0: Original landmarks (individual states)
///   Level 1: Connected components grouped by co-activation
///   Level 2: Super-regions grouping Level 1 clusters
///   Level 3: Strategic clusters grouping Level 2 regions
///
/// Each meta-level landmark represents a group of lower-level landmarks
/// with a centroid embedding. Meta-level transitions aggregate cross-boundary
/// transitions from the level below.
///
/// This is the hierarchical planning component — it enables multi-scale
/// planning (room → building → city) without pre-specifying the depth.
/// </summary>
public sealed class MetaLevelBuilder
{
    private readonly IGraphMemory _memory;

    public MetaLevelBuilder(IGraphMemory memory)
    {
        _memory = memory;
    }

    /// <summary>
    /// Build all meta-levels from the existing base graph.
    /// Returns the counts created at each level.
    /// </summary>
    public async Task<MetaLevelReport> BuildAllLevelsAsync()
    {
        var report = new MetaLevelReport();

        var baseLandmarks = await _memory.GetAllLandmarksAsync(hierarchyLevel: 0);
        if (baseLandmarks.Count < 2) return report;

        // Ensure clusters are assigned at base level
        await _memory.AssignClustersAsync();

        // Level 1: Group by cluster ID
        var meta1 = await BuildMetaLevelAsync(baseLandmarks, 1);
        report.Level1Count = meta1.Count;

        // Level 2: Group meta-1 landmarks by connectivity
        if (meta1.Count >= 2)
        {
            var meta2 = await BuildMetaLevelAsync(meta1, 2);
            report.Level2Count = meta2.Count;

            // Level 3: Group meta-2 landmarks
            if (meta2.Count >= 2)
            {
                var meta3 = await BuildMetaLevelAsync(meta2, 3);
                report.Level3Count = meta3.Count;
            }
        }

        return report;
    }

    /// <summary>
    /// Build a single meta-level by grouping the input landmarks.
    /// Uses union-find on transition connectivity.
    /// </summary>
    private async Task<IReadOnlyList<StateLandmark>> BuildMetaLevelAsync(
        IReadOnlyList<StateLandmark> inputLandmarks, int targetLevel)
    {
        var groups = GroupByConnectivity(inputLandmarks);
        var metaLandmarks = new List<StateLandmark>();

        foreach (var group in groups.Where(g => g.Count >= 2))
        {
            var centroid = ComputeCentroid(group);
            var childIds = group.Select(l => l.Id).ToList();

            var metaLandmark = new StateLandmark
            {
                Id = $"META_{targetLevel}_{centroid[..Math.Min(12, centroid.Length)]}",
                Embedding = group.First().Embedding, // Use first as representative
                VisitCount = group.Sum(l => l.VisitCount),
                ValueEstimate = group.Average(l => l.ValueEstimate),
                NoveltyScore = group.Average(l => l.NoveltyScore),
                HierarchyLevel = targetLevel,
                ChildNodeIds = childIds,
                CreatedTimestep = group.Max(l => l.CreatedTimestep),
                LastVisitedTimestep = group.Max(l => l.LastVisitedTimestep),
            };

            await _memory.UpsertLandmarkAsync(metaLandmark);
            metaLandmarks.Add(metaLandmark);
        }

        // Synthesise meta-level transitions
        await SynthesiseMetaTransitionsAsync(inputLandmarks, metaLandmarks, targetLevel);

        return metaLandmarks;
    }

    /// <summary>
    /// Groups landmarks by transition connectivity using union-find.
    /// </summary>
    private IReadOnlyList<IReadOnlyList<StateLandmark>> GroupByConnectivity(
        IReadOnlyList<StateLandmark> landmarks)
    {
        // Union-Find
        var parent = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var lm in landmarks)
            parent[lm.Id] = lm.Id;

        string Find(string x)
        {
            while (parent[x] != x)
            {
                parent[x] = parent[parent[x]]; // path compression
                x = parent[x];
            }
            return x;
        }

        void Union(string a, string b)
        {
            var rootA = Find(a);
            var rootB = Find(b);
            if (rootA != rootB) parent[rootA] = rootB;
        }

        // Group by cluster ID (which was assigned by label propagation)
        foreach (var lm in landmarks)
        {
            var sameCluster = landmarks
                .Where(other => other.Id != lm.Id && other.ClusterId == lm.ClusterId);

            foreach (var other in sameCluster)
                Union(lm.Id, other.Id);
        }

        // Collect groups
        var groups = new Dictionary<string, List<StateLandmark>>(StringComparer.Ordinal);
        foreach (var lm in landmarks)
        {
            var root = Find(lm.Id);
            if (!groups.TryGetValue(root, out var list))
            {
                list = [];
                groups[root] = list;
            }
            list.Add(lm);
        }

        return groups.Values.Select(g => (IReadOnlyList<StateLandmark>)g).ToList();
    }

    /// <summary>
    /// Creates transitions between meta-level landmarks by aggregating
    /// cross-boundary transitions from the level below.
    /// </summary>
    private async Task SynthesiseMetaTransitionsAsync(
        IReadOnlyList<StateLandmark> lowerLandmarks,
        IReadOnlyList<StateLandmark> metaLandmarks,
        int targetLevel)
    {
        // Build child→meta mapping
        var childToMeta = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var meta in metaLandmarks)
        {
            foreach (var childId in meta.ChildNodeIds)
                childToMeta[childId] = meta.Id;
        }

        // Aggregate cross-boundary transitions
        var metaEdges = new Dictionary<string, (string Src, string Tgt, double TotalReward, int Count)>();

        foreach (var lm in lowerLandmarks)
        {
            if (!childToMeta.TryGetValue(lm.Id, out var srcMeta))
                continue;

            var transitions = await _memory.GetOutgoingTransitionsAsync(lm.Id);
            foreach (var t in transitions)
            {
                if (!childToMeta.TryGetValue(t.TargetId, out var tgtMeta))
                    continue;

                if (srcMeta == tgtMeta) continue; // same meta-group, skip

                var edgeKey = $"{srcMeta}→{tgtMeta}";
                if (metaEdges.TryGetValue(edgeKey, out var existing))
                {
                    metaEdges[edgeKey] = (srcMeta, tgtMeta,
                        existing.TotalReward + t.Reward, existing.Count + 1);
                }
                else
                {
                    metaEdges[edgeKey] = (srcMeta, tgtMeta, t.Reward, 1);
                }
            }
        }

        foreach (var (_, edge) in metaEdges)
        {
            await _memory.UpsertTransitionAsync(new StateTransition
            {
                SourceId = edge.Src,
                TargetId = edge.Tgt,
                Action = 0,
                Reward = edge.TotalReward / edge.Count,
                TransitionCount = edge.Count,
                IsMacroEdge = true,
            });
        }
    }

    /// <summary>Compute centroid string from group of landmarks.</summary>
    private static string ComputeCentroid(IReadOnlyList<StateLandmark> group)
    {
        if (group.Count == 0) return "empty";
        // Use the landmark with highest value estimate as centroid
        return group.OrderByDescending(l => l.ValueEstimate).First().Id;
    }
}

/// <summary>Report of how many landmarks were created at each meta-level.</summary>
public sealed record MetaLevelReport
{
    public int Level1Count { get; set; }
    public int Level2Count { get; set; }
    public int Level3Count { get; set; }

    public override string ToString() =>
        $"Meta-levels: L1={Level1Count}, L2={Level2Count}, L3={Level3Count}";
}
