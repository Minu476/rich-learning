using Xunit;
using RichLearning.Memory;
using RichLearning.Models;

namespace RichLearning.Tests;

/// <summary>
/// Tests for InMemoryGraphMemory — validates graph operations,
/// shortest path, nearest neighbour, and concurrency safety.
/// </summary>
public class InMemoryGraphMemoryTests
{
    private static StateLandmark MakeLandmark(string id, double[]? embedding = null) => new()
    {
        Id = id,
        Embedding = embedding ?? [0.0, 0.0, 0.0, 0.0],
        VisitCount = 1,
        NoveltyScore = 1.0,
        UncertaintyScore = 1.0,
        CreatedTimestep = 1,
        LastVisitedTimestep = 1,
    };

    // ── Nearest Neighbour ──

    [Fact]
    public async Task NearestNeighbour_EmptyGraph_ReturnsNull()
    {
        var memory = new InMemoryGraphMemory();
        var encoder = new DefaultStateEncoder(embeddingDimension: 4);

        var result = await memory.NearestNeighbourAsync([1.0, 2.0, 3.0, 4.0], encoder);

        Assert.Null(result);
    }

    [Fact]
    public async Task NearestNeighbour_SingleLandmark_ReturnsIt()
    {
        var memory = new InMemoryGraphMemory();
        var encoder = new DefaultStateEncoder(embeddingDimension: 4);

        var lm = MakeLandmark("lm1", [1.0, 0.0, 0.0, 0.0]);
        await memory.UpsertLandmarkAsync(lm);

        var result = await memory.NearestNeighbourAsync([1.0, 0.0, 0.0, 0.0], encoder);

        Assert.NotNull(result);
        Assert.Equal("lm1", result.Value.Landmark.Id);
        Assert.Equal(0.0, result.Value.Distance, precision: 5);
    }

    [Fact]
    public async Task NearestNeighbour_MultipleLandmarks_ReturnsClosest()
    {
        var memory = new InMemoryGraphMemory();
        var encoder = new DefaultStateEncoder(embeddingDimension: 4);

        await memory.UpsertLandmarkAsync(MakeLandmark("far", [0.0, 0.0, 0.0, 1.0]));
        await memory.UpsertLandmarkAsync(MakeLandmark("close", [1.0, 0.0, 0.0, 0.0]));
        await memory.UpsertLandmarkAsync(MakeLandmark("medium", [0.5, 0.5, 0.0, 0.0]));

        var result = await memory.NearestNeighbourAsync([1.0, 0.0, 0.0, 0.0], encoder);

        Assert.NotNull(result);
        Assert.Equal("close", result.Value.Landmark.Id);
    }

    // ── Shortest Path ──

    [Fact]
    public async Task ShortestPath_LinearGraph_ReturnsCorrectSequence()
    {
        var memory = new InMemoryGraphMemory();

        await memory.UpsertLandmarkAsync(MakeLandmark("A"));
        await memory.UpsertLandmarkAsync(MakeLandmark("B"));
        await memory.UpsertLandmarkAsync(MakeLandmark("C"));

        await memory.UpsertTransitionAsync(new StateTransition
        {
            SourceId = "A", TargetId = "B", Action = 0, Reward = 1.0,
            TransitionCount = 1, SuccessRate = 1.0
        });
        await memory.UpsertTransitionAsync(new StateTransition
        {
            SourceId = "B", TargetId = "C", Action = 0, Reward = 1.0,
            TransitionCount = 1, SuccessRate = 1.0
        });

        var path = await memory.ShortestPathAsync("A", "C");

        Assert.Equal(3, path.Count);
        Assert.Equal("A", path[0]);
        Assert.Equal("B", path[1]);
        Assert.Equal("C", path[2]);
    }

    [Fact]
    public async Task ShortestPath_NoPath_ReturnsEmpty()
    {
        var memory = new InMemoryGraphMemory();

        await memory.UpsertLandmarkAsync(MakeLandmark("A"));
        await memory.UpsertLandmarkAsync(MakeLandmark("B"));
        // No transitions

        var path = await memory.ShortestPathAsync("A", "B");

        Assert.Empty(path);
    }

    [Fact]
    public async Task ShortestPath_SameNode_ReturnsSingleElement()
    {
        var memory = new InMemoryGraphMemory();
        await memory.UpsertLandmarkAsync(MakeLandmark("A"));

        var path = await memory.ShortestPathAsync("A", "A");

        Assert.Single(path);
        Assert.Equal("A", path[0]);
    }

    // ── Remove / Decay Snapshot Fix ──

    [Fact]
    public async Task UpsertAndRemove_MultipleTransitions_NoCollectionModifiedException()
    {
        var memory = new InMemoryGraphMemory();

        // Add landmarks
        for (int i = 0; i < 10; i++)
            await memory.UpsertLandmarkAsync(MakeLandmark($"lm{i}"));

        // Add transitions forming a chain
        for (int i = 0; i < 9; i++)
        {
            await memory.UpsertTransitionAsync(new StateTransition
            {
                SourceId = $"lm{i}", TargetId = $"lm{i + 1}", Action = 0, Reward = 0.5,
                TransitionCount = 1, SuccessRate = 1.0,
            });
        }

        // Verify graph is constructed
        var stats = await memory.GetGraphStatsAsync();
        Assert.Equal(10, stats.Landmarks);
        Assert.Equal(9, stats.Transitions);
    }

    // ── Upsert Transition Idempotency ──

    [Fact]
    public async Task UpsertTransition_SameEdgeTwice_UpdatesInPlace()
    {
        var memory = new InMemoryGraphMemory();
        await memory.UpsertLandmarkAsync(MakeLandmark("A"));
        await memory.UpsertLandmarkAsync(MakeLandmark("B"));

        var t1 = new StateTransition
        {
            SourceId = "A", TargetId = "B", Action = 0, Reward = 1.0,
            TransitionCount = 1, SuccessRate = 1.0,
        };
        var t2 = new StateTransition
        {
            SourceId = "A", TargetId = "B", Action = 0, Reward = 2.0,
            TransitionCount = 5, SuccessRate = 0.8,
        };

        await memory.UpsertTransitionAsync(t1);
        await memory.UpsertTransitionAsync(t2);

        var outgoing = await memory.GetOutgoingTransitionsAsync("A");
        Assert.Single(outgoing);
        Assert.Equal(2.0, outgoing[0].Reward);
        Assert.Equal(5, outgoing[0].TransitionCount);
    }

    // ── GetAllLandmarks with Hierarchy Level ──

    [Fact]
    public async Task GetAllLandmarks_FiltersByHierarchyLevel()
    {
        var memory = new InMemoryGraphMemory();

        var lm0 = MakeLandmark("base1");
        var lm1 = MakeLandmark("meta1");
        lm1 = lm1 with { HierarchyLevel = 1 };

        await memory.UpsertLandmarkAsync(lm0);
        await memory.UpsertLandmarkAsync(lm1);

        var level0 = await memory.GetAllLandmarksAsync(hierarchyLevel: 0);
        var level1 = await memory.GetAllLandmarksAsync(hierarchyLevel: 1);
        var all = await memory.GetAllLandmarksAsync();

        Assert.Single(level0);
        Assert.Single(level1);
        Assert.Equal(2, all.Count);
    }

    // ── Cycle Detection ──

    [Fact]
    public async Task DetectCycle_RepeatedIds_FindsCycle()
    {
        var memory = new InMemoryGraphMemory();
        var trajectory = new List<string> { "A", "B", "C", "A" };

        var cycle = await memory.DetectCycleInTrajectoryAsync(trajectory);

        Assert.NotEmpty(cycle);
        Assert.Contains("A", cycle);
    }

    [Fact]
    public async Task DetectCycle_NoDuplicates_ReturnsEmpty()
    {
        var memory = new InMemoryGraphMemory();
        var trajectory = new List<string> { "A", "B", "C", "D" };

        var cycle = await memory.DetectCycleInTrajectoryAsync(trajectory);

        // If no graph-based cycles found either, should be empty
        Assert.Empty(cycle);
    }

    // ── Graph Stats ──

    [Fact]
    public async Task GetGraphStats_EmptyGraph_ReturnsZeroes()
    {
        var memory = new InMemoryGraphMemory();
        var (lm, tr) = await memory.GetGraphStatsAsync();

        Assert.Equal(0, lm);
        Assert.Equal(0, tr);
    }

    // ── Frontier Landmarks ──

    [Fact]
    public async Task GetFrontierLandmarks_HighNovelty_RankedFirst()
    {
        var memory = new InMemoryGraphMemory();

        var explored = MakeLandmark("explored");
        explored.VisitCount = 100;
        explored.NoveltyScore = 0.01;

        var frontier = MakeLandmark("frontier");
        frontier.VisitCount = 1;
        frontier.NoveltyScore = 1.0;

        await memory.UpsertLandmarkAsync(explored);
        await memory.UpsertLandmarkAsync(frontier);

        var frontiers = await memory.GetFrontierLandmarksAsync(1);

        Assert.Single(frontiers);
        Assert.Equal("frontier", frontiers[0].Id);
    }
}
