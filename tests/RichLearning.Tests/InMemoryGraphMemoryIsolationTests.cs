using RichLearning.Memory;
using RichLearning.Models;
using Xunit;

namespace RichLearning.Tests;

public class InMemoryGraphMemoryIsolationTests
{
    [Fact]
    public async Task GetLandmarkAsync_ReturnsDefensiveCopy()
    {
        await using var memory = new InMemoryGraphMemory();
        await memory.UpsertLandmarkAsync(new StateLandmark
        {
            Id = "A",
            Embedding = [1.0, 2.0],
            VisitCount = 3,
            CreatedTimestep = 10,
            LastVisitedTimestep = 11,
            Metadata = new Dictionary<string, object> { ["count"] = 5L }
        });

        var fetched = await memory.GetLandmarkAsync("A");
        Assert.NotNull(fetched);

        fetched!.Embedding[0] = 99.0;
        fetched.Metadata["count"] = 42L;

        var reread = await memory.GetLandmarkAsync("A");
        Assert.NotNull(reread);
        Assert.Equal(1.0, reread!.Embedding[0]);
        Assert.Equal(5L, reread.Metadata["count"]);
    }

    [Fact]
    public async Task GetOutgoingTransitionsAsync_ReturnsDefensiveCopies()
    {
        await using var memory = new InMemoryGraphMemory();
        await memory.UpsertLandmarkAsync(new StateLandmark
        {
            Id = "A",
            Embedding = [0.0],
            CreatedTimestep = 1,
            LastVisitedTimestep = 1
        });
        await memory.UpsertLandmarkAsync(new StateLandmark
        {
            Id = "B",
            Embedding = [1.0],
            CreatedTimestep = 1,
            LastVisitedTimestep = 1
        });
        await memory.UpsertTransitionAsync(new StateTransition
        {
            SourceId = "A",
            TargetId = "B",
            Action = 1,
            Reward = 0.5,
            ActionCounts = new Dictionary<int, int> { [1] = 2 },
            MacroPath = ["mid"]
        });

        var fetched = await memory.GetOutgoingTransitionsAsync("A");
        fetched[0].ActionCounts[1] = 100;

        var reread = await memory.GetOutgoingTransitionsAsync("A");
        Assert.Equal(2, reread[0].ActionCounts[1]);
        Assert.Equal("mid", reread[0].MacroPath[0]);
    }

    [Fact]
    public async Task RemoveLandmarkAsync_RemovesIncomingAndOutgoingEdges()
    {
        await using var memory = new InMemoryGraphMemory();

        foreach (var id in new[] { "A", "B", "C" })
        {
            await memory.UpsertLandmarkAsync(new StateLandmark
            {
                Id = id,
                Embedding = [0.0],
                CreatedTimestep = 1,
                LastVisitedTimestep = 1
            });
        }

        await memory.UpsertTransitionAsync(new StateTransition { SourceId = "A", TargetId = "B", Action = 0, Reward = 1.0 });
        await memory.UpsertTransitionAsync(new StateTransition { SourceId = "B", TargetId = "C", Action = 0, Reward = 1.0 });
        await memory.UpsertTransitionAsync(new StateTransition { SourceId = "C", TargetId = "B", Action = 1, Reward = 1.0 });

        var removed = await memory.RemoveLandmarkAsync("B");

        Assert.True(removed);
        Assert.Empty(await memory.GetOutgoingTransitionsAsync("A"));
        Assert.Empty(await memory.GetOutgoingTransitionsAsync("C"));
        var stats = await memory.GetGraphStatsAsync();
        Assert.Equal(2, stats.Landmarks);
        Assert.Equal(0, stats.Transitions);
    }
}