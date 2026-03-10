using Microsoft.Extensions.Logging.Abstractions;
using RichLearning.Memory;
using RichLearning.Models;
using Xunit;

namespace RichLearning.Tests;

public class LiteDbGraphMemoryContractTests
{
    [Fact]
    public async Task LandmarkRoundTrip_PreservesTypedMetadataAndEpisodicTraces()
    {
        var dbPath = Path.Combine(Path.GetTempPath(), $"richlearning-{Guid.NewGuid():N}.db");

        try
        {
            await using var memory = new LiteDbGraphMemory(dbPath, NullLogger<LiteDbGraphMemory>.Instance);
            await memory.UpsertLandmarkAsync(new StateLandmark
            {
                Id = "landmark-1",
                Embedding = [0.1, 0.2],
                VisitCount = 7,
                ValueEstimate = 1.5,
                CreatedTimestep = 5,
                LastVisitedTimestep = 6,
                ActionCounts = new Dictionary<int, int> { [2] = 3 },
                EpisodicTraces =
                [
                    new EpisodicTrace
                    {
                        Steps = [(4, 1.25, "landmark-2")],
                        Return = 9.5,
                        RecordedTimestep = 12
                    }
                ],
                Metadata = new Dictionary<string, object>
                {
                    ["name"] = "typed",
                    ["count"] = 3L,
                    ["ratio"] = 0.75,
                    ["enabled"] = true,
                    ["tags"] = new List<object> { "a", 2L }
                }
            });

            var reread = await memory.GetLandmarkAsync("landmark-1");
            Assert.NotNull(reread);
            Assert.Equal(3L, reread!.Metadata["count"]);
            Assert.Equal(true, reread.Metadata["enabled"]);
            Assert.Equal(0.75, reread.Metadata["ratio"]);
            var tags = Assert.IsType<List<object>>(reread.Metadata["tags"]);
            Assert.Equal("a", tags[0]);
            Assert.Equal(2L, tags[1]);
            Assert.Single(reread.EpisodicTraces);
            Assert.Single(reread.EpisodicTraces[0].Steps);
            Assert.Equal(4, reread.EpisodicTraces[0].Steps[0].Action);
        }
        finally
        {
            if (File.Exists(dbPath))
                File.Delete(dbPath);
        }
    }

    [Fact]
    public async Task TransitionRoundTrip_PreservesActionCountsAndMacroFields()
    {
        var dbPath = Path.Combine(Path.GetTempPath(), $"richlearning-{Guid.NewGuid():N}.db");

        try
        {
            await using var memory = new LiteDbGraphMemory(dbPath, NullLogger<LiteDbGraphMemory>.Instance);
            await memory.UpsertLandmarkAsync(new StateLandmark { Id = "A", Embedding = [0.0], CreatedTimestep = 1, LastVisitedTimestep = 1 });
            await memory.UpsertLandmarkAsync(new StateLandmark { Id = "B", Embedding = [1.0], CreatedTimestep = 1, LastVisitedTimestep = 1 });

            await memory.UpsertTransitionAsync(new StateTransition
            {
                SourceId = "A",
                TargetId = "B",
                Action = 7,
                ActionCounts = new Dictionary<int, int> { [7] = 4, [8] = 1 },
                Reward = 2.5,
                RewardVariance = 0.125,
                TransitionCount = 9,
                SuccessRate = 0.8,
                Confidence = 0.9,
                TemporalDistance = 3,
                TdError = 0.2,
                LastTrainedTimestep = 15,
                IsMacroEdge = true,
                MacroPath = ["M1", "M2"]
            });

            var reread = await memory.GetOutgoingTransitionsAsync("A");
            Assert.Single(reread);
            Assert.Equal(4, reread[0].ActionCounts[7]);
            Assert.Equal(0.125, reread[0].RewardVariance, precision: 6);
            Assert.Equal(0.9, reread[0].Confidence, precision: 6);
            Assert.True(reread[0].IsMacroEdge);
            Assert.Equal(new[] { "M1", "M2" }, reread[0].MacroPath);
        }
        finally
        {
            if (File.Exists(dbPath))
                File.Delete(dbPath);
        }
    }
}