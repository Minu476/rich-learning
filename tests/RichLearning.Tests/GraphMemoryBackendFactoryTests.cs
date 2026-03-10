using Microsoft.Extensions.Logging;
using RichLearning.Abstractions;
using RichLearning.Memory;
using RichLearning.Models;
using Xunit;

namespace RichLearning.Tests;

public class GraphMemoryBackendFactoryTests
{
    [Fact]
    public void FromArgs_DefaultsToLiteDb()
    {
        var options = GraphMemoryBackendFactory.FromArgs([], defaultLiteDbPath: "/tmp/demo.db");

        Assert.Equal("litedb", options.Kind);
        Assert.Equal("/tmp/demo.db", options.LiteDbPath);
    }

    [Fact]
    public async Task CreateAsync_CustomBackend_LoadsTypeFromAssembly()
    {
        using var loggerFactory = LoggerFactory.Create(builder => { });
        var options = new GraphMemoryBackendOptions
        {
            Kind = "custom",
            CustomAssemblyPath = typeof(TestCustomGraphMemory).Assembly.Location,
            CustomTypeName = typeof(TestCustomGraphMemory).FullName,
            CustomSettings = new Dictionary<string, string> { ["url"] = "ws://localhost:8000/rpc" }
        };

        var backend = await GraphMemoryBackendFactory.CreateAsync(options, loggerFactory);

        var custom = Assert.IsType<TestCustomGraphMemory>(backend);
        Assert.Equal("ws://localhost:8000/rpc", custom.Settings["url"]);
    }

    public sealed class TestCustomGraphMemory : IGraphMemory
    {
        public TestCustomGraphMemory(IReadOnlyDictionary<string, string> settings)
        {
            Settings = new Dictionary<string, string>(settings);
        }

        public IReadOnlyDictionary<string, string> Settings { get; }

        public Task InitialiseSchemaAsync() => Task.CompletedTask;
        public Task UpsertLandmarkAsync(StateLandmark landmark) => Task.CompletedTask;
        public Task<StateLandmark?> GetLandmarkAsync(string id) => Task.FromResult<StateLandmark?>(null);
        public Task<IReadOnlyList<StateLandmark>> GetAllLandmarksAsync(int? hierarchyLevel = null) => Task.FromResult<IReadOnlyList<StateLandmark>>([]);
        public Task<(StateLandmark Landmark, double Distance)?> NearestNeighbourAsync(double[] embedding, IStateEncoder encoder) => Task.FromResult<(StateLandmark Landmark, double Distance)?>(null);
        public Task UpsertTransitionAsync(StateTransition transition) => Task.CompletedTask;
        public Task<IReadOnlyList<StateTransition>> GetOutgoingTransitionsAsync(string landmarkId) => Task.FromResult<IReadOnlyList<StateTransition>>([]);
        public Task<IReadOnlyList<string>> ShortestPathAsync(string fromId, string toId) => Task.FromResult<IReadOnlyList<string>>([]);
        public Task<IReadOnlyList<string>> DetectCycleInTrajectoryAsync(IReadOnlyList<string> recentIds) => Task.FromResult<IReadOnlyList<string>>([]);
        public Task<IReadOnlyList<StateLandmark>> GetFrontierLandmarksAsync(int topK = 5) => Task.FromResult<IReadOnlyList<StateLandmark>>([]);
        public Task<IReadOnlyList<StateTransition>> PrioritisedSampleAsync(int batchSize, long currentTimestep) => Task.FromResult<IReadOnlyList<StateTransition>>([]);
        public Task AssignClustersAsync(int rounds = 5) => Task.CompletedTask;
        public Task<(int Landmarks, int Transitions)> GetGraphStatsAsync() => Task.FromResult((0, 0));
        public Task<bool> RemoveLandmarkAsync(string id) => Task.FromResult(false);
        public Task<bool> RemoveTransitionAsync(string sourceId, string targetId, int action) => Task.FromResult(false);
        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }
}