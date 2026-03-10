using System.Text.Json;
using System.Text.Json.Nodes;
using RichLearning.Models;

namespace RichLearning.Memory;

internal static class GraphMemorySerialization
{
    private static readonly JsonSerializerOptions s_json = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };

    public static string SerializeMetadata(IReadOnlyDictionary<string, object> metadata) =>
        JsonSerializer.Serialize(metadata, s_json);

    public static Dictionary<string, object> DeserializeMetadata(string? json)
    {
        if (string.IsNullOrWhiteSpace(json))
            return new Dictionary<string, object>();

        var node = JsonNode.Parse(json) as JsonObject;
        if (node is null)
            return new Dictionary<string, object>();

        return node.ToDictionary(
            pair => pair.Key,
            pair => ConvertJsonNode(pair.Value));
    }

    public static string SerializeActionCounts(IReadOnlyDictionary<int, int> actionCounts) =>
        JsonSerializer.Serialize(actionCounts, s_json);

    public static Dictionary<int, int> DeserializeActionCounts(string? json)
    {
        if (string.IsNullOrWhiteSpace(json))
            return new Dictionary<int, int>();

        return JsonSerializer.Deserialize<Dictionary<int, int>>(json, s_json)
            ?? new Dictionary<int, int>();
    }

    public static string SerializeEpisodicTraces(IReadOnlyList<EpisodicTrace> traces)
    {
        var docs = traces.Select(trace => new EpisodicTraceDoc
        {
            Return = trace.Return,
            RecordedTimestep = trace.RecordedTimestep,
            Steps = trace.Steps.Select(step => new EpisodicTraceStepDoc
            {
                Action = step.Action,
                Reward = step.Reward,
                NextLandmarkId = step.NextLandmarkId
            }).ToList()
        }).ToList();

        return JsonSerializer.Serialize(docs, s_json);
    }

    public static List<EpisodicTrace> DeserializeEpisodicTraces(string? json)
    {
        if (string.IsNullOrWhiteSpace(json))
            return new List<EpisodicTrace>();

        var docs = JsonSerializer.Deserialize<List<EpisodicTraceDoc>>(json, s_json)
            ?? new List<EpisodicTraceDoc>();

        return docs.Select(doc => new EpisodicTrace
        {
            Return = doc.Return,
            RecordedTimestep = doc.RecordedTimestep,
            Steps = doc.Steps.Select(step =>
                (step.Action, step.Reward, step.NextLandmarkId)).ToList()
        }).ToList();
    }

    public static string SerializeMacroPath(IReadOnlyList<string> macroPath) =>
        JsonSerializer.Serialize(macroPath, s_json);

    public static List<string> DeserializeMacroPath(string? json)
    {
        if (string.IsNullOrWhiteSpace(json))
            return new List<string>();

        return JsonSerializer.Deserialize<List<string>>(json, s_json)
            ?? new List<string>();
    }

    public static StateLandmark CloneLandmark(StateLandmark landmark) => new()
    {
        Id = landmark.Id,
        Embedding = landmark.Embedding.ToArray(),
        VisitCount = landmark.VisitCount,
        ValueEstimate = landmark.ValueEstimate,
        NoveltyScore = landmark.NoveltyScore,
        UncertaintyScore = landmark.UncertaintyScore,
        ClusterId = landmark.ClusterId,
        HierarchyLevel = landmark.HierarchyLevel,
        ChildNodeIds = landmark.ChildNodeIds.ToList(),
        LastVisitedTimestep = landmark.LastVisitedTimestep,
        CreatedTimestep = landmark.CreatedTimestep,
        ActionCounts = new Dictionary<int, int>(landmark.ActionCounts),
        EpisodicTraces = landmark.EpisodicTraces.Select(trace => new EpisodicTrace
        {
            Return = trace.Return,
            RecordedTimestep = trace.RecordedTimestep,
            Steps = trace.Steps.Select(step =>
                (step.Action, step.Reward, step.NextLandmarkId)).ToList()
        }).ToList(),
        Metadata = DeserializeMetadata(SerializeMetadata(landmark.Metadata))
    };

    public static StateTransition CloneTransition(StateTransition transition) => new()
    {
        SourceId = transition.SourceId,
        TargetId = transition.TargetId,
        Action = transition.Action,
        ActionCounts = new Dictionary<int, int>(transition.ActionCounts),
        Reward = transition.Reward,
        RewardVariance = transition.RewardVariance,
        TransitionCount = transition.TransitionCount,
        SuccessRate = transition.SuccessRate,
        Confidence = transition.Confidence,
        TemporalDistance = transition.TemporalDistance,
        TdError = transition.TdError,
        LastTrainedTimestep = transition.LastTrainedTimestep,
        IsMacroEdge = transition.IsMacroEdge,
        MacroPath = transition.MacroPath.ToList()
    };

    private static object ConvertJsonNode(JsonNode? node)
    {
        if (node is null)
            return null!;

        return node switch
        {
            JsonValue value => ConvertJsonValue(value),
            JsonArray array => array.Select(ConvertJsonNode).ToList(),
            JsonObject obj => obj.ToDictionary(pair => pair.Key, pair => ConvertJsonNode(pair.Value)),
            _ => node.ToJsonString()
        };
    }

    private static object ConvertJsonValue(JsonValue value)
    {
        if (value.TryGetValue<bool>(out var boolValue))
            return boolValue;
        if (value.TryGetValue<long>(out var longValue))
            return longValue;
        if (value.TryGetValue<double>(out var doubleValue))
            return doubleValue;
        if (value.TryGetValue<string>(out var stringValue))
            return stringValue ?? string.Empty;

        return value.ToJsonString();
    }

    private sealed class EpisodicTraceDoc
    {
        public List<EpisodicTraceStepDoc> Steps { get; set; } = new();
        public double Return { get; set; }
        public long RecordedTimestep { get; set; }
    }

    private sealed class EpisodicTraceStepDoc
    {
        public int Action { get; set; }
        public double Reward { get; set; }
        public string NextLandmarkId { get; set; } = string.Empty;
    }
}