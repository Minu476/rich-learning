namespace RichLearning.Models;

/// <summary>
/// A lightweight subgoal directive for navigation between landmarks.
/// </summary>
public sealed record SubgoalDirective
{
    /// <summary>The target landmark ID to navigate towards.</summary>
    public required string TargetLandmarkId { get; init; }

    /// <summary>Reason for selecting this subgoal.</summary>
    public string Reason { get; init; } = string.Empty;

    /// <summary>Planned path of landmark IDs from current position to target.</summary>
    public IReadOnlyList<string> PlannedPath { get; init; } = [];
}
