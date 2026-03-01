using RichLearning.Abstractions;
using RichLearning.Models;

namespace RichLearning.Planning;

/// <summary>
/// Builds hierarchical meta-level abstractions of the topological graph.
///
/// Meta-levels create progressively coarser views of the state space:
///   Level 0 → Level 1 → Level 2 → Level 3
///
/// <remarks>This is the public API stub. Full implementation is provided by
/// the rich-learning-base package.</remarks>
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
    /// </summary>
    /// <remarks>Full implementation in rich-learning-base.</remarks>
    public Task<MetaLevelReport> BuildAllLevelsAsync()
        => throw new NotImplementedException("See rich-learning-base for full implementation.");
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
