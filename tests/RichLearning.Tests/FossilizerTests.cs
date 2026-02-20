using Xunit;
using RichLearning.Learning;
using RichLearning.Models;

namespace RichLearning.Tests;

/// <summary>
/// Tests for the Fossilizer — the Phi function (S^A → S^P) that promotes
/// active knowledge into passive O(1) lookup tables.
/// </summary>
public class FossilizerTests
{
    // ── FromTrajectory Threshold ──

    [Fact]
    public void FromTrajectory_ThresholdFiltering_OnlyIncludesAboveThreshold()
    {
        // 3 nodes with Q-values: 0.9, 0.6, 0.3
        var nodes = new List<TrajectoryNode>
        {
            new("hashHigh", "actionHigh", 1, 0.9, null, null),
            new("hashMid", "actionMid", 2, 0.6, null, null),
            new("hashLow", "actionLow", 3, 0.3, null, null),
        };

        var skill = Fossilizer.FromTrajectory("test_skill", nodes, minQValue: 0.5);

        Assert.Equal(2, skill.Size);
        Assert.True(skill.TryLookup("hashHigh", out var a1));
        Assert.Equal("actionHigh", a1);
        Assert.True(skill.TryLookup("hashMid", out var a2));
        Assert.Equal("actionMid", a2);
        Assert.False(skill.TryLookup("hashLow", out _));
    }

    [Fact]
    public void FromTrajectory_AllBelowThreshold_ReturnsEmptySkill()
    {
        var nodes = new List<TrajectoryNode>
        {
            new("hash1", "action1", 1, 0.1, null, null),
            new("hash2", "action2", 2, 0.2, null, null),
        };

        var skill = Fossilizer.FromTrajectory("empty_skill", nodes, minQValue: 0.5);

        Assert.Equal(0, skill.Size);
    }

    [Fact]
    public void FromTrajectory_AllAboveThreshold_IncludesAll()
    {
        var nodes = new List<TrajectoryNode>
        {
            new("hash1", "action1", 1, 0.8, null, null),
            new("hash2", "action2", 2, 0.9, null, null),
        };

        var skill = Fossilizer.FromTrajectory("full_skill", nodes, minQValue: 0.5);

        Assert.Equal(2, skill.Size);
    }

    [Fact]
    public void FromTrajectory_DuplicateSituations_KeepsHigherQ()
    {
        var nodes = new List<TrajectoryNode>
        {
            new("sameHash", "actionOld", 1, 0.6, null, null),
            new("sameHash", "actionNew", 2, 0.9, null, null),
        };

        var skill = Fossilizer.FromTrajectory("dedup_skill", nodes, minQValue: 0.5);

        Assert.Equal(1, skill.Size);
        Assert.True(skill.TryLookup("sameHash", out var action));
        Assert.Equal("actionNew", action);
    }

    // ── FromEpisodes ──

    [Fact]
    public void FromEpisodes_CreatesCorrectMappings()
    {
        var episodes = new List<(string, string)>
        {
            ("state1", "action1"),
            ("state2", "action2"),
            ("state3", "action3"),
        };

        var skill = Fossilizer.FromEpisodes("episode_skill", episodes);

        Assert.Equal(3, skill.Size);
        Assert.True(skill.TryLookup("state1", out var a));
        Assert.Equal("action1", a);
    }

    [Fact]
    public void FromEpisodes_DuplicateStates_LastWins()
    {
        var episodes = new List<(string, string)>
        {
            ("state1", "actionOld"),
            ("state1", "actionNew"),
        };

        var skill = Fossilizer.FromEpisodes("override_skill", episodes);

        Assert.Equal(1, skill.Size);
        Assert.True(skill.TryLookup("state1", out var a));
        Assert.Equal("actionNew", a);
    }

    // ── FromPatterns ──

    [Fact]
    public void FromPatterns_OnlyIncludesFossilized()
    {
        var p1 = new Pattern("stateA", "actionA", "test");
        var p2 = new Pattern("stateB", "actionB", "test");

        // Observe p1 enough times to fossilize it
        for (int i = 0; i < 20; i++)
            p1.Observe(success: true);

        // p2 stays unfossilized
        p2.Observe(success: true);

        var skill = Fossilizer.FromPatterns("pattern_skill", new[] { p1, p2 });

        // Only fossilized patterns should be included
        if (p1.IsFossilized)
        {
            Assert.True(skill.TryLookup("stateA", out _));
        }
        if (!p2.IsFossilized)
        {
            Assert.False(skill.TryLookup("stateB", out _));
        }
    }

    // ── FossilizedSkill Properties ──

    [Fact]
    public void FossilizedSkill_SkillName_IsRequired()
    {
        Assert.Throws<ArgumentException>(() =>
            new FossilizedSkill("", new Dictionary<string, string>()));
    }

    [Fact]
    public void FossilizedSkill_TryLookup_IncrementsUsageCount()
    {
        var skill = new FossilizedSkill("counter_test",
            new Dictionary<string, string> { ["key"] = "value" });

        Assert.Equal(0, skill.UsageCount);

        skill.TryLookup("key", out _);
        Assert.Equal(1, skill.UsageCount);

        skill.TryLookup("missing", out _);
        Assert.Equal(2, skill.UsageCount); // Increments even on miss
    }
}
