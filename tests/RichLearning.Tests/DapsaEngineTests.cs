using Xunit;
using RichLearning.Abstractions;
using RichLearning.Encoders;
using RichLearning.Engine;
using RichLearning.Memory;
using RichLearning.Models;

namespace RichLearning.Tests;

/// <summary>
/// Tests for the DapsaEngine — the central DAPSA orchestrator.
///
/// NOTE: In the public repository, the engine methods (QueryAsync, EndEpisodeAsync,
/// NewSession) are stubs that throw NotImplementedException. These tests validate
/// the public API surface and stub contracts. Full behavioural tests live in
/// the rich-learning-base private repository.
/// </summary>
public class DapsaEngineTests
{
    private static DapsaEngine<double[], string> CreateEngine(
        IConsonanceChecker<double[]>? consonanceChecker = null)
    {
        var memory = new InMemoryGraphMemory();
        var encoder = new DefaultStateEncoder(dimension: 4);
        return new DapsaEngine<double[], string>(
            memory, encoder, consonanceChecker);
    }

    // ── Construction & Configuration ──

    [Fact]
    public void Constructor_SetsDefaults()
    {
        var engine = CreateEngine();
        Assert.Equal(0.95, engine.DiscountFactor);
        Assert.Equal(0.5, engine.FossilizationQThreshold);
        Assert.Equal(0.15, engine.FossilLookupRadius);
    }

    [Fact]
    public void Constructor_AcceptsCustomDiscountFactor()
    {
        var memory = new InMemoryGraphMemory();
        var encoder = new DefaultStateEncoder(dimension: 4);
        var engine = new DapsaEngine<double[], string>(
            memory, encoder, discountFactor: 0.99);
        Assert.Equal(0.99, engine.DiscountFactor);
    }

    [Fact]
    public void Constructor_AcceptsConsonanceChecker()
    {
        var checker = new AlwaysConsonantChecker();
        var engine = CreateEngine(consonanceChecker: checker);
        // Should construct without exception
        Assert.NotNull(engine);
    }

    // ── Stub Methods Throw NotImplementedException ──

    [Fact]
    public async Task QueryAsync_IsStub_ThrowsNotImplementedException()
    {
        var engine = CreateEngine();
        var state = new double[] { 1.0, 2.0, 3.0, 4.0 };

        await Assert.ThrowsAsync<NotImplementedException>(
            () => engine.QueryAsync(state, state,
                async obs => ("answer", 0.9, 5)));
    }

    [Fact]
    public async Task EndEpisodeAsync_IsStub_ThrowsNotImplementedException()
    {
        var engine = CreateEngine();

        await Assert.ThrowsAsync<NotImplementedException>(
            () => engine.EndEpisodeAsync(1.0));
    }

    [Fact]
    public void NewSession_IsStub_ThrowsNotImplementedException()
    {
        var engine = CreateEngine();

        Assert.Throws<NotImplementedException>(() => engine.NewSession());
    }

    // ── Diagnostics ──

    [Fact]
    public void PassiveHitRate_StartsAtZero()
    {
        var engine = CreateEngine();
        Assert.Equal(0.0, engine.PassiveHitRate);
    }

    [Fact]
    public void FossilCount_StartsAtZero()
    {
        var engine = CreateEngine();
        Assert.Equal(0, engine.FossilCount);
    }

    [Fact]
    public void HierarchyLevels_StartsAtZero()
    {
        var engine = CreateEngine();
        Assert.Equal(0, engine.HierarchyLevels);
    }

    [Fact]
    public void PatternCount_StartsAtZero()
    {
        var engine = CreateEngine();
        Assert.Equal(0, engine.PatternCount);
    }

    [Fact]
    public void Cartographer_IsAccessible()
    {
        var engine = CreateEngine();
        Assert.NotNull(engine.Cartographer);
    }

    [Fact]
    public void MetaHierarchy_IsAccessible()
    {
        var engine = CreateEngine();
        Assert.NotNull(engine.MetaHierarchy);
    }

    [Fact]
    public void EfficiencyTracker_IsAccessible()
    {
        var engine = CreateEngine();
        Assert.NotNull(engine.EfficiencyTracker);
    }

    // ── Helper Implementations ──

    private class AlwaysDissonantChecker : IConsonanceChecker<double[]>
    {
        public ConsonanceResult Check(double[] observation, string? predictedAction = null)
            => new(Error: 0.99, IsDissonant: true, Reason: "Always dissonant for testing");
    }

    private class AlwaysConsonantChecker : IConsonanceChecker<double[]>
    {
        public ConsonanceResult Check(double[] observation, string? predictedAction = null)
            => new(Error: 0.01, IsDissonant: false, Reason: "Always consonant for testing");
    }
}
