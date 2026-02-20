using Xunit;
using RichLearning.Memory;
using RichLearning.Models;

namespace RichLearning.Tests;

/// <summary>
/// Tests for the TrajectoryDag — the mathematical core of backward reinforcement.
/// Validates temporal discounting, Merkle integrity, and loop detection.
/// </summary>
public class TrajectoryDagTests
{
    // ── Temporal Discounting Math ──

    [Fact]
    public void BackwardReinforce_ThreeStepTrajectory_AppliesCorrectDiscounting()
    {
        // Arrange: A → B → C (3-step chain)
        var dag = new TrajectoryDag();
        var (nodeA, _) = dag.Append("hashA", "moveA", 1, 0.0, null);
        var (nodeB, _) = dag.Append("hashB", "moveB", 2, 0.0, nodeA.Id);
        var (nodeC, _) = dag.Append("hashC", "moveC", 3, 0.0, nodeB.Id);

        const double gamma = 0.95;
        const double terminalReward = 1.0;

        // Act: backward reinforce from terminal node C
        dag.BackwardReinforce(nodeC.Id, terminalReward, gamma);

        // Assert: exact DAPSA v2.1 Section 3.4 math
        // Q(C) += 1.0 * 0.95^0 = 1.0
        // Q(B) += 1.0 * 0.95^1 = 0.95
        // Q(A) += 1.0 * 0.95^2 = 0.9025
        Assert.Equal(1.0, nodeC.QValue, precision: 10);
        Assert.Equal(0.95, nodeB.QValue, precision: 10);
        Assert.Equal(0.9025, nodeA.QValue, precision: 10);
    }

    [Fact]
    public void BackwardReinforce_NegativeReward_PropagatesCorrectly()
    {
        var dag = new TrajectoryDag();
        var (nodeA, _) = dag.Append("hashA", "moveA", 1, 0.0, null);
        var (nodeB, _) = dag.Append("hashB", "moveB", 2, 0.0, nodeA.Id);

        dag.BackwardReinforce(nodeB.Id, -1.0, 0.95);

        Assert.Equal(-1.0, nodeB.QValue, precision: 10);
        Assert.Equal(-0.95, nodeA.QValue, precision: 10);
    }

    [Fact]
    public void BackwardReinforce_MultipleRewards_Accumulates()
    {
        // Two backward passes from different terminal nodes should accumulate
        var dag = new TrajectoryDag();
        var (nodeA, _) = dag.Append("hashA", "a", 1, 0.0, null);
        var (nodeB, _) = dag.Append("hashB", "b", 2, 0.0, nodeA.Id);
        var (nodeC, _) = dag.Append("hashC", "c", 3, 0.0, nodeA.Id); // branch from A

        dag.BackwardReinforce(nodeB.Id, 1.0, 0.95);
        dag.BackwardReinforce(nodeC.Id, 0.5, 0.95);

        // A should accumulate credit from both branches
        // From B: 1.0 * 0.95^1 = 0.95
        // From C: 0.5 * 0.95^1 = 0.475
        // Total:  0.95 + 0.475 = 1.425
        Assert.Equal(1.425, nodeA.QValue, precision: 10);
    }

    [Fact]
    public void BackwardReinforce_SingleNode_GetsFullReward()
    {
        var dag = new TrajectoryDag();
        var (nodeA, _) = dag.Append("hashA", "a", 1, 0.0, null);

        dag.BackwardReinforce(nodeA.Id, 2.5, 0.95);

        Assert.Equal(2.5, nodeA.QValue, precision: 10);
    }

    [Fact]
    public void BackwardReinforce_NonexistentNode_NoException()
    {
        var dag = new TrajectoryDag();
        // Should not throw
        dag.BackwardReinforce("nonexistent", 1.0, 0.95);
    }

    // ── Merkle Hash Integrity ──

    [Fact]
    public void MerkleHash_ChildIncorporatesParentHash()
    {
        var dag = new TrajectoryDag();
        var (nodeA, _) = dag.Append("hashA", "moveA", 1, 0.0, null);
        var (nodeB, _) = dag.Append("hashB", "moveB", 2, 0.0, nodeA.Id);

        // Child's ParentMerkleHash should be the parent's MerkleHash
        Assert.Equal(nodeA.MerkleHash, nodeB.ParentMerkleHash);
    }

    [Fact]
    public void MerkleHash_RootHasNullParentMerkle()
    {
        var dag = new TrajectoryDag();
        var (root, _) = dag.Append("hashA", "moveA", 1, 0.0, null);

        Assert.Null(root.ParentMerkleHash);
    }

    [Fact]
    public void MerkleHash_DifferentParents_ProduceDifferentChildHashes()
    {
        var dag = new TrajectoryDag();
        var (parentA, _) = dag.Append("hashA", "moveA", 1, 0.0, null);
        var (parentB, _) = dag.Append("hashB", "moveB", 1, 0.0, null);

        // Same child state but different parents → different Merkle hashes
        var (childOfA, _) = dag.Append("hashC", "moveC", 2, 0.0, parentA.Id);
        var (childOfB, _) = dag.Append("hashC", "moveC", 2, 0.0, parentB.Id);

        Assert.NotEqual(childOfA.MerkleHash, childOfB.MerkleHash);
    }

    [Fact]
    public void MerkleHash_SameInputs_ProduceSameHash()
    {
        // Two independent nodes with identical inputs but no parent
        // should produce the same Merkle hash (deterministic)
        var nodeA = new TrajectoryNode("hashX", "moveX", 5, 1.0, null, null);
        var nodeB = new TrajectoryNode("hashX", "moveX", 5, 1.0, null, null);

        Assert.Equal(nodeA.MerkleHash, nodeB.MerkleHash);
    }

    // ── Loop Detection ──

    [Fact]
    public void Append_RepeatedSituation_DetectsLoop()
    {
        var dag = new TrajectoryDag();
        var (nodeA, loopA) = dag.Append("hashA", "moveA", 1, 0.0, null);
        var (nodeB, loopB) = dag.Append("hashB", "moveB", 2, 0.0, nodeA.Id);
        var (nodeA2, loopA2) = dag.Append("hashA", "moveA", 3, 0.0, nodeB.Id); // revisit A

        Assert.False(loopA);
        Assert.False(loopB);
        Assert.True(loopA2);
    }

    [Fact]
    public void Append_UniqueStates_NoLoop()
    {
        var dag = new TrajectoryDag();
        var (_, loop1) = dag.Append("hash1", "a", 1, 0.0, null);
        var (n1, loop2) = dag.Append("hash2", "b", 2, 0.0, null);
        var (_, loop3) = dag.Append("hash3", "c", 3, 0.0, n1.Id);

        Assert.False(loop1);
        Assert.False(loop2);
        Assert.False(loop3);
    }

    // ── High-Value Node Extraction ──

    [Fact]
    public void GetHighValueNodes_FiltersByThreshold()
    {
        var dag = new TrajectoryDag();
        var (nodeA, _) = dag.Append("hashA", "a", 1, 0.9, null);
        var (nodeB, _) = dag.Append("hashB", "b", 2, 0.3, nodeA.Id);
        var (nodeC, _) = dag.Append("hashC", "c", 3, 0.6, nodeB.Id);

        var highValue = dag.GetHighValueNodes(0.5);

        Assert.Equal(2, highValue.Count);
        Assert.Contains(highValue, n => n.SituationHash == "hashA");
        Assert.Contains(highValue, n => n.SituationHash == "hashC");
        Assert.DoesNotContain(highValue, n => n.SituationHash == "hashB");
    }

    // ── Leaf Nodes ──

    [Fact]
    public void GetLeafNodes_ReturnsNodesWithNoChildren()
    {
        var dag = new TrajectoryDag();
        var (nodeA, _) = dag.Append("hashA", "a", 1, 0.0, null);
        var (nodeB, _) = dag.Append("hashB", "b", 2, 0.0, nodeA.Id);
        var (nodeC, _) = dag.Append("hashC", "c", 3, 0.0, nodeA.Id); // branch

        var leaves = dag.GetLeafNodes();

        Assert.Equal(2, leaves.Count);
        Assert.Contains(leaves, n => n.Id == nodeB.Id);
        Assert.Contains(leaves, n => n.Id == nodeC.Id);
        Assert.DoesNotContain(leaves, n => n.Id == nodeA.Id);
    }

    // ── Ancestor Chain ──

    [Fact]
    public void GetAncestorChain_ReturnsFullPath()
    {
        var dag = new TrajectoryDag();
        var (nodeA, _) = dag.Append("hashA", "a", 1, 0.0, null);
        var (nodeB, _) = dag.Append("hashB", "b", 2, 0.0, nodeA.Id);
        var (nodeC, _) = dag.Append("hashC", "c", 3, 0.0, nodeB.Id);

        var chain = dag.GetAncestorChain(nodeC.Id);

        Assert.Equal(3, chain.Count);
        Assert.Equal(nodeC.Id, chain[0].Id);
        Assert.Equal(nodeB.Id, chain[1].Id);
        Assert.Equal(nodeA.Id, chain[2].Id);
    }

    // ── Clear ──

    [Fact]
    public void Clear_ResetsEverything()
    {
        var dag = new TrajectoryDag();
        dag.Append("hashA", "a", 1, 0.0, null);
        dag.Append("hashB", "b", 2, 0.0, null);


        dag.Clear();

        Assert.Equal(0, dag.NodeCount);
        Assert.Empty(dag.AllNodes);

        // After clear, same hash should not detect loop
        var (_, isLoop) = dag.Append("hashA", "a", 1, 0.0, null);
        Assert.False(isLoop);
    }
}
