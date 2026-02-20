using RichLearning.Models;

namespace RichLearning.Memory;

/// <summary>
/// Trajectory DAG: the Active Manifold (S^A) data structure for tracking
/// causal chains of state evaluations.
///
/// This is a domain-agnostic implementation of the Merkle-linked trajectory
/// from DAPSA v2.1.
///
/// Core operations:
///   - Append: add a new node with loop detection
///   - BackwardReinforce: propagate reward from terminal node to root
///     Q(v_t) += reward * gamma^distance  (DAPSA v2.1 Section 3.4)
///   - GetHighValueNodes: extract fossilization candidates
///   - GetAncestorChain: traverse back to root for analysis
///
/// Not thread-safe. Each session/episode should use its own instance.
/// </summary>
public sealed class TrajectoryDag
{
    private readonly Dictionary<string, TrajectoryNode> _nodes = new();
    private readonly Dictionary<string, List<string>> _children = new();
    private readonly HashSet<string> _visitedSituations = new();

    /// <summary>
    /// Append a new node to the trajectory.
    ///
    /// Returns (node, isLoop) where isLoop=true if this situation was already
    /// visited in this trajectory (analogous to threefold repetition in chess,
    /// circular references in LLM reasoning, etc.).
    /// </summary>
    public (TrajectoryNode Node, bool IsLoop) Append(
        string situationHash,
        string action,
        int depth,
        double qValue,
        string? parentId)
    {
        string? parentMerkle = null;
        if (parentId is not null && _nodes.TryGetValue(parentId, out var parent))
        {
            parentMerkle = parent.MerkleHash;

            if (!_children.TryGetValue(parentId, out var childList))
            {
                childList = [];
                _children[parentId] = childList;
            }
        }

        bool isLoop = !_visitedSituations.Add(situationHash);

        var node = new TrajectoryNode(
            situationHash, action, depth, qValue, parentId, parentMerkle);

        _nodes[node.Id] = node;

        if (parentId is not null && _children.TryGetValue(parentId, out var children))
        {
            children.Add(node.Id);
        }

        return (node, isLoop);
    }

    /// <summary>
    /// Backward reinforcement: propagate a reward signal from a terminal node
    /// up through all ancestors.
    ///
    /// From DAPSA v2.1:
    ///   Q(v_t) += reward * gamma^distance
    ///
    /// Where:
    ///   reward = R_T (terminal reward: domain-specific, e.g., +1.0 win, -1.0 loss)
    ///   gamma = temporal discount factor (default 0.95)
    ///   distance = number of steps from terminal node to ancestor
    ///
    /// This is the core learning signal in Rich Learning. It assigns credit to
    /// all decisions in the causal chain, weighted by their temporal proximity
    /// to the outcome.
    /// </summary>
    public void BackwardReinforce(string terminalNodeId, double reward, double discount = 0.95)
    {
        if (!_nodes.TryGetValue(terminalNodeId, out var current))
            return;

        int distance = 0;
        while (current is not null)
        {
            current.QValue += reward * Math.Pow(discount, distance);
            distance++;

            if (current.ParentId is not null && _nodes.TryGetValue(current.ParentId, out var parentNode))
                current = parentNode;
            else
                current = null;
        }
    }

    /// <summary>
    /// Extract high-value nodes: trajectory nodes with Q-value above a threshold.
    /// These are candidates for fossilization promotion and meta-hierarchy teaching.
    /// </summary>
    public IReadOnlyList<TrajectoryNode> GetHighValueNodes(double minQValue)
    {
        return _nodes.Values.Where(n => n.QValue >= minQValue).ToList();
    }

    /// <summary>
    /// Get the ancestor chain from a node back to the root.
    /// Used for analysis, debugging, and explainability.
    /// </summary>
    public IReadOnlyList<TrajectoryNode> GetAncestorChain(string nodeId)
    {
        var chain = new List<TrajectoryNode>();
        var currentId = nodeId;

        while (currentId is not null && _nodes.TryGetValue(currentId, out var node))
        {
            chain.Add(node);
            currentId = node.ParentId;
        }

        return chain;
    }

    /// <summary>
    /// Get leaf nodes (nodes with no children). These are the most recent
    /// states in each trajectory branch.
    /// </summary>
    public IReadOnlyList<TrajectoryNode> GetLeafNodes()
    {
        return _nodes.Values.Where(n => !_children.ContainsKey(n.Id)).ToList();
    }

    /// <summary>Get all nodes in the trajectory (for full traversal).</summary>
    public IReadOnlyCollection<TrajectoryNode> AllNodes => _nodes.Values;

    public int NodeCount => _nodes.Count;

    /// <summary>Clear the trajectory for a new session/episode.</summary>
    public void Clear()
    {
        _nodes.Clear();
        _children.Clear();
        _visitedSituations.Clear();
    }
}
