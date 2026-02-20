using System.Security.Cryptography;
using System.Text;

namespace RichLearning.Models;

/// <summary>
/// A single node in the trajectory DAG (Active Manifold S^A).
///
/// Each node represents one state evaluation in a session/episode.
/// Nodes are Merkle-linked to their parent, forming a verifiable causal chain.
///
/// From DAPSA v2.1:
///   v_t contains: S_p(t), S_a(t), A(t), Q(v_t), H(v_t)
///
/// The domain adapter maps these generic fields to domain-specific concepts:
///   - Chess: SituationHash = Zobrist, Action = move/evaluation
///   - Robotics: SituationHash = pose hash, Action = motor command
///   - LLM: SituationHash = query embedding hash, Action = answer
///   - Trading: SituationHash = market regime hash, Action = trade decision
/// </summary>
public sealed class TrajectoryNode
{
    public string Id { get; }
    public string SituationHash { get; }
    public string Action { get; }
    public int Depth { get; }
    public double QValue { get; internal set; }
    public string? ParentId { get; }
    public string? ParentMerkleHash { get; }
    public string MerkleHash { get; }
    public DateTimeOffset Timestamp { get; }

    /// <summary>Optional domain-specific metadata.</summary>
    public Dictionary<string, object>? Metadata { get; set; }

    public TrajectoryNode(
        string situationHash,
        string action,
        int depth,
        double qValue,
        string? parentId,
        string? parentMerkleHash)
    {
        Id = Guid.NewGuid().ToString("N")[..16];
        SituationHash = situationHash;
        Action = action;
        Depth = depth;
        QValue = qValue;
        ParentId = parentId;
        ParentMerkleHash = parentMerkleHash;
        Timestamp = DateTimeOffset.UtcNow;
        MerkleHash = ComputeMerkleHash();
    }

    /// <summary>
    /// Merkle hash = SHA256(situationHash | action | depth | parentMerkleHash).
    /// Truncated to 16 hex chars (64 bits) for space efficiency.
    ///
    /// This provides ordering proof: changing any ancestor invalidates all descendants.
    /// </summary>
    private string ComputeMerkleHash()
    {
        var input = $"{SituationHash}|{Action}|{Depth}|{ParentMerkleHash ?? "root"}";
        var hash = SHA256.HashData(Encoding.UTF8.GetBytes(input));
        return Convert.ToHexString(hash)[..16];
    }
}
