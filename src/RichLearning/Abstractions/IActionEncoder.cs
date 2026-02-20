namespace RichLearning.Abstractions;

/// <summary>
/// Encodes a complex action type into a deterministic string representation.
/// Essential for fossilizing non-string generic actions and trajectory tracking.
/// </summary>
public interface IActionEncoder<TAction>
{
    /// <summary>
    /// Serializes an action into a deterministic string that can be stored 
    /// inside DapsaEngine fossils and Merkle Trajectories.
    /// </summary>
    string Encode(TAction? action);

    /// <summary>
    /// Deserializes a string back into the corresponding domain action.
    /// Essential when retrieving a passive hit from the fossil vault.
    /// </summary>
    TAction Decode(string encodedAction);
}
