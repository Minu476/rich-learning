using RichLearning.Abstractions;

namespace RichLearning;

/// <summary>
/// A fallback implementation of IActionEncoder that simply uses .ToString().
/// Assumes TAction is string, int, or some basic type that can be serialized/deserialized natively.
/// For complex objects, provide a custom IActionEncoder instead.
/// </summary>
public class DefaultActionEncoder<TAction> : IActionEncoder<TAction>
{
    public string Encode(TAction? action) => action?.ToString() ?? "";

    public TAction Decode(string encodedAction)
    {
        // For strings, this is a direct cast if TAction is string
        if (typeof(TAction) == typeof(string))
            return (TAction)(object)encodedAction;

        // Try standard type conversion
        try
        {
            return (TAction)Convert.ChangeType(encodedAction, typeof(TAction));
        }
        catch
        {
            return default!;
        }
    }
}
