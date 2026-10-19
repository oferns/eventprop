namespace CW.EventProp;

/// <summary>
/// A batch of SpikesData — behaves like a list for indexing, slicing, and iteration.
/// </summary>
public sealed class SpikesVector : List<SpikesData>
{
    public SpikesVector() { }

    public SpikesVector(int capacity) : base(capacity) { }

    public SpikesVector(IEnumerable<SpikesData> collection) : base(collection) { }
}

/// <summary>
/// A batch of MaximaData — behaves like a list for indexing, slicing, and iteration.
/// </summary>
public sealed class MaximaVector : List<MaximaData>
{
    public MaximaVector() { }

    public MaximaVector(int capacity) : base(capacity) { }

    public MaximaVector(IEnumerable<MaximaData> collection) : base(collection) { }
}
