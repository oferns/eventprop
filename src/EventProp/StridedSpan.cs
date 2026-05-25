namespace CW.EventProp;

/// <summary>
/// A read-only span that strides through an array with a given step.
/// Used for column access in row-major matrices without allocation.
/// </summary>
public readonly ref struct StridedSpan
{
    private readonly ReadOnlySpan<double> _data;
    private readonly int _offset;
    private readonly int _stride;

    public StridedSpan(ReadOnlySpan<double> data, int offset, int stride)
    {
        _data = data;
        _offset = offset;
        _stride = stride;
    }

    public int Length => (_data.Length - 1 - _offset) / _stride + 1;

    public double this[int index] => _data[_offset + index * _stride];
}
