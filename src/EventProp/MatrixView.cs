namespace CW.EventProp;

/// <summary>
/// Zero-copy view of a row-major double matrix stored in a flat array.
/// </summary>
public ref struct MatrixView
{
    public Span<double> Data { get; }
    public int Rows { get; }
    public int Cols { get; }

    public MatrixView(Span<double> data, int rows, int cols)
    {
        Data = data;
        Rows = rows;
        Cols = cols;
    }

    /// <summary>Get a row as a span.</summary>
    public ReadOnlySpan<double> Row(int r) => Data.Slice(r * Cols, Cols);

    /// <summary>Get a column as a strided view (no allocation).</summary>
    public StridedSpan Col(int c) => new(Data, c, Cols);

    /// <summary>Element access: w[row, col].</summary>
    public double this[int row, int col] => Data[row * Cols + col];

    /// <summary>Transpose this matrix into the destination array.</summary>
    public void Transpose(Span<double> dest)
    {
        for (int r = 0; r < Rows; r++)
            for (int c = 0; c < Cols; c++)
                dest[c * Rows + r] = Data[r * Cols + c];
    }

    /// <summary>Transpose this matrix into a new buffer. Caller constructs MatrixView from result.</summary>
    public double[] TransposeOwned()
    {
        var buf = new double[Rows * Cols];
        Transpose(buf);
        return buf;
    }

    /// <summary>Reset all values to zero.</summary>
    public void Zero() => Data.Clear();

    /// <summary>Create a zero matrix.</summary>
    public static MatrixView Zero(int rows, int cols) => new(new double[rows * cols], rows, cols);
}
