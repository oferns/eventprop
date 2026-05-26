using CW.EventProp;
using CW.EventProp.Tests.SyncAssertions;

namespace CW.EventProp.Tests;

public class StridedSpanTests
{
    [Test]
    public void Constructor_CorrectlyInitializes()
    {
        double[] data = [1, 2, 3, 4, 5, 6];
        var span = new StridedSpan(data, 0, 2);

        SyncAssert.That((int)span.Length).IsEqualTo(3);
    }

    [Test]
    public void Indexer_AccessesCorrectElements()
    {
        double[] data = [10, 20, 30, 40, 50];
        var span = new StridedSpan(data, 1, 2);

        double v0 = span[0], v1 = span[1];
        SyncAssert.That(v0).IsCloseTo(20.0, 1e-12);
        SyncAssert.That(v1).IsCloseTo(40.0, 1e-12);
    }

    [Test]
    [MatrixDataSource]
    public void Indexer_StrideVariants_AccessCorrectElements([Matrix("strideOne", "strideTwo", "strideThree")] string scenario)
    {
        double[] data;
        int offset, stride;
        double[] expected;

        switch (scenario)
        {
            case "strideOne": // contiguous slice
                data = [1, 2, 3, 4, 5]; offset = 1; stride = 1;
                expected = [2.0, 3.0, 4.0, 5.0];
                break;
            case "strideTwo": // row-major [1 2 3; 4 5 6], column access
                data = [1, 2, 3, 4, 5, 6]; offset = 0; stride = 2;
                expected = [1.0, 3.0, 5.0];
                break;
            case "strideThree": // row-major [1 2 3; 4 5 6], column 1 access
                data = [1, 2, 3, 4, 5, 6]; offset = 1; stride = 3;
                expected = [2.0, 5.0];
                break;
            default:
                throw new ArgumentException($"Unknown scenario: {scenario}");
        }

        var span = new StridedSpan(data, offset, stride);
        SyncAssert.That((int)span.Length).IsEqualTo(expected.Length);
        for (int i = 0; i < expected.Length; i++)
            SyncAssert.That(span[i]).IsCloseTo(expected[i], 1e-12);
    }

    [Test]
    [MatrixDataSource]
    public void Length_CorrectForOffsetAndStride([Matrix("default", "nonZeroOffset", "largeStride", "singleElement")] string scenario)
    {
        double[] data;
        int offset, stride, expectedLength;

        switch (scenario)
        {
            case "default":
                data = [1.0, 2.0, 3.0, 4.0, 5.0]; offset = 0; stride = 2; expectedLength = 3;
                break;
            case "nonZeroOffset":
                data = [1.0, 2.0, 3.0, 4.0, 5.0]; offset = 1; stride = 2; expectedLength = 2;
                break;
            case "largeStride":
                data = [1.0, 2.0, 3.0]; offset = 0; stride = 5; expectedLength = 1;
                break;
            case "singleElement":
                data = [42.0]; offset = 0; stride = 1; expectedLength = 1;
                break;
            default:
                throw new ArgumentException($"Unknown scenario: {scenario}");
        }

        var span = new StridedSpan(data, offset, stride);
        SyncAssert.That((int)span.Length).IsEqualTo(expectedLength);
    }

    [Test]
    public void Length_EdgeCase_OffsetAtEnd()
    {
        double[] data = [1, 2, 3, 4, 5];
        var span = new StridedSpan(data, 4, 2);

        SyncAssert.That((int)span.Length).IsEqualTo(1);
        double v = span[0];
        SyncAssert.That(v).IsCloseTo(5.0, 1e-12);
    }

    [Test]
    public void StridedSpan_EmptyData()
    {
        double[] data = [];
        var span = new StridedSpan(data, 0, 1);

        SyncAssert.That((int)span.Length).IsEqualTo(0);
    }

    [Test]
    public void StridedSpan_OffsetBeyondDataLength()
    {
        double[] data = [1, 2, 3];
        var span = new StridedSpan(data, data.Length + 5, 1);

        int len = span.Length;
        SyncAssert.That(len).IsLessThanOrEqualTo(0);
    }

    [Test]
    public void StridedSpan_StrideEqualsDataLength()
    {
        double[] data = [1, 2, 3, 4, 5];
        var span = new StridedSpan(data, 0, data.Length);

        SyncAssert.That((int)span.Length).IsEqualTo(1);
        double v = span[0];
        SyncAssert.That(v).IsCloseTo(1.0, 1e-12);
    }

    [Test]
    public void StridedSpan_OffsetAtDataEnd()
    {
        double[] data = [1, 2, 3];
        var span = new StridedSpan(data, data.Length - 1, 1);

        SyncAssert.That((int)span.Length).IsEqualTo(1);
        double v = span[0];
        SyncAssert.That(v).IsCloseTo(data[data.Length - 1], 1e-12);
    }

    [Test]
    [MatrixDataSource]
    public void StridedSpan_ValueRanges_AccessCorrectly([Matrix("negative", "large", "small")] string range)
    {
        double[] data;

        switch (range)
        {
            case "negative":
                data = [-1, -2, -3, -4];
                break;
            case "large":
                data = [1e6, 2e6, 3e6];
                break;
            case "small":
                data = [1e-6, 2e-6];
                break;
            default:
                throw new ArgumentException($"Unknown range: {range}");
        }

        var span = new StridedSpan(data, 0, 1);
        double tolerance = range == "large" ? 1e-2 : 1e-12;
        for (int i = 0; i < data.Length; i++)
            SyncAssert.That(span[i]).IsCloseTo(data[i], tolerance);
    }

    [Test]
    public void StridedSpan_MatrixColumnAccessPattern()
    {
        double[] data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        int cols = 3;

        var col0 = new StridedSpan(data, 0, cols);
        double c0v0 = col0[0], c0v1 = col0[1], c0v2 = col0[2], c0v3 = col0[3];

        SyncAssert.That((int)col0.Length).IsEqualTo(4);
        SyncAssert.That(c0v0).IsCloseTo(1.0, 1e-12);
        SyncAssert.That(c0v1).IsCloseTo(4.0, 1e-12);
        SyncAssert.That(c0v2).IsCloseTo(7.0, 1e-12);
        SyncAssert.That(c0v3).IsCloseTo(10.0, 1e-12);

        var col1 = new StridedSpan(data, 1, cols);
        double c1v0 = col1[0], c1v3 = col1[3];

        SyncAssert.That((int)col1.Length).IsEqualTo(4);
        SyncAssert.That(c1v0).IsCloseTo(2.0, 1e-12);
        SyncAssert.That(c1v3).IsCloseTo(11.0, 1e-12);

        var col2 = new StridedSpan(data, 2, cols);
        double c2v0 = col2[0], c2v3 = col2[3];

        SyncAssert.That((int)col2.Length).IsEqualTo(4);
        SyncAssert.That(c2v0).IsCloseTo(3.0, 1e-12);
        SyncAssert.That(c2v3).IsCloseTo(12.0, 1e-12);
    }

    [Test]
    public void StridedSpan_MultipleFromSameData()
    {
        double[] data = [1, 2, 3, 4, 5, 6];
        int cols = 3;

        var col0 = new StridedSpan(data, 0, cols);
        var col1 = new StridedSpan(data, 1, cols);

        double c0v0 = col0[0], c1v0 = col1[0];

        SyncAssert.That(c0v0).IsCloseTo(1.0, 1e-12);
        SyncAssert.That(c1v0).IsCloseTo(2.0, 1e-12);
    }

    [Test]
    public void StridedSpan_LargeMatrix()
    {
        int rows = 100, cols = 50;
        double[] data = new double[rows * cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                data[r * cols + c] = r * cols + c;

        var col25 = new StridedSpan(data, 25, cols);
        SyncAssert.That((int)col25.Length).IsEqualTo(rows);

        double first = col25[0], last = col25[rows - 1];
        SyncAssert.That(first).IsCloseTo(25.0, 1e-12);
        SyncAssert.That(last).IsCloseTo((rows - 1) * cols + 25, 1e-12);
    }

    [Test]
    public void StridedSpan_Indexer_ReturnsCorrectValueForLargeIndex()
    {
        double[] data = [10, 20, 30, 40, 50];
        var span = new StridedSpan(data, 0, 2);

        double last = span[2];
        SyncAssert.That(last).IsCloseTo(50.0, 1e-12);
    }

    [Test]
    public void StridedSpan_ComplexStriding()
    {
        double[] data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        var span = new StridedSpan(data, 1, 3);

        double v0 = span[0], v1 = span[1], v2 = span[2];

        SyncAssert.That((int)span.Length).IsEqualTo(3);
        SyncAssert.That(v0).IsCloseTo(1.0, 1e-12);
        SyncAssert.That(v1).IsCloseTo(4.0, 1e-12);
        SyncAssert.That(v2).IsCloseTo(7.0, 1e-12);
    }

    [Test]
    public void StridedSpan_StrideOne_BasicAccess()
    {
        double[] data = [1, 2, 3];
        var span = new StridedSpan(data, 0, 1);
        SyncAssert.That((int)span.Length).IsEqualTo(3);
    }

    [Test]
    public void StridedSpan_ReadOnlyRefStruct()
    {
        double[] data = [1, 2, 3];
        var span = new StridedSpan(data, 0, 1);

        double val = span[0];
        SyncAssert.That(val).IsCloseTo(1.0, 1e-12);
    }

    [Test]
    public void StridedSpan_IdentityMatrixColumns()
    {
        double[] data = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
        int cols = 4;

        for (int c = 0; c < cols; c++)
        {
            var col = new StridedSpan(data, c, cols);
            double diag = col[c];

            SyncAssert.That((int)col.Length).IsEqualTo(4);
            SyncAssert.That(diag).IsCloseTo(1.0, 1e-12);
        }
    }

    [Test]
    public void StridedSpan_AllOnesMatrixColumns()
    {
        double[] data = [1, 1, 1, 1, 1, 1, 1, 1, 1];
        int cols = 3;

        for (int c = 0; c < cols; c++)
        {
            var col = new StridedSpan(data, c, cols);
            double v0 = col[0], v1 = col[1], v2 = col[2];

            SyncAssert.That((int)col.Length).IsEqualTo(3);
            SyncAssert.That(v0).IsCloseTo(1.0, 1e-12);
            SyncAssert.That(v1).IsCloseTo(1.0, 1e-12);
            SyncAssert.That(v2).IsCloseTo(1.0, 1e-12);
        }
    }

    [Test]
    public void StridedSpan_SequentialAccessPattern()
    {
        double[] data = new double[20];
        for (int i = 0; i < 20; i++)
            data[i] = i;

        int rows = 5, cols = 4;
        for (int c = 0; c < cols; c++)
        {
            var col = new StridedSpan(data, c, cols);
            for (int r = 0; r < rows; r++)
            {
                double val = col[r];
                double expected = r * cols + c;
                SyncAssert.That(val).IsCloseTo(expected, 1e-12);
            }
        }
    }
}
