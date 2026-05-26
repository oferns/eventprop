using System.Runtime.CompilerServices;
using CW.EventProp;
using CW.EventProp.Tests.SyncAssertions;

namespace CW.EventProp.Tests;

public class MatrixViewTests
{
    [Test]
    public void Constructor_SetsPropertiesCorrectly()
    {
        double[] data = [1, 2, 3, 4, 5, 6];
        var view = new MatrixView(data, 2, 3);

        SyncAssert.That(view.Rows).IsEqualTo(2);
        SyncAssert.That(view.Cols).IsEqualTo(3);

        // Data should reference the same array
        SyncAssert.IsSameReferenceAs(view.Data, data);
    }

    [Test]
    public void Row_ReturnsCorrectSlice()
    {
        double[] data = [1, 2, 3, 4, 5, 6];
        var view = new MatrixView(data, 2, 3);

        var row0 = view.Row(0);
        double r0c0 = row0[0], r0c1 = row0[1], r0c2 = row0[2];

        SyncAssert.That((int)row0.Length).IsEqualTo(3);
        SyncAssert.That(r0c0).IsCloseTo(1.0, 1e-12);
        SyncAssert.That(r0c1).IsCloseTo(2.0, 1e-12);
        SyncAssert.That(r0c2).IsCloseTo(3.0, 1e-12);

        var row1 = view.Row(1);
        double r1c0 = row1[0], r1c1 = row1[1], r1c2 = row1[2];

        SyncAssert.That(r1c0).IsCloseTo(4.0, 1e-12);
        SyncAssert.That(r1c1).IsCloseTo(5.0, 1e-12);
        SyncAssert.That(r1c2).IsCloseTo(6.0, 1e-12);
    }

    [Test]
    public void Col_ReturnsCorrectStridedAccess()
    {
        double[] data = [1, 2, 3, 4, 5, 6];
        var view = new MatrixView(data, 2, 3);

        var col0 = view.Col(0);
        double c0v0 = col0[0], c0v1 = col0[1];

        SyncAssert.That((int)col0.Length).IsEqualTo(2);
        SyncAssert.That(c0v0).IsCloseTo(1.0, 1e-12);
        SyncAssert.That(c0v1).IsCloseTo(4.0, 1e-12);

        var col1 = view.Col(1);
        double c1v0 = col1[0], c1v1 = col1[1];

        SyncAssert.That(c1v0).IsCloseTo(2.0, 1e-12);
        SyncAssert.That(c1v1).IsCloseTo(5.0, 1e-12);
    }

    [Test]
    public void Indexer_AccessByRowCol()
    {
        double[] data = [1, 2, 3, 4, 5, 6];
        var view = new MatrixView(data, 2, 3);

        double v00 = view[0, 0], v01 = view[0, 1], v02 = view[0, 2];
        double v10 = view[1, 0], v11 = view[1, 1], v12 = view[1, 2];

        SyncAssert.That(v00).IsCloseTo(1.0, 1e-12);
        SyncAssert.That(v01).IsCloseTo(2.0, 1e-12);
        SyncAssert.That(v02).IsCloseTo(3.0, 1e-12);
        SyncAssert.That(v10).IsCloseTo(4.0, 1e-12);
        SyncAssert.That(v11).IsCloseTo(5.0, 1e-12);
        SyncAssert.That(v12).IsCloseTo(6.0, 1e-12);
    }

    [Test]
    public void TransposeOwned_ReturnsCorrectTransposedMatrix()
    {
        // [1 2 3; 4 5 6] → transpose → [1 4; 2 5; 3 6]
        // Row-major: [1, 4, 2, 5, 3, 6]
        double[] data = [1, 2, 3, 4, 5, 6];
        var view = new MatrixView(data, 2, 3);

        var transposed = view.TransposeOwned();
        SyncAssert.That((int)transposed.Length).IsEqualTo(6);
        double t0 = transposed[0], t1 = transposed[1], t2 = transposed[2];
        double t3 = transposed[3], t4 = transposed[4], t5 = transposed[5];
        SyncAssert.That(t0).IsCloseTo(1.0, 1e-12);
        SyncAssert.That(t1).IsCloseTo(4.0, 1e-12);
        SyncAssert.That(t2).IsCloseTo(2.0, 1e-12);
        SyncAssert.That(t3).IsCloseTo(5.0, 1e-12);
        SyncAssert.That(t4).IsCloseTo(3.0, 1e-12);
        SyncAssert.That(t5).IsCloseTo(6.0, 1e-12);
    }

    [Test]
    public void TransposeOwned_SquareMatrix_ReturnsCorrectTranspose()
    {
        double[] data = [1, 2, 3, 4];
        var view = new MatrixView(data, 2, 2);

        var transposed = view.TransposeOwned();
        SyncAssert.That(transposed[0]).IsCloseTo(1.0, 1e-12);
        SyncAssert.That(transposed[1]).IsCloseTo(3.0, 1e-12);
        SyncAssert.That(transposed[2]).IsCloseTo(2.0, 1e-12);
        SyncAssert.That(transposed[3]).IsCloseTo(4.0, 1e-12);
    }

    [Test]
    public void TransposeOwned_DoesNotModifyOriginal()
    {
        double[] data = [1, 2, 3, 4];
        var view = new MatrixView(data, 2, 2);

        _ = view.TransposeOwned();
        SyncAssert.That(view[0, 0]).IsCloseTo(1.0, 1e-12);
        SyncAssert.That(view[0, 1]).IsCloseTo(2.0, 1e-12);
    }

    [Test]
    public void Transpose_WritesIntoDestBuffer()
    {
        double[] data = [1, 2, 3, 4]; // [1 2; 3 4]
        var view = new MatrixView(data, 2, 2);

        var dest = new double[4];
        view.Transpose(dest);

        // Should become [1 3; 2 4]
        SyncAssert.That(dest[0]).IsCloseTo(1.0, 1e-12);
        SyncAssert.That(dest[1]).IsCloseTo(3.0, 1e-12);
        SyncAssert.That(dest[2]).IsCloseTo(2.0, 1e-12);
        SyncAssert.That(dest[3]).IsCloseTo(4.0, 1e-12);
    }

    [Test]
    public void Transpose_DoesNotModifyOriginalData()
    {
        double[] data = [1, 2, 3, 4];
        var view = new MatrixView(data, 2, 2);

        var dest = new double[4];
        view.Transpose(dest);

        SyncAssert.That(data[0]).IsCloseTo(1.0, 1e-12);
        SyncAssert.That(data[1]).IsCloseTo(2.0, 1e-12);
    }

    [Test]
    public void Zero_ZerosAllElements()
    {
        double[] data = [1, 2, 3, 4];
        var view = new MatrixView(data, 2, 2);

        view.Zero();
        SyncAssert.That(view[0, 0]).IsCloseTo(0.0, 1e-12);
        SyncAssert.That(view[0, 1]).IsCloseTo(0.0, 1e-12);
        SyncAssert.That(view[1, 0]).IsCloseTo(0.0, 1e-12);
        SyncAssert.That(view[1, 1]).IsCloseTo(0.0, 1e-12);
    }

    [Test]
    public void Zero_ModifiesUnderlyingArray()
    {
        double[] data = [1, 2, 3, 4];
        var view = new MatrixView(data, 2, 2);

        view.Zero();
        SyncAssert.That(data[0]).IsCloseTo(0.0, 1e-12);
        SyncAssert.That(data[3]).IsCloseTo(0.0, 1e-12);
    }

    [Test]
    [MatrixDataSource]
    public void Zero_FactoryShapeVariants_ReturnsAllZeros([Matrix("rectangular", "square")] string shape)
    {
        (int rows, int cols) = shape switch
        {
            "rectangular" => (3, 4),
            "square" => (5, 5),
            _ => throw new ArgumentException($"Unknown shape: {shape}")
        };

        var view = MatrixView.Zero(rows, cols);

        SyncAssert.That(view.Rows).IsEqualTo(rows);
        SyncAssert.That(view.Cols).IsEqualTo(cols);

        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                SyncAssert.That(view[r, c]).IsCloseTo(0.0, 1e-12);
    }

    [Test]
    [MatrixDataSource]
    public void MatrixView_RowColumnVectors([Matrix("row", "column")] string orientation)
    {
        (int rows, int cols) = orientation switch
        {
            "row" => (1, 4),
            "column" => (3, 1),
            _ => throw new ArgumentException($"Unknown orientation: {orientation}")
        };

        double[] data = new double[rows * cols];
        for (int i = 0; i < data.Length; i++)
            data[i] = i + 1.0;

        var view = new MatrixView(data, rows, cols);

        SyncAssert.That(view.Rows).IsEqualTo(rows);
        SyncAssert.That(view.Cols).IsEqualTo(cols);

        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                SyncAssert.That(view[r, c]).IsCloseTo(r * cols + c + 1.0, 1e-12);
    }

    [Test]
    [MatrixDataSource]
    public void TransposeOwned_ShapeVariants_ReturnsCorrectTranspose([Matrix("rectangular", "square", "wide", "tall")] string shape)
    {
        (int rows, int cols) = shape switch
        {
            "rectangular" => (10, 20),
            "square" => (5, 5),
            "wide" => (1, 6),
            "tall" => (6, 1),
            _ => throw new ArgumentException($"Unknown shape: {shape}")
        };

        double[] data = new double[rows * cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                data[r * cols + c] = r * cols + c;

        var view = new MatrixView(data, rows, cols);
        var transposed = view.TransposeOwned();

        SyncAssert.That((int)transposed.Length).IsEqualTo(rows * cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                SyncAssert.That(transposed[c * rows + r]).IsCloseTo(data[r * cols + c], 1e-12);
    }

    [Test]
    public void MatrixView_DataReferenceIsSameAsOriginal()
    {
        double[] data = [1, 2, 3];
        var view = new MatrixView(data, 1, 3);

        SyncAssert.IsSameReferenceAs(view.Data, data);
    }

    [Test]
    public void MatrixView_IndexerIsReadOnly()
    {
        double[] data = [1, 2, 3];
        var view = new MatrixView(data, 1, 3);

        // Indexer should be read-only — can't assign to it
        double val = view[0, 0];
        SyncAssert.That(val).IsCloseTo(1.0, 1e-12);
    }

    [Test]
    [MatrixDataSource]
    public void MatrixView_Transpose_ShapeVariants_ReturnsCorrectTranspose([Matrix("rectangular", "square")] string shape)
    {
        double[] data;
        int rows, cols;

        switch (shape)
        {
            case "rectangular": // 2x3 → 3x2
                data = [1, 2, 3, 4, 5, 6];
                rows = 2; cols = 3;
                break;
            case "square": // 2x2
                data = [1, 2, 3, 4];
                rows = 2; cols = 2;
                break;
            default:
                throw new ArgumentException($"Unknown shape: {shape}");
        }

        var view = new MatrixView(data, rows, cols);
        var dest = new double[rows * cols];
        view.Transpose(dest);

        // Verify transpose: dest[c * rows + r] == data[r * cols + c]
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                SyncAssert.That(dest[c * rows + r]).IsCloseTo(data[r * cols + c], 1e-12);
    }

    [Test]
    public void MatrixView_RowAndColAreConsistent()
    {
        double[] data = [1, 2, 3, 4, 5, 6];
        var view = new MatrixView(data, 2, 3);

        // Row-major: [1 2 3; 4 5 6]
        var row0 = view.Row(0);
        double r0c0 = row0[0], r0c1 = row0[1], r0c2 = row0[2];

        var col0 = view.Col(0);
        double c0v0 = col0[0], c0v1 = col0[1];

        // Row 0, Col 0 should be the same element
        SyncAssert.That(r0c0).IsCloseTo(c0v0, 1e-12);
        // Row 0 should be [1, 2, 3]
        SyncAssert.That(r0c0).IsCloseTo(1.0, 1e-12);
        SyncAssert.That(r0c1).IsCloseTo(2.0, 1e-12);
        SyncAssert.That(r0c2).IsCloseTo(3.0, 1e-12);
        // Col 0 should be [1, 4]
        SyncAssert.That(c0v0).IsCloseTo(1.0, 1e-12);
        SyncAssert.That(c0v1).IsCloseTo(4.0, 1e-12);
    }

    [Test]
    public void MatrixView_ZeroIsStaticFactory()
    {
        var view = MatrixView.Zero(2, 3);

        SyncAssert.That(view.Rows).IsEqualTo(2);
        SyncAssert.That(view.Cols).IsEqualTo(3);

        // All elements should be zero
        for (int r = 0; r < view.Rows; r++) {
            for (int c = 0; c < view.Cols; c++) {
                SyncAssert.That(view[r, c]).IsCloseTo(0.0, 1e-12);
            }
        }
    }

    [Test]
    public void MatrixView_TransposeIdentity()
    {
        double[] data = [1, 0, 0, 2]; // Identity-like
        var view = new MatrixView(data, 2, 2);

        var dest = new double[4];
        view.Transpose(dest);

        SyncAssert.That(dest[0]).IsCloseTo(1.0, 1e-12);
        SyncAssert.That(dest[3]).IsCloseTo(2.0, 1e-12);
        SyncAssert.That(dest[1]).IsCloseTo(0.0, 1e-12);
        SyncAssert.That(dest[2]).IsCloseTo(0.0, 1e-12);
    }

    [Test]
    public void MatrixView_TransposeLargerMatrix()
    {
        // 3x4 matrix
        double[] data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        var view = new MatrixView(data, 3, 4);

        var dest = new double[12];
        view.Transpose(dest);

        // 4x3 result: column j of original = row j of dest
        SyncAssert.That(dest[0]).IsCloseTo(1.0, 1e-12); // dest[0,0] = orig[0,0]
        SyncAssert.That(dest[1]).IsCloseTo(5.0, 1e-12); // dest[0,1] = orig[1,0]
        SyncAssert.That(dest[2]).IsCloseTo(9.0, 1e-12); // dest[0,2] = orig[2,0]
        SyncAssert.That(dest[3]).IsCloseTo(2.0, 1e-12); // dest[1,0] = orig[0,1]
        SyncAssert.That(dest[11]).IsCloseTo(12.0, 1e-12); // dest[3,2] = orig[2,3]
    }
}
