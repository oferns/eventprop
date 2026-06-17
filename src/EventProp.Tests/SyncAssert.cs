using System.Runtime.CompilerServices;

namespace CW.EventProp.Tests.SyncAssertions;

/// <summary>
/// Synchronous assertion helpers for ref struct types that cannot cross async boundaries.
/// For value types: use <c>SyncAssert.That(value)</c> → chained assertions.
/// For ref structs: use direct methods like <c>SyncAssert.IsCloseTo(value, expected, tolerance)</c>.
/// </summary>
public static class SyncAssert
{
    // ── Non-ref struct entry points (double, int) ─────────────────────

    public static ValueAssertions That(double value) => new(value);
    public static IntAssertions That(int value) => new(value);

    // ── Ref struct entry points (direct static methods, no wrapper) ───

    // StridedSpan helpers
    public static void IsEqualTo(StridedSpan span, int expected)
    {
        if (span.Length != expected)
            throw new EqualException(expected, span.Length);
    }

    public static void IsCloseTo(StridedSpan span, double expected, double tolerance)
    {
        if (Math.Abs(span[0] - expected) > tolerance)
            throw new InRangeException(expected, span[0], expected - tolerance, expected + tolerance);
    }

    public static void IsLessThanOrEqualTo(StridedSpan span, int expected)
    {
        if (span.Length > expected)
            throw new InRangeException(expected, span.Length, expected, int.MaxValue);
    }

    // MatrixView helpers
    public static void IsEqualTo(MatrixView view, int expected)
    {
        if (view.Rows != expected && view.Cols != expected)
            throw new EqualException(expected, $"{view.Rows}x{view.Cols}");
    }

    public static void IsCloseTo(MatrixView view, double expected, double tolerance)
    {
        if (Math.Abs(view[0, 0] - expected) > tolerance)
            throw new InRangeException(expected, view[0, 0], expected - tolerance, expected + tolerance);
    }

    // Span<double> helpers
    public static void IsCloseTo(Span<double> span, double expected, double tolerance)
    {
        if (Math.Abs(span[0] - expected) > tolerance)
            throw new InRangeException(expected, span[0], expected - tolerance, expected + tolerance);
    }

    public static void IsEqualTo(Span<double> span, int expected)
    {
        if (span.Length != expected)
            throw new EqualException(expected, span.Length);
    }

    public static void IsLessThanOrEqualTo(Span<double> span, int expected)
    {
        if (span.Length > expected)
            throw new InRangeException(expected, span.Length, expected, int.MaxValue);
    }

    public static void IsSameReferenceAs(Span<double> actual, Span<double> expected)
    {
        if (!Unsafe.AreSame(ref actual[0], ref expected[0]))
            throw new NotEqualException("<same reference>", "<different reference>");
    }

    // ── Value type chained assertions (double) ────────────────────────

    public sealed class ValueAssertions
    {
        private readonly double _value;
        internal ValueAssertions(double value) => _value = value;

        public void IsEqualTo(double expected, double tolerance = 0)
        {
            if (tolerance == 0 && _value != expected)
                throw new EqualException(expected, _value);
            if (tolerance > 0 && Math.Abs(_value - expected) > tolerance)
                throw new InRangeException(expected, _value, expected - tolerance, expected + tolerance);
        }

        public void IsNaN()
        {
            if (!double.IsNaN(_value)) throw new NotEqualException(double.NaN, _value);
        }

        public void IsNotNaN()
        {
            if (double.IsNaN(_value)) throw new EqualException(double.NaN, _value);
        }

        public void IsGreaterThan(double expected)
        {
            if (_value <= expected) throw new InRangeException(expected, _value, double.NegativeInfinity, expected);
        }

        public void IsGreaterThanOrEqualTo(double expected)
        {
            if (_value < expected) throw new InRangeException(expected, _value, double.NegativeInfinity, expected);
        }

        public void IsLessThan(double expected)
        {
            if (_value >= expected) throw new InRangeException(expected, _value, expected, double.PositiveInfinity);
        }

        public void IsLessThanOrEqualTo(double expected)
        {
            if (_value > expected) throw new InRangeException(expected, _value, expected, double.PositiveInfinity);
        }

        public void IsCloseTo(double expected, double tolerance) => IsEqualTo(expected, tolerance);

        /// <summary>Compare with relative tolerance (for accumulated sums where FP noise scales with magnitude).</summary>
        public void IsCloseToRelative(double expected, double relativeTolerance)
        {
            if (expected == 0.0 && _value == 0.0) return;
            double scale = Math.Max(Math.Abs(expected), Math.Abs(_value));
            if (Math.Abs(_value - expected) > relativeTolerance * scale)
                throw new InRangeException(expected, _value, expected - relativeTolerance * scale, expected + relativeTolerance * scale);
        }
    }

    // ── Value type chained assertions (int) ───────────────────────────

    public sealed class IntAssertions
    {
        private readonly int _value;
        internal IntAssertions(int value) => _value = value;

        public void IsEqualTo(int expected)
        {
            if (_value != expected) throw new EqualException(expected, _value);
        }

        public void IsGreaterThan(int expected)
        {
            if (_value <= expected) throw new InRangeException(expected, _value, int.MinValue, expected);
        }

        public void IsGreaterThanOrEqualTo(int expected)
        {
            if (_value < expected) throw new InRangeException(expected, _value, int.MinValue, expected);
        }

        public void IsLessThan(int expected)
        {
            if (_value >= expected) throw new InRangeException(expected, _value, expected, int.MaxValue);
        }

        public void IsLessThanOrEqualTo(int expected)
        {
            if (_value > expected) throw new InRangeException(expected, _value, expected, int.MaxValue);
        }

        public void IsNaN() => throw new InvalidOperationException("Cannot assert NaN on int.");
    }

    // ── Object-based assertions (for IsInstanceOfType, IsNull) ────────

    public static ObjectAssertions That(object? value) => new(value);

    public sealed class ObjectAssertions
    {
        private readonly object? _value;
        internal ObjectAssertions(object? value) => _value = value;

        public void IsNull()
        {
            if (_value is not null) throw new EqualException(null, _value);
        }

        public void IsInstanceOfType(Type expectedType)
        {
            if (_value is not null && !expectedType.IsInstanceOfType(_value))
                throw new NotEqualException(expectedType.FullName, _value.GetType().FullName);
        }

        public void IsEqualTo(int expected)
        {
            if (!Equals(_value, expected)) throw new EqualException(expected, _value);
        }

        public void IsGreaterThan(int expected)
        {
            if (_value is not int v || v <= expected)
                throw new InRangeException(expected, _value ?? 0, int.MinValue, expected);
        }

        public void IsGreaterThanOrEqualTo(int expected)
        {
            if (_value is not int v || v < expected)
                throw new InRangeException(expected, _value ?? 0, int.MinValue, expected);
        }

        public void IsEqualTo(double expected, double tolerance = 0)
        {
            if (tolerance == 0 && !Equals(_value, expected)) throw new EqualException(expected, _value);
            if (tolerance > 0 && Math.Abs(Convert.ToDouble(_value) - expected) > tolerance)
                throw new InRangeException(expected, _value, expected - tolerance, expected + tolerance);
        }

        public void IsCloseTo(double expected, double tolerance) => IsEqualTo(expected, tolerance);
    }

    // ── Minimal exception types (avoids xUnit dependency) ─────────────

    private sealed class EqualException : Exception
    {
        public object? Expected { get; }
        public object? Actual { get; }
        public EqualException(object? expected, object? actual)
            : base($"Expected: {expected}{Environment.NewLine}Actual: {actual}")
        { Expected = expected; Actual = actual; }
    }

    private sealed class NotEqualException : Exception
    {
        public object? Expected { get; }
        public object? Actual { get; }
        public NotEqualException(object? expected, object? actual)
            : base($"Expected not equal: {expected}{Environment.NewLine}Actual: {actual}")
        { Expected = expected; Actual = actual; }
    }

    private sealed class InRangeException : Exception
    {
        public object? ExpectedInRange { get; }
        public object? Actual { get; }
        public InRangeException(object? expected, object? actual, object? rangeStart, object? rangeEnd)
            : base($"Expected in range [{rangeStart}, {rangeEnd}]{Environment.NewLine}Actual: {actual}")
        { ExpectedInRange = expected; Actual = actual; }
    }
}
