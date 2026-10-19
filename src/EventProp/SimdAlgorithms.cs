using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace CW.EventProp;

/// <summary>
/// SIMD-accelerated span operations using Vector256<double>.
/// Falls back to scalar when hardware acceleration is unavailable or spans are too small.
/// Zero-allocation — all methods operate on spans.
///
/// Pointer acquisition MUST use `fixed`, never `Unsafe.AsPointer(ref GetReference(span))`.
/// The latter takes a raw address from unpinned managed memory: if a concurrent GC
/// relocates the array while the loop runs, the loads read a dead copy and the stores
/// land at the stale address. The live array then keeps its zero initialisation (the
/// symptom is an output buffer of all zeros), and the stale region — already recycled
/// to another live object — is silently overwritten. `fixed` pins for the lifetime of
/// the pointer, which is the only correct form here.
/// </summary>
public static class SimdAlgorithms
{
    // Vector256<double>.Count is a compile-time constant: 256 / 64 = 4. It is NOT
    // runtime-variable — hardware support affects Vector256.IsHardwareAccelerated,
    // not Count. On a CPU without 256-bit vectors the JIT emits a software fallback.
    private static readonly int VEC = Vector256<double>.Count; // always 4

    /// <summary>
    /// True when the Vector256 path is usable. Count is the constant 4, so this is
    /// always true — it is NOT a hardware-acceleration probe. On CPUs without 256-bit
    /// vectors the JIT substitutes a software fallback, which is why the SIMD path is
    /// still taken. A real probe would be Vector256.IsHardwareAccelerated.
    /// </summary>
    public static bool IsAvailable => Vector256<double>.Count > 1;

    // ========================================================================
    // Fused multiply-add row: dest[j] = src[j] + scalar * weights[j]
    // Used by ComputeSums for both sum0 and sum1 accumulation.
    // ========================================================================

    public static void FmaRow(Span<double> dest, ReadOnlySpan<double> src, ReadOnlySpan<double> weights, double scalar)
    {
        if (dest.Length < VEC || !IsAvailable)
        {
            ScalarFmaRow(dest, src, weights, scalar);
            return;
        }

        int vecCount = dest.Length / VEC;
        unsafe
        {
            fixed (double* pDestBase = dest)
            fixed (double* pSrcBase = src)
            fixed (double* pWBase = weights)
            {
            for (int i = 0; i < vecCount; i++)
            {
                int offset = i * VEC;
                double* pDest = (double*)pDestBase + offset;
                double* pSrc = (double*)pSrcBase + offset;
                double* pW = (double*)pWBase + offset;

                var s = Vector256.Load<double>(pSrc);
                var w = Vector256.Load<double>(pW);
                (s + w * scalar).Store(pDest);
            }
            }
        }

        int remainder = dest.Length - vecCount * VEC;
        if (remainder > 0)
        {
            int start = vecCount * VEC;
            for (int i = 0; i < remainder; i++)
                dest[start + i] = src[start + i] + weights[start + i] * scalar;
        }
    }

    static void ScalarFmaRow(Span<double> dest, ReadOnlySpan<double> src, ReadOnlySpan<double> weights, double scalar)
    {
        for (int i = 0; i < dest.Length; i++)
            dest[i] = src[i] + weights[i] * scalar;
    }

    // ========================================================================
    // Scale in-place: span[j] *= scalar
    // Used by BackwardSpikes lambda_v decay.
    // ========================================================================

    public static void ScaleInPlace(Span<double> span, double scalar)
    {
        if (span.Length < VEC || !IsAvailable)
        {
            for (int i = 0; i < span.Length; i++)
                span[i] *= scalar;
            return;
        }

        var vScalar = Vector256.Create(scalar);
        int vecCount = span.Length / VEC;
        unsafe
        {
            fixed (double* pBase = span)
            {
                for (int i = 0; i < vecCount; i++)
                {
                    double* p = pBase + i * VEC;
                    var s = Vector256.Load<double>(p);
                    (s * vScalar).Store(p);
                }
            }
        }

        int remainder = span.Length - vecCount * VEC;
        for (int i = 0; i < remainder; i++)
            span[vecCount * VEC + i] *= scalar;
    }

    // ========================================================================
    // Dot product with SIMD horizontal sum
    // ========================================================================

    public static double Dot(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        if (a.Length < VEC || !IsAvailable)
        {
            double sum = 0.0;
            for (int i = 0; i < a.Length; i++)
                sum += a[i] * b[i];
            return sum;
        }

        int vecCount = a.Length / VEC;
        double acc = 0.0;

        unsafe
        {
            fixed (double* pA = a)
            fixed (double* pB = b)
            {
            for (int i = 0; i < vecCount; i++)
            {
                double* pa = pA + i * VEC;
                double* pb = pB + i * VEC;
                var va = Vector256.Load<double>(pa);
                var vb = Vector256.Load<double>(pb);
                acc += HorizontalSum(va * vb);
            }
            }
        }

        int start = vecCount * VEC;
        for (int i = 0; i < a.Length - start; i++)
            acc += a[start + i] * b[start + i];

        return acc;
    }

    /// <summary>Horizontal sum of Vector256<double> elements.</summary>
    static double HorizontalSum(Vector256<double> v)
    {
        // Count is the constant 4 (256/64), so this sums exactly 4 elements.
        double sum = 0.0;
        for (int i = 0; i < Vector256<double>.Count; i++)
            sum += v.GetElement(i);
        return sum;
    }

    // ========================================================================
    // Element-wise add: dest[j] = a[j] + b[j]
    // ========================================================================

    public static void Add(Span<double> dest, ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        if (dest.Length < VEC || !IsAvailable)
        {
            for (int i = 0; i < dest.Length; i++)
                dest[i] = a[i] + b[i];
            return;
        }

        int vecCount = dest.Length / VEC;
        unsafe
        {
            fixed (double* pDest = dest)
            fixed (double* pA = a)
            fixed (double* pB = b)
            {
            for (int i = 0; i < vecCount; i++)
            {
                int offset = i * VEC;
                var va = Vector256.Load<double>(pA + offset);
                var vb = Vector256.Load<double>(pB + offset);
                (va + vb).Store(pDest + offset);
            }
            }
        }

        int remainder = dest.Length - vecCount * VEC;
        for (int i = 0; i < remainder; i++)
            dest[vecCount * VEC + i] = a[vecCount * VEC + i] + b[vecCount * VEC + i];
    }

    // ========================================================================
    // Element-wise subtract: dest[j] = a[j] - b[j]
    // ========================================================================

    public static void Subtract(Span<double> dest, ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        if (dest.Length < VEC || !IsAvailable)
        {
            for (int i = 0; i < dest.Length; i++)
                dest[i] = a[i] - b[i];
            return;
        }

        int vecCount = dest.Length / VEC;
        unsafe
        {
            fixed (double* pDest = dest)
            fixed (double* pA = a)
            fixed (double* pB = b)
            {
            for (int i = 0; i < vecCount; i++)
            {
                int offset = i * VEC;
                var va = Vector256.Load<double>(pA + offset);
                var vb = Vector256.Load<double>(pB + offset);
                (va - vb).Store(pDest + offset);
            }
            }
        }

        int remainder = dest.Length - vecCount * VEC;
        for (int i = 0; i < remainder; i++)
            dest[vecCount * VEC + i] = a[vecCount * VEC + i] - b[vecCount * VEC + i];
    }

    // ========================================================================
    // Element-wise multiply: dest[j] = a[j] * b[j]
    // ========================================================================

    public static void Multiply(Span<double> dest, ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        if (dest.Length < VEC || !IsAvailable)
        {
            for (int i = 0; i < dest.Length; i++)
                dest[i] = a[i] * b[i];
            return;
        }

        int vecCount = dest.Length / VEC;
        unsafe
        {
            fixed (double* pDest = dest)
            fixed (double* pA = a)
            fixed (double* pB = b)
            {
            for (int i = 0; i < vecCount; i++)
            {
                int offset = i * VEC;
                var va = Vector256.Load<double>(pA + offset);
                var vb = Vector256.Load<double>(pB + offset);
                (va * vb).Store(pDest + offset);
            }
            }
        }

        int remainder = dest.Length - vecCount * VEC;
        for (int i = 0; i < remainder; i++)
            dest[vecCount * VEC + i] = a[vecCount * VEC + i] * b[vecCount * VEC + i];
    }

    // ========================================================================
    // Accumulate scaled: dest[j] += src[j] * scalar
    // Used by BackwardSpikes for gradient accumulation.
    // ========================================================================

    public static void AddScaled(Span<double> dest, ReadOnlySpan<double> src, double scalar)
    {
        if (dest.Length < VEC || !IsAvailable)
        {
            for (int i = 0; i < dest.Length; i++)
                dest[i] += src[i] * scalar;
            return;
        }

        var vScalar = Vector256.Create(scalar);
        int vecCount = dest.Length / VEC;
        unsafe
        {
            fixed (double* pDest = dest)
            fixed (double* pSrc = src)
            {
            for (int i = 0; i < vecCount; i++)
            {
                int offset = i * VEC;
                double* pD = pDest + offset;
                double* pS = pSrc + offset;

                var d = Vector256.Load<double>(pD);
                var s = Vector256.Load<double>(pS);
                (d + s * vScalar).Store(pD);
            }
            }
        }

        int remainder = dest.Length - vecCount * VEC;
        for (int i = 0; i < remainder; i++)
            dest[vecCount * VEC + i] += src[vecCount * VEC + i] * scalar;
    }
}
