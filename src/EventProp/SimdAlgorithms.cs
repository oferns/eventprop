using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace CW.EventProp;

/// <summary>
/// SIMD-accelerated span operations using Vector256<double>.
/// Falls back to scalar when hardware acceleration is unavailable or spans are too small.
/// Zero-allocation — all methods operate on spans. Uses unsafe pointers for Load/Store only.
/// </summary>
public static class SimdAlgorithms
{
    private static readonly int VEC = Vector256<double>.Count; // Runtime: 4 on AVX-512, 3 on SVE

    /// <summary>Whether SIMD double operations are hardware-accelerated.</summary>
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
            void* pDestBase = Unsafe.AsPointer(ref MemoryMarshal.GetReference(dest));
            void* pSrcBase = Unsafe.AsPointer(ref MemoryMarshal.GetReference(src));
            void* pWBase = Unsafe.AsPointer(ref MemoryMarshal.GetReference(weights));

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
            void* pBase = Unsafe.AsPointer(ref MemoryMarshal.GetReference(span));
            for (int i = 0; i < vecCount; i++)
            {
                double* p = (double*)pBase + i * VEC;
                var s = Vector256.Load<double>(p);
                (s * vScalar).Store(p);
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
            void* pA = Unsafe.AsPointer(ref MemoryMarshal.GetReference(a));
            void* pB = Unsafe.AsPointer(ref MemoryMarshal.GetReference(b));

            for (int i = 0; i < vecCount; i++)
            {
                double* pa = (double*)pA + i * VEC;
                double* pb = (double*)pB + i * VEC;
                var va = Vector256.Load<double>(pa);
                var vb = Vector256.Load<double>(pb);
                acc += HorizontalSum(va * vb);
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
        // ARM64 SVE: Count=3, AVX-512: Count=4. Scalar loop is fine for 3-4 elements.
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
            void* pDest = Unsafe.AsPointer(ref MemoryMarshal.GetReference(dest));
            void* pA = Unsafe.AsPointer(ref MemoryMarshal.GetReference(a));
            void* pB = Unsafe.AsPointer(ref MemoryMarshal.GetReference(b));

            for (int i = 0; i < vecCount; i++)
            {
                int offset = i * VEC;
                var va = Vector256.Load<double>((double*)pA + offset);
                var vb = Vector256.Load<double>((double*)pB + offset);
                (va + vb).Store((double*)pDest + offset);
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
            void* pDest = Unsafe.AsPointer(ref MemoryMarshal.GetReference(dest));
            void* pA = Unsafe.AsPointer(ref MemoryMarshal.GetReference(a));
            void* pB = Unsafe.AsPointer(ref MemoryMarshal.GetReference(b));

            for (int i = 0; i < vecCount; i++)
            {
                int offset = i * VEC;
                var va = Vector256.Load<double>((double*)pA + offset);
                var vb = Vector256.Load<double>((double*)pB + offset);
                (va - vb).Store((double*)pDest + offset);
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
            void* pDest = Unsafe.AsPointer(ref MemoryMarshal.GetReference(dest));
            void* pA = Unsafe.AsPointer(ref MemoryMarshal.GetReference(a));
            void* pB = Unsafe.AsPointer(ref MemoryMarshal.GetReference(b));

            for (int i = 0; i < vecCount; i++)
            {
                int offset = i * VEC;
                var va = Vector256.Load<double>((double*)pA + offset);
                var vb = Vector256.Load<double>((double*)pB + offset);
                (va * vb).Store((double*)pDest + offset);
            }
        }

        int remainder = dest.Length - vecCount * VEC;
        for (int i = 0; i < remainder; i++)
            dest[vecCount * VEC + i] = a[vecCount * VEC + i] * b[vecCount * VEC + i];
    }
}
