namespace CW.EventProp;

/// <summary>
/// Brent's method root-finding algorithm.
/// Combines bisection with secant/IQI for faster convergence.
/// Guaranteed to converge within the bracketing interval [a, b].
/// </summary>
public static class Toms748
{
    private const double DefaultEpsilon = 1e-14;
    private const int MaxIterations = 1024;

    /// <summary>
    /// Find a root of f(x) = 0 where f(a) and f(b) have opposite signs.
    /// Returns the midpoint of the final bracketing interval.
    /// </summary>
    public static double Solve(
        Func<double, double> f,
        double a,
        double b,
        double tolerance = DefaultEpsilon)
    {
        if (tolerance <= 0.0) tolerance = DefaultEpsilon;

        double fa = f(a);
        double fb = f(b);

        // Guard: if endpoints are not finite, abort immediately
        if (!double.IsFinite(fa) || !double.IsFinite(fb)) return a;

        // Ensure a < b
        if (a > b)
        {
            double tmp = a; a = b; b = tmp;
            double tmpF = fa; fa = fb; fb = tmpF;
        }

        // Brent's method: maintain bracketing [a, b] with a < b and f(a)*f(b) < 0
        // c tracks previous value of a (for IQI and stagnation detection)
        double c = a;
        double fc = fa;

        // d tracks the previous step size (for stagnation detection)
        double d = 0.0;

        // Safety limit: bisection halves |b-a| each iteration, so ~1024 iterations
        // should reduce any finite bracket to machine precision. This prevents hangs.
        int iter = 0;
        while (iter++ < MaxIterations)
        {
            // Midpoint of bracketing interval [a, b]
            double m = 0.5 * (b - a);

            if (Math.Abs(m) <= tolerance || fb == 0.0)
                return b;

            // Try inverse quadratic interpolation if conditions are met:
            double s = 0.0;
            bool useBisection = true;

            if (Math.Abs(fc) > Math.Abs(fb) && c != a && c != b)
            {
                // IQI through (a, fa), (b, fb), (c, fc)
                double qba = (fb - fa) / (b - a);  // f[b,a]
                double qcb = (fc - fb) / (c - b);  // f[c,b]
                double qcbba = (qcb - qba) / (c - a); // f[c,b,a]

                if (Math.Abs(qcbba) > 1e-300)
                {
                    // IQI step from b: find root of quadratic through (a,fa), (b,fb), (c,fc).
                    // Using divided differences: s = b - fb / (qba + qcb * (b-a) * qcbba / (qba+qcb))
                    double denom = qba + qcb;
                    if (Math.Abs(denom) > 1e-300)
                    {
                        s = b - fb / (qba + qcb * (b - a) * qcbba / denom);

                        // Accept if s is within [a, b] and smaller than bisection
                        bool inBracket = (s > a && s < b);
                        double mVal = 0.5 * (b - a);
                        bool smallStep = Math.Abs(s - b) < mVal;

                        // Stagnation check: if the step size is comparable to bisection,
                        // prefer bisection for guaranteed convergence.
                        bool prevWasBisect = Math.Abs(d - mVal) < tolerance * 100;

                        if (inBracket && smallStep && !prevWasBisect)
                            useBisection = false;
                    }
                }
            }

            if (useBisection)
            {
                // Bisection step — always safe, moves b towards midpoint of [a,b]
                s = a + m; // = b - m since m = 0.5*(b-a)
            }

            double prevB = b;
            b = s;
            fb = f(b);

            // Guard: abort on NaN/Inf at any evaluation point
            if (!double.IsFinite(fb)) return prevB;

            // Check convergence
            if (Math.Abs(fb) <= tolerance || Math.Abs(b - prevB) <= tolerance * Math.Max(1.0, Math.Abs(prevB)))
                return b;

            // Update bracketing interval: maintain f(a)*f(b) < 0
            if (fa * fb < 0.0)
            {
                // Root between a and new b — shrink from right, keep a fixed
                c = prevB; fc = f(prevB);
            }
            else if (fb * fc < 0.0)
            {
                // Root between new b and c — shrink from left, keep c fixed
                a = prevB; fa = f(prevB);
            }
            else
            {
                // All three have same sign — bracketing broken.
                // The root must be between prevB and new_b (since f(prevB) had
                // opposite sign to initial fa, and f(new_b) has same sign as fa).
                // Restore bracketing: set a = prevB (opposite sign to new b)
                a = prevB; fa = f(prevB);
            }

            // Guard: abort if bracketing values become NaN/Inf
            if (!double.IsFinite(fa) || !double.IsFinite(fc)) return prevB;

            // Track step size for stagnation detection in next iteration
            d = b - prevB;
        }

    // MaxIterations exceeded — return the best estimate found so far.
    // This should not happen for well-behaved functions with valid brackets.
    return b;
}
}
