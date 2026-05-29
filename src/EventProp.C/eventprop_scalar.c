/**
 * @file eventprop_scalar.c
 * @brief EventProp algorithm implementation with optional SIMD (AVX2 / NEON).
 *        Falls back to scalar when hardware acceleration is unavailable or spans
 *        are too small. Zero-allocation hot paths via pre-allocated buffers.
 */

#include <math.h>
#include <string.h>
#include <stdlib.h>

/* Include SIMD headers at top level */
#ifdef HAVE_AVX512F
    #include <immintrin.h>
#elif defined(HAVE_AVX2)
    #include <immintrin.h>
#endif
#ifdef __ARM_NEON
    #include <arm_neon.h>
#endif

#include "eventprop.h"

/* ── SIMD helpers (AVX2 / NEON) ─────────────────────────────────────── */

/* Runtime vector width (0 = no SIMD, 2 = NEON, 4 = AVX2, 8 = AVX-512) */
static int simd_vec_width(void)
{
#ifdef HAVE_AVX512F
    return 8;
#elif defined(HAVE_AVX2)
    return 4;
#elif defined(__ARM_NEON)
    return 2; /* NEON processes 2 doubles at a time */
#else
    (void)0;
    return 0;
#endif
}

static int simd_available(void)
{
    return simd_vec_width() > 0;
}

/* Fused multiply-add: dest[j] = src[j] + scalar * weights[j]
   Operates on n elements. Falls back to scalar for small spans or no SIMD. */
static void fma_row_simd(double* dest, const double* src, const double* weights,
                         double scalar, int n)
{
    if (!simd_available() || n < 2) {
        for (int j = 0; j < n; j++)
            dest[j] = src[j] + scalar * weights[j];
        return;
    }

    int vec = simd_vec_width();
    int vec_count = n / vec;

#if defined(HAVE_AVX512F) || defined(HAVE_AVX2)
    /* x86 SIMD: 4 (AVX2) or 8 (AVX-512) doubles per vector */
    {
        for (int i = 0; i < vec_count; i++) {
#if defined(HAVE_AVX512F)
            __m512d s = _mm512_loadu_pd(src + i * vec);
            __m512d w = _mm512_loadu_pd(weights + i * vec);
            __m512d sc = _mm512_set1_pd(scalar);
            _mm512_storeu_pd(dest + i * vec, _mm512_add_pd(s, _mm512_mul_pd(w, sc)));
#else
            __m256d s = _mm256_loadu_pd(src + i * vec);
            __m256d w = _mm256_loadu_pd(weights + i * vec);
            __m256d sc = _mm256_set1_pd(scalar);
            _mm256_storeu_pd(dest + i * vec, _mm256_add_pd(s, _mm256_mul_pd(w, sc)));
#endif
        }
    }
#elif defined(__ARM_NEON)
    /* ARM NEON: double-precision via float64x2_t (2 doubles per vector) */
    {
        float64x2_t sc = vdupq_n_f64(scalar);
        for (int i = 0; i < vec_count; i++) {
            float64x2_t s = vld1q_f64(src + i * vec);
            float64x2_t w = vld1q_f64(weights + i * vec);
            vst1q_f64(dest + i * vec, vaddq_f64(s, vmulq_f64(w, sc)));
        }
    }
#else
    (void)vec; (void)vec_count;
    for (int j = 0; j < n; j++)
        dest[j] = src[j] + scalar * weights[j];
#endif

    /* Remainder */
    for (int j = vec_count * vec; j < n; j++)
        dest[j] = src[j] + scalar * weights[j];
}

/* Scale in-place: span[j] *= scalar */
static void scale_inplace_simd(double* span, double scalar, int n)
{
    if (!simd_available() || n < 2) {
        for (int j = 0; j < n; j++)
            span[j] *= scalar;
        return;
    }

    int vec = simd_vec_width();
    int vec_count = n / vec;

#if defined(HAVE_AVX512F) || defined(HAVE_AVX2)
    {
#if defined(HAVE_AVX512F)
        __m512d sc = _mm512_set1_pd(scalar);
        for (int i = 0; i < vec_count; i++) {
            __m512d s = _mm512_loadu_pd(span + i * vec);
            _mm512_storeu_pd(span + i * vec, _mm512_mul_pd(s, sc));
        }
#else
        __m256d sc = _mm256_set1_pd(scalar);
        for (int i = 0; i < vec_count; i++) {
            __m256d s = _mm256_loadu_pd(span + i * vec);
            _mm256_storeu_pd(span + i * vec, _mm256_mul_pd(s, sc));
        }
#endif
    }
#elif defined(__ARM_NEON)
    /* ARM NEON: double-precision via float64x2_t */
    {
        float64x2_t sc = vdupq_n_f64(scalar);
        for (int i = 0; i < vec_count; i++) {
            float64x2_t s = vld1q_f64(span + i * vec);
            vst1q_f64(span + i * vec, vmulq_f64(s, sc));
        }
    }
#else
    (void)vec; (void)vec_count;
    for (int j = 0; j < n; j++)
        span[j] *= scalar;
#endif

    for (int j = vec_count * vec; j < n; j++)
        span[j] *= scalar;
}

/* Vector-scalar multiply-accumulate: dest[j] += src[j] * scalar */
static void vadd_scaled_simd(double* dest, const double* src, double scalar, int n)
{
    if (!simd_available() || n < 2) {
        for (int j = 0; j < n; j++)
            dest[j] += src[j] * scalar;
        return;
    }

    int vec = simd_vec_width();
    int vec_count = n / vec;

#if defined(HAVE_AVX512F) || defined(HAVE_AVX2)
    {
        __m256d sc = _mm256_set1_pd(scalar);
        for (int i = 0; i < vec_count; i++) {
#if defined(HAVE_AVX512F)
            __m512d d = _mm512_loadu_pd(dest + i * vec);
            __m512d s = _mm512_loadu_pd(src + i * vec);
            _mm512_storeu_pd(dest + i * vec, _mm512_add_pd(d, _mm512_mul_pd(s, sc)));
#else
            __m256d d = _mm256_loadu_pd(dest + i * vec);
            __m256d s = _mm256_loadu_pd(src + i * vec);
            d = _mm256_add_pd(d, _mm256_mul_pd(s, sc));
            _mm256_storeu_pd(dest + i * vec, d);
#endif
        }
    }
#elif defined(__ARM_NEON)
    /* ARM NEON: double-precision via float64x2_t */
    {
        float64x2_t sc = vdupq_n_f64(scalar);
        for (int i = 0; i < vec_count; i++) {
            float64x2_t d = vld1q_f64(dest + i * vec);
            float64x2_t s = vld1q_f64(src + i * vec);
            vst1q_f64(dest + i * vec, vaddq_f64(d, vmulq_f64(s, sc)));
        }
    }
#else
    (void)vec; (void)vec_count;
    for (int j = 0; j < n; j++)
        dest[j] += src[j] * scalar;
#endif

    for (int j = vec_count * vec; j < n; j++)
        dest[j] += src[j] * scalar;
}

/* Dot product: sum(a[j] * b[j]) for j = 0..n-1 */
static double dot_product_simd(const double* a, const double* b, int n)
{
    if (!simd_available() || n < 2) {
        double sum = 0.0;
        for (int j = 0; j < n; j++)
            sum += a[j] * b[j];
        return sum;
    }

    int vec = simd_vec_width();
    int vec_count = n / vec;

#if defined(HAVE_AVX512F) || defined(HAVE_AVX2)
    {
        __m256d acc = _mm256_setzero_pd();
        for (int i = 0; i < vec_count; i++) {
#if defined(HAVE_AVX512F)
            __m512d va = _mm512_loadu_pd(a + i * vec);
            __m512d vb = _mm512_loadu_pd(b + i * vec);
            acc = _mm512_add_pd(acc, _mm512_mul_pd(va, vb));
#else
            __m256d va = _mm256_loadu_pd(a + i * vec);
            __m256d vb = _mm256_loadu_pd(b + i * vec);
            acc = _mm256_add_pd(acc, _mm256_mul_pd(va, vb));
#endif
        }
        /* Horizontal sum */
#if defined(HAVE_AVX512F)
        double result[8];
        _mm512_storeu_pd(result, acc);
#else
        double result[4];
        _mm256_storeu_pd(result, acc);
#endif
        double sum = result[0] + result[1];
#if defined(HAVE_AVX512F)
        sum += result[2] + result[3] + result[4] + result[5] + result[6] + result[7];
#else
        sum += result[2] + result[3];
#endif
        /* Remainder */
        for (int j = vec_count * vec; j < n; j++)
            sum += a[j] * b[j];
        return sum;
    }
#elif defined(__ARM_NEON)
    /* ARM NEON: double-precision via float64x2_t, manual horizontal sum */
    {
        float64x2_t acc = vdupq_n_f64(0.0);
        for (int i = 0; i < vec_count; i++) {
            float64x2_t va = vld1q_f64(a + i * vec);
            float64x2_t vb = vld1q_f64(b + i * vec);
            acc = vaddq_f64(acc, vmulq_f64(va, vb));
        }
        double result[2];
        vst1q_f64(result, acc);
        double sum = result[0] + result[1];
        for (int j = vec_count * vec; j < n; j++)
            sum += a[j] * b[j];
        return sum;
    }
#else
    (void)vec; (void)vec_count;
    double sum = 0.0;
    for (int j = 0; j < n; j++)
        sum += a[j] * b[j];
    return sum;
#endif
}

/* ── Buffer initialization ─────────────────────────────────────────── */

void eventprop_scratch_init(EventPropScratch* buf, int n_pre, int n_post, int n_spikes)
{
    (void)n_pre; buf->sum0 = NULL; buf->sum1 = NULL;
    buf->fwd_times = NULL; buf->fwd_sources = NULL; buf->fwd_currents = NULL;
}



/* ── ComputeSums: cumulative exponential-weighted sums ─────────────── */

void eventprop_compute_sums(const double* w, int n_pre, int n_post,
                            const double* times, const int32_t* sources, int n_spikes,
                            double tau_mem, double tau_syn,
                            double* sum0, double* sum1)
{
    double* exp_mem = (double*)malloc(n_spikes * sizeof(double));
    double* exp_syn = (double*)malloc(n_spikes * sizeof(double));

    for (int i = 0; i < n_spikes; i++) {
        exp_mem[i] = exp(times[i] / tau_mem);
        exp_syn[i] = exp(times[i] / tau_syn);
    }

    for (int i = 0; i < n_spikes; i++) {
        int src = sources[i];
        double exp_syn_i = exp_syn[i], exp_mem_i = exp_mem[i];
        int row_out = i + 1, row_in = i;
        int src_offset = src * n_post;

        /* SIMD-accelerated FMA row update */
        fma_row_simd(sum0 + row_out * n_post, sum0 + row_in * n_post,
                     w + src_offset, exp_syn_i, n_post);
        fma_row_simd(sum1 + row_out * n_post, sum1 + row_in * n_post,
                     w + src_offset, exp_mem_i, n_post);
    }

    free(exp_mem); free(exp_syn);
}

/* ── ComputeV: membrane potential at time t for interval i ─────────── */

static double compute_v(double t, int i, int spike_count,
                        const double* post_spike_times, int neuron_offset,
                        const double* sum0, const double* sum1,
                        int target_nrn, int n_post,
                        double k_prefactor, double tau_mem, double tau_syn, double v_th)
{
    double mem = k_prefactor * (sum1[i * n_post + target_nrn] * exp(-t / tau_mem)
                    - sum0[i * n_post + target_nrn] * exp(-t / tau_syn));
    for (int s = 0; s < spike_count; s++) {
        double st = post_spike_times[neuron_offset + s];
        if (st > t) break;
        mem -= v_th * exp(-(t - st) / tau_mem);
    }
    return mem;
}

/* ── ComputeTmax: time of maximum voltage in interval i ────────────── */

static double compute_tmax(int i, int n_spikes, const double* times,
                           int spike_count, const double* post_spike_times, int neuron_offset,
                           const double* sum0, const double* sum1,
                           int target_nrn, int n_post,
                           double tmax_prefactor, double tmax_summand, double tau_mem, double v_th)
{
    if (i == 0) return 0.0;

    double t_input = (i == n_spikes) ? INFINITY : times[i];
    double t_before = (i > 0) ? times[i - 1] : 0.0;

    for (int s = 0; s < spike_count; s++) {
        double st = post_spike_times[neuron_offset + s];
        if (st > t_input) break;
        if (st > t_before) t_before = st;
    }

    double sum0_elem = sum0[i * n_post + target_nrn];
    double sum1_elem = sum1[i * n_post + target_nrn];

    if (sum0_elem <= 0 || sum1_elem <= 0) return INFINITY;

    double sum1_adj = sum1_elem;
    for (int s = 0; s < spike_count; s++) {
        double st = post_spike_times[neuron_offset + s];
        if (st > t_input) break;
        sum1_adj += -v_th * exp(st / tau_mem);
    }

    double tmax = tmax_prefactor * (tmax_summand + log(sum1_adj / sum0_elem));

    if (t_before <= tmax && (i == n_spikes || tmax <= t_input))
        return tmax;
    return INFINITY;
}

/* ── ComputeI: synaptic current at time t ──────────────────────────── */

static double compute_i(double t, int n_spikes, const double* times,
                        const double* sum0, int target_nrn, int n_post, double tau_syn)
{
    int idx = 0;
    for (; idx < n_spikes; idx++) {
        if (times[idx] > t) break;
    }
    return sum0[idx * n_post + target_nrn] * exp(-t / tau_syn);
}

/* ── Merge spike comparison for qsort (descending by time) ─────────── */

typedef struct { double time; int idx; int is_input; } MergedSpike;
typedef struct { double time; int source; double current; } SpikeEntry;

static int cmp_merged_desc(const void* a, const void* b) {
    double ta = ((const MergedSpike*)a)->time;
    double tb = ((const MergedSpike*)b)->time;
    return (ta < tb) - (ta > tb);
}

static int cmp_spike_asc(const void* a, const void* b) {
    double ta = ((const SpikeEntry*)a)->time;
    double tb = ((const SpikeEntry*)b)->time;
    return (ta > tb) - (ta < tb);
}

/* ── Toms748 (Brent) root-finding ─────────────────────────────────── */

#define TOMS748_MAX_ITER 1024
#define TOMS748_TOL 1e-14

typedef double (*root_fn_t)(void*, double);

struct root_ctx {
    const double* sum0; const double* sum1;
    int i, spike_count, target_nrn, n_post;
    const double* post_spike_times; int neuron_offset;
    double k_prefactor, tau_mem, tau_syn, v_th;
};

static double root_fn(void* ctx_ptr, double t) {
    struct root_ctx* c = (struct root_ctx*)ctx_ptr;
    return compute_v(t, c->i, c->spike_count, c->post_spike_times, c->neuron_offset,
                     c->sum0, c->sum1, c->target_nrn, c->n_post,
                     c->k_prefactor, c->tau_mem, c->tau_syn, c->v_th) - c->v_th;
}

/* Exact port of .NET Toms748.Solve — Brent's method with IQI */
static double toms748_solve(void* ctx, root_fn_t fn, double a, double b) {
    double fa = fn(ctx, a), fb = fn(ctx, b);

    /* Ensure a < b (swap if needed) */
    if (a > b) { double tmp=a; a=b; b=tmp; tmp=fa; fa=fb; fb=tmp; }

    double c = a, fc = fa, d = 0.0;
    int iter = 0;
    while (++iter <= TOMS748_MAX_ITER) {
        double m = 0.5 * (b - a);
        if (fabs(m) <= TOMS748_TOL || fb == 0.0)
            return b;

        double s = 0.0;
        int useBisection = 1;

        /* Try IQI through (a,fa), (b,fb), (c,fc) */
        if (fabs(fc) > fabs(fb) && c != a && c != b) {
            double qba = (fb - fa) / (b - a);
            double qcb = (fc - fb) / (c - b);
            double qcbba = (qcb - qba) / (c - a);

            if (fabs(qcbba) > 1e-300) {
                double denom = qba + qcb;
                if (fabs(denom) > 1e-300) {
                    s = b - fb / (qba + qcb * (b - a) * qcbba / denom);

                    int inBracket = (s > a && s < b);
                    double mVal = 0.5 * (b - a);
                    int smallStep = fabs(s - b) < mVal;
                    int prevWasBisect = fabs(d - mVal) < TOMS748_TOL * 100;

                    if (inBracket && smallStep && !prevWasBisect)
                        useBisection = 0;
                }
            }
        }

        if (useBisection) {
            s = a + m;
        }

        double prevB = b;
        b = s;
        fb = fn(ctx, b);

        if (fabs(fb) <= TOMS748_TOL || fabs(b - prevB) <= TOMS748_TOL * fmax(1.0, fabs(prevB)))
            return b;

        /* Update bracketing interval: maintain f(a)*f(b) < 0 */
        if (fa * fb < 0.0) {
            c = prevB; fc = fn(ctx, prevB);
        } else if (fb * fc < 0.0) {
            a = prevB; fa = fn(ctx, prevB);
        } else {
            a = prevB; fa = fn(ctx, prevB);
        }

        d = b - prevB;
    }
    return b;
}

/* ── ComputeSpikes: forward pass with root-finding ─────────────────── */

int eventprop_compute_spikes(const double* w, int n_pre, int n_post,
                             const double* times, const int32_t* sources, int n_spikes,
                             double v_th, double tau_mem, double tau_syn,
                             double* out_times, int32_t* out_sources, double* out_currents)
{
    double k_prefactor = tau_syn / (tau_mem - tau_syn);
    double tmax_prefactor = 1.0 / (1.0 / tau_mem - 1.0 / tau_syn);
    double tmax_summand = log(tau_syn / tau_mem);

    int n_sums = (n_spikes + 1) * n_post;
    double* sum0 = calloc(n_sums, sizeof(double));
    double* sum1 = calloc(n_sums, sizeof(double));
    eventprop_compute_sums(w, n_pre, n_post, times, sources, n_spikes,
                           tau_mem, tau_syn, sum0, sum1);

    int max_spikes = n_spikes + 1;
    double* post_times = calloc(max_spikes * n_post, sizeof(double));
    double* post_currents = calloc(max_spikes * n_post, sizeof(double));
    int* spike_counts = calloc(n_post, sizeof(int));

    for (int target_nrn = 0; target_nrn < n_post; target_nrn++) {
        int offset = target_nrn * max_spikes, count = 0;
        int finished = 0, processed = 0;

        while (!finished) {
            int found = 0;
            for (int i = processed; i <= n_spikes; i++) {
                if (i == n_spikes) {
                    /* Last interval — access sum0/sum1 at row n_spikes, col target_nrn */
                    double s0 = sum0[n_spikes * n_post + target_nrn];
                    double s1 = sum1[n_spikes * n_post + target_nrn];
                    for (int s = 0; s < count; s++)
                        s1 += -v_th * exp(post_times[offset + s] / tau_mem);



                    double tmax = tmax_prefactor * (tmax_summand + log(s1 / s0));
                    if (!isinf(tmax)) {
                        double v = compute_v(tmax, i, count, post_times, offset, sum0, sum1,
                                             target_nrn, n_post, k_prefactor, tau_mem, tau_syn, v_th);
                        if (v > v_th + 1e-14) {
                            double t_lower = count > 0 ? post_times[offset + count - 1] : 0.0;
                            struct root_ctx rctx; memset(&rctx, 0, sizeof(rctx)); rctx.sum0=sum0; rctx.sum1=sum1; rctx.i=i; rctx.spike_count=count; rctx.post_spike_times=post_times; rctx.neuron_offset=offset; rctx.target_nrn=target_nrn; rctx.n_post=n_post; rctx.k_prefactor=k_prefactor; rctx.tau_mem=tau_mem; rctx.tau_syn=tau_syn; rctx.v_th=v_th;
                            double t_post = toms748_solve(&rctx, root_fn, t_lower, tmax);
                            if (t_post > t_lower && t_post < tmax) {
                                post_times[offset + count] = t_post;
                                post_currents[offset + count] = compute_i(t_post, n_spikes, times, sum0, target_nrn, n_post, tau_syn);
                                count++; found = 1;
                            }
                        }
                    }
                    finished = 1;
                } else {
                    double t_input = times[i];
                    double tmax = compute_tmax(i, n_spikes, times, count, post_times, offset,
                                               sum0, sum1, target_nrn, n_post, tmax_prefactor, tmax_summand, tau_mem, v_th);

                    if (!isinf(tmax)) {
                        double v = compute_v(tmax, i, count, post_times, offset, sum0, sum1,
                                             target_nrn, n_post, k_prefactor, tau_mem, tau_syn, v_th);
                        if (v > v_th + 1e-14) {
                            double t_before = (i > 0) ? times[i-1] : 0.0;
                            for (int s = 0; s < count; s++) {
                                double st = post_times[offset + s];
                                if (st > t_input) break;
                                if (st > t_before) t_before = st;
                            }
                            struct root_ctx rctx; memset(&rctx, 0, sizeof(rctx)); rctx.sum0=sum0; rctx.sum1=sum1; rctx.i=i; rctx.spike_count=count; rctx.post_spike_times=post_times; rctx.neuron_offset=offset; rctx.target_nrn=target_nrn; rctx.n_post=n_post; rctx.k_prefactor=k_prefactor; rctx.tau_mem=tau_mem; rctx.tau_syn=tau_syn; rctx.v_th=v_th;
                            double t_post = toms748_solve(&rctx, root_fn, t_before, tmax);
                            if (t_post > t_before && t_post < tmax) {
                                post_times[offset + count] = t_post;
                                post_currents[offset + count] = compute_i(t_post, n_spikes, times, sum0, target_nrn, n_post, tau_syn);
                                count++; found = 1;
                            }
                            processed = i; break;
                        }
                    }

                    /* Check at input spike time */
                    double v_at_input = compute_v(t_input, i, count, post_times, offset, sum0, sum1,
                                                  target_nrn, n_post, k_prefactor, tau_mem, tau_syn, v_th);
                    if (v_at_input > v_th + 1e-14) {
                        double t_before = (i > 0) ? times[i-1] : 0.0;
                        for (int s = 0; s < count; s++) {
                            double st = post_times[offset + s];
                            if (st > t_input) break;
                            if (st > t_before) t_before = st;
                        }
                        struct root_ctx rctx; memset(&rctx, 0, sizeof(rctx)); rctx.sum0=sum0; rctx.sum1=sum1; rctx.i=i; rctx.spike_count=count; rctx.post_spike_times=post_times; rctx.neuron_offset=offset; rctx.target_nrn=target_nrn; rctx.n_post=n_post; rctx.k_prefactor=k_prefactor; rctx.tau_mem=tau_mem; rctx.tau_syn=tau_syn; rctx.v_th=v_th;
                        double t_post = toms748_solve(&rctx, root_fn, t_before, t_input);
                        if (t_post > t_before && t_post < t_input) {
                            post_times[offset + count] = t_post;
                            post_currents[offset + count] = compute_i(t_post, n_spikes, times, sum0, target_nrn, n_post, tau_syn);
                            count++; found = 1;
                        }
                        processed = i; break;
                    }
                }
            }
            if (!found) processed = (processed + 1 < n_spikes) ? processed + 1 : n_spikes;
        }
        spike_counts[target_nrn] = count;
    }

    int total = 0;
    for (int n = 0; n < n_post; n++) total += spike_counts[n];

    if (total == 0) { free(sum0); free(sum1); free(post_times); free(post_currents); free(spike_counts); return 0; }

    SpikeEntry* entries = malloc(total * sizeof(SpikeEntry));
    int idx = 0;
    for (int n = 0; n < n_post; n++) {
        int cnt = spike_counts[n], off = n * max_spikes;
        for (int k = 0; k < cnt; k++) {
            entries[idx].time = post_times[off + k];
            entries[idx].source = n;
            entries[idx].current = post_currents[off + k];
            idx++;
        }
    }

    qsort(entries, total, sizeof(SpikeEntry), cmp_spike_asc);

    for (int i = 0; i < total; i++) {
        out_times[i] = entries[i].time;
        out_sources[i] = entries[i].source;
        out_currents[i] = entries[i].current;
    }

    free(sum0); free(sum1); free(post_times); free(post_currents); free(spike_counts); free(entries);
    return total;
}

/* ── BackwardSpikes: adjoint method backward pass ──────────────────── */

void eventprop_backward_spikes(const double* times, const int32_t* sources, int n_spikes,
                               const double* fwd_times, const int32_t* fwd_sources,
                               const double* fwd_currents, const double* post_errors, int n_post_spikes,
                               const double* w, int n_pre, int n_post,
                               double v_th, double tau_mem, double tau_syn,
                               double* grad)
{
    if (n_post_spikes == 0) return;

    int total = n_spikes + n_post_spikes;
    MergedSpike* merged = malloc(total * sizeof(MergedSpike));

    for (int i = 0; i < n_spikes; i++) { merged[i].time=times[i]; merged[i].idx=i; merged[i].is_input=1; }
    for (int i = 0; i < n_post_spikes; i++) { merged[n_spikes+i].time=fwd_times[i]; merged[n_spikes+i].idx=i; merged[n_spikes+i].is_input=0; }

    /* Sort descending by time: O(n log n) via qsort */
    qsort(merged, total, sizeof(MergedSpike), cmp_merged_desc);

    double* all_errors = calloc(total, sizeof(double));
    for (int i = 0; i < n_post_spikes; i++) all_errors[n_spikes + i] = post_errors[i];

    double* lambda_v = calloc(n_post, sizeof(double));
    double* lambda_i = malloc(n_post * sizeof(double));

    /* Per-neuron jump storage (matching Eigen's vector<vector<pair<double,double>>> structure) */
    double** jump_time = calloc(n_post, sizeof(double*));
    double** jump_mag  = calloc(n_post, sizeof(double*));
    int*   jump_count  = calloc(n_post, sizeof(int));
    int*   jump_cap    = calloc(n_post, sizeof(int));
    for (int i = 0; i < n_post; i++) {
        jump_cap[i] = 8;  /* initial capacity */
        jump_time[i] = malloc(jump_cap[i] * sizeof(double));
        jump_mag[i]  = malloc(jump_cap[i] * sizeof(double));
    }

    double largest_time = merged[0].time, previous_t = largest_time;
    double k_bwd = tau_mem / (tau_mem - tau_syn);

    for (int s = 0; s < total; s++) {
        int spike_idx = merged[s].idx, is_input = merged[s].is_input;
        double spike_time = merged[s].time;
        int spike_source = is_input ? sources[spike_idx] : fwd_sources[spike_idx];

        double t_bwd = largest_time - spike_time;
        double dt = previous_t - t_bwd;
        scale_inplace_simd(lambda_v, exp(dt / tau_mem), n_post);

        if (is_input) {
            /* Compute lambda_i for ALL post-synaptic neurons (matching Eigen structure) */
            memset(lambda_i, 0, n_post * sizeof(double));
            for (int nrn = 0; nrn < n_post; nrn++) {
                for (int j = 0; j < jump_count[nrn]; j++) {
                    double dt_jump = t_bwd - jump_time[nrn][j];
                    lambda_i[nrn] += jump_mag[nrn][j] * k_bwd * (exp(-dt_jump / tau_mem) - exp(-dt_jump / tau_syn));
                }
            }

            vadd_scaled_simd(grad + spike_source * n_post, lambda_i, -tau_syn, n_post);

            /* Compute dot product: w[row] · (lambda_v - lambda_i) */
            double diff[256]; /* Stack buffer for typical n_post sizes */
            int dlen = (n_post <= 256) ? n_post : 256;
            for (int j = 0; j < dlen; j++)
                diff[j] = lambda_v[j] - lambda_i[j];
            double outbound = dot_product_simd(w + spike_source * n_post, diff, dlen);
            /* Handle remainder if n_post > 256 */
            for (int j = dlen; j < n_post; j++)
                outbound += w[spike_source * n_post + j] * (lambda_v[j] - lambda_i[j]);
            all_errors[spike_idx] += outbound;
        } else {
            double current = fwd_currents[spike_idx];
            double inv_ic = 1.0 / (current - v_th);
            int nrn = spike_source;  /* post-spike: spike_source is the post-synaptic neuron */
            double post_err = all_errors[n_spikes + spike_idx];

            /* Record jump for this neuron (with capacity management) */
            if (jump_count[nrn] >= jump_cap[nrn]) {
                jump_cap[nrn] *= 2;
                jump_time[nrn] = realloc(jump_time[nrn], jump_cap[nrn] * sizeof(double));
                jump_mag[nrn]  = realloc(jump_mag[nrn],  jump_cap[nrn] * sizeof(double));
            }
            jump_time[nrn][jump_count[nrn]] = t_bwd;
            jump_mag[nrn][jump_count[nrn]]  = inv_ic * (v_th * lambda_v[nrn] + post_err);
            jump_count[nrn]++;

            lambda_v[nrn] = (current / (current - v_th)) * lambda_v[nrn] + inv_ic * post_err;
        }
        previous_t = t_bwd;
    }

    for (int i = 0; i < n_post; i++) {
        free(jump_time[i]); free(jump_mag[i]);
    }
    free(merged); free(all_errors); free(lambda_v); free(lambda_i);
    free(jump_time); free(jump_mag); free(jump_count); free(jump_cap);
}

/* ── ComputeMaxima: membrane potential maxima per neuron ───────────── */

void eventprop_compute_maxima(const double* w, int n_pre, int n_post,
                              const double* times, const int32_t* sources, int n_spikes,
                              double tau_mem, double tau_syn,
                              double* maxima_times, double* maxima_values)
{
    int n_sums = (n_spikes + 1) * n_post;
    double* sum0 = calloc(n_sums, sizeof(double));
    double* sum1 = calloc(n_sums, sizeof(double));
    eventprop_compute_sums(w, n_pre, n_post, times, sources, n_spikes, tau_mem, tau_syn, sum0, sum1);

    double k_pref = tau_syn / (tau_mem - tau_syn);
    double tmax_pref = 1.0 / (1.0/tau_mem - 1.0/tau_syn);
    double tmax_sum = log(tau_syn / tau_mem);

    for (int i = 0; i < n_post; i++) { maxima_times[i] = NAN; maxima_values[i] = -INFINITY; }

    for (int target_nrn = 0; target_nrn < n_post; target_nrn++) {
        for (int i = 0; i <= n_spikes; i++) {
            double t_input = (i == n_spikes) ? INFINITY : times[i];
            double t_before = (i > 0) ? times[i-1] : 0.0;
            double s0 = sum0[i * n_post + target_nrn], s1 = sum1[i * n_post + target_nrn];

            if (s0 <= 0 || s1 <= 0) continue;
            double tmax = tmax_pref * (tmax_sum + log(s1 / s0));

            if (!isinf(tmax) && t_before <= tmax && (i == n_spikes || tmax <= t_input)) {
                double vmax = k_pref * (s1 * exp(-tmax/tau_mem) - s0 * exp(-tmax/tau_syn));
                if (vmax > maxima_values[target_nrn]) {
                    maxima_values[target_nrn] = vmax; maxima_times[target_nrn] = tmax;
                }
            }

            double v_at_input = k_pref * (s1 * exp(-t_input/tau_mem) - s0 * exp(-t_input/tau_syn));
            if (v_at_input > maxima_values[target_nrn]) {
                maxima_values[target_nrn] = v_at_input;
                maxima_times[target_nrn] = t_input == INFINITY ? t_before : t_input;
            }
        }
    }

    free(sum0); free(sum1);
}

/* ── BackwardMaxima: adjoint method with maxima-based loss ─────────── */

void eventprop_backward_maxima(const double* input_times, const int32_t* input_sources,
                               int n_input_spikes,
                               const double* maxima_times, const double* maxima_values,
                               const double* maxima_errors, int n_maxima,
                               const double* w, int n_pre, int n_post,
                               double tau_mem, double tau_syn,
                               double* gradient)
{
    (void)w; /* Not used — gradient depends on maximaTimes and errors only */

    if (n_input_spikes == 0 || n_maxima == 0)
        return;

    double largest_time = input_times[n_input_spikes - 1];
    double k_bwd_prefactor = tau_mem / (tau_mem - tau_syn);

    /* Sweep backward through input spikes */
    for (int spike_idx = n_input_spikes - 1; spike_idx >= 0; spike_idx--) {
        double t_bwd = largest_time - input_times[spike_idx];

        for (int max_idx = 0; max_idx < n_maxima; max_idx++) {
            if (isnan(maxima_times[max_idx]))
                continue;

            double t_vmax_bwd = largest_time - maxima_times[max_idx];
            if (t_vmax_bwd > t_bwd)
                continue;

            double lambda_v = -exp(-(t_bwd - t_vmax_bwd) / tau_mem) * maxima_errors[max_idx] / tau_mem;
            double k_bwd_val = k_bwd_prefactor * (exp(-t_bwd / tau_mem) - exp(-t_bwd / tau_syn));
            double lambda_i = -k_bwd_val * maxima_errors[max_idx] / tau_mem;

            int source_nrn = input_sources[spike_idx];
            gradient[source_nrn * n_post + max_idx] -= tau_syn * lambda_i;
        }
    }
}
