/**
 * @file eventprop.c
 * @brief EventProp: Event-based backpropagation for SNN training.
 *
 * Consolidated implementation with automatic memory allocation (alloc versions).
 * Includes legacy diagnostic functions for compatibility.
 */

#include "eventprop.h"
#include "matrixview.inc"  /* inline MatrixView helpers */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <errno.h>
#include <limits.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ── SIMD detection and intrinsics ─────────────────────────────────────── */
#ifndef EVENTPROP_SIMD_AVAILABLE
#define EVENTPROP_SIMD_AVAILABLE 1
#endif

#if defined(__AVX2__) || (defined(__x86_64__) && EVENTPROP_SIMD_AVAILABLE)
#define HAVE_AVX2 1
#include <immintrin.h>
#endif

#if defined(__aarch64__) || (defined(__arm__) && EVENTPROP_SIMD_AVAILABLE)
#define HAVE_NEON 1
#include <arm_neon.h>
#endif

#define EXP_STACK_SIZE 1024
#define DIFF_STACK_SIZE 512

#define TOMS748_MAX_ITER 1024
#define TOMS748_TOL 1e-14


/* ── Input validation ──────────────────────────────────────── */

EventPropError eventprop_validate_input(const EventPropInput input)
{
    /* ── Dimension checks ─────────────────────────────────────── */
    if (input.n_pre <= 0 || input.n_post <= 0)
        return EVENTPROP_INVALID_DIMENSION;
    if (input.n_pre > EVENTPROP_MAX_DIM || input.n_post > EVENTPROP_MAX_DIM)
        return EVENTPROP_DIMENSION_TOO_LARGE;

    /* ── Spike count ──────────────────────────────────────────── */
    if (input.n_spikes < 0)
        return EVENTPROP_NEGATIVE_SPIKES;

    /* ── Tau bounds and proximity ─────────────────────────────── */
    if (input.tau_mem <= 0.0 || input.tau_syn <= 0.0)
        return EVENTPROP_INVALID_TAU;
    if (input.tau_mem < EVENTPROP_MIN_TAU || input.tau_syn < EVENTPROP_MIN_TAU)
        return EVENTPROP_TAU_TOO_SMALL;
    if (input.tau_mem > EVENTPROP_MAX_TAU || input.tau_syn > EVENTPROP_MAX_TAU)
        return EVENTPROP_TAU_TOO_LARGE;
    if (input.tau_mem == input.tau_syn)
        return EVENTPROP_INVALID_TAU;

    /* Relative difference: |tau_mem - tau_syn| / min(tau) < 1e-6 → k_prefactor blows up */
    {
        double min_tau = fmin(input.tau_mem, input.tau_syn);
        double rel_diff = fabs(input.tau_mem - input.tau_syn) / min_tau;
        if (rel_diff < EVENTPROP_TAU_REL_DIFF_MIN)
            return EVENTPROP_TAU_TOO_CLOSE;
    }

    /* ── Threshold voltage ────────────────────────────────────── */
    if (input.v_th <= 0.0)
        return EVENTPROP_INVALID_VTH;

    /* ── Pointer checks ───────────────────────────────────────── */
    if (input.w == NULL)
        return EVENTPROP_NULL_POINTER;
    if (input.n_spikes > 0) {
        if (input.times == NULL || input.sources == NULL)
            return EVENTPROP_NULL_POINTER;
    } else {
        /* n_spikes == 0: times/sources may be NULL, nothing to iterate */
        return EVENTPROP_OK;
    }

    /* ── Per-spike checks (single pass) ──────────────────────── */
    double max_time = -1.0;   /* sentinel: will become >= 0 on first spike */
    for (int i = 0; i < input.n_spikes; i++) {
        if (input.sources[i] < 0 || input.sources[i] >= input.n_pre)
            return EVENTPROP_INVALID_SOURCE;

        double t = input.times[i];
        if (t < 0.0)
            return EVENTPROP_NEGATIVE_SPIKE_TIME;
        if (t > EVENTPROP_MAX_SPIKE_TIME)
            return EVENTPROP_EXCESSIVE_SPIKE_TIME;
        if (max_time < 0.0 || t > max_time)
            max_time = t;
    }

    /* ── Cross-field: times vs tau overflow check ─────────────── */
    /* exp(max_time / min_tau) must stay finite (< INF at ~709.78). */
    {
        double min_tau = fmin(input.tau_mem, input.tau_syn);
        if (max_time > 0.0 && max_time / min_tau >= EVENTPROP_EXP_OVERFLOW_THRESHOLD)
            return EVENTPROP_TIMES_TAU_MISMATCH;
    }

    return EVENTPROP_OK;
}

/* ── Inline helpers ───────────────────────────────────────────────────── */

// Inline functions replacing GCC statement expressions ({ ... }) for MSVC compatibility.
// compute_v_inline — direct evaluation, no precomputed exponentials.
static inline double compute_v_inline(double t, int i, int spike_count,
    const double* post_spike_times, int neuron_offset,
    const double* sum0, const double* sum1, int target_nrn,
    const EventPropInput input, double k_prefactor)
{
    double mem = k_prefactor * (sum1[i * input.n_post + target_nrn] * exp(-t / input.tau_mem)
                    - sum0[i * input.n_post + target_nrn] * exp(-t / input.tau_syn));
    for (int s = 0; s < spike_count; s++) {
        double st = post_spike_times[neuron_offset + s];
        if (st <= t) {  /* Include all spikes where st <= t, don't break early */
            mem -= input.v_th * exp(-(t - st) / input.tau_mem);
        }
    }
    return mem;
}

// compute_v_factored — uses precomputed exp(st*tau-1) to share one exp() call.
static inline double compute_v_factored(double t, int i, int spike_count,
    const double* post_spike_times, int neuron_offset,
    const double* sum0, const double* sum1, int target_nrn, int n_post,
    double k_prefactor, double tau_mem_inv, double tau_syn_inv,
    double v_th, const double* post_spike_exp_st)
{
    double e_t = exp(-(t) * tau_mem_inv);
    double mem = k_prefactor * (sum1[i * n_post + target_nrn] * e_t
                    - sum0[i * n_post + target_nrn] * exp(-(t) * tau_syn_inv));
    if (post_spike_exp_st) {
        for (int s = 0; s < spike_count; s++) {
            double st = post_spike_times[neuron_offset + s];
            if (st <= t) {
                mem -= v_th * e_t * post_spike_exp_st[s];
            }
        }
    } else {
        for (int s = 0; s < spike_count; s++) {
            double st = post_spike_times[neuron_offset + s];
            if (st <= t) {
                mem -= v_th * exp(-(t - st) * tau_mem_inv);
            }
        }
    }
    return mem;
}

typedef struct RootContext {
    const double* sum0; const double* sum1;
    int i, spike_count, target_nrn, n_post;
    const double* post_spike_times; int neuron_offset;
    double k_prefactor, tau_mem_inv, tau_syn_inv, v_th;
    /* Precomputed: exp(post_spike_times[s] * tau_mem_inv) for each spike.
     * Factorization: exp(-(t-st)*tau-1) = exp(-t*tau-1) × exp(st*tau-1)
     * Eliminates N exp() calls per iteration → just 1 (for base terms). */
    const double* post_spike_exp_st;   /* NULL if not precomputed */
} RootContext;

typedef struct SpikeEntry {
     double time; 
     int source; 
     double current; 
} SpikeEntry;

/* Merge spike comparison for qsort (descending by time) */
typedef struct MergedSpike {
     double time; 
     int idx; 
     int is_input; 
} MergedSpike;


void eventprop_free_spike_output(SpikeOutput* output)
{
    if (!output) return;
    free(output->times);
    free(output->sources);
    free(output->currents);
    free(output->errors);
    output->times = NULL;
    output->sources = NULL;
    output->currents = NULL;
    output->errors = NULL;
    output->n_spikes = 0;
}
void eventprop_free_sums_output(SumsOutput* output)
{
    if (!output) return;
    free(output->sum0);
    free(output->sum1);
    output->sum0 = NULL;
    output->sum1 = NULL;
}
void eventprop_free_maxima_output(MaximaOutput* output)
{
    if (!output) return;
    free(output->times);
    free(output->values);
    free(output->errors);
    output->times = NULL;
    output->values = NULL;
    output->errors = NULL;
}
void eventprop_free_backward_output(BackwardOutput* output){
    if (!output) return;
    free(output->grad);
    free(output->spike_errors);
    output->grad = NULL;
    output->spike_errors = NULL;
}
/* ── SIMD helper functions ───────────────────────────────────── */

static int simd_vec_width(void)
{
#ifdef HAVE_AVX512F
    return 8;
#elif defined(HAVE_AVX2)
    return 4;
#elif defined(__ARM_NEON)
    return 2;
#else
    return 0;
#endif
}

static int simd_available(void)
{
    return simd_vec_width() > 0;
}

/* Fused multiply-add: dest[j] = src[j] + scalar * weights[j] */
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
    {
        float64x2_t sc = vdupq_n_f64(scalar);
        for (int i = 0; i < vec_count; i++) {
            float64x2_t s = vld1q_f64(src + i * vec);
            float64x2_t w = vld1q_f64(weights + i * vec);
            vst1q_f64(dest + i * vec, vaddq_f64(s, vmulq_f64(w, sc)));
        }
    }
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
    {
        float64x2_t sc = vdupq_n_f64(scalar);
        for (int i = 0; i < vec_count; i++) {
            float64x2_t s = vld1q_f64(span + i * vec);
            vst1q_f64(span + i * vec, vmulq_f64(s, sc));
        }
    }
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
#if defined(HAVE_AVX512F)
        __m512d sc = _mm512_set1_pd(scalar);
        for (int i = 0; i < vec_count; i++) {
            __m512d d = _mm512_loadu_pd(dest + i * vec);
            __m512d s = _mm512_loadu_pd(src + i * vec);
            _mm512_storeu_pd(dest + i * vec, _mm512_add_pd(d, _mm512_mul_pd(s, sc)));
        }
#else
        __m256d sc = _mm256_set1_pd(scalar);
        for (int i = 0; i < vec_count; i++) {
            __m256d d = _mm256_loadu_pd(dest + i * vec);
            __m256d s = _mm256_loadu_pd(src + i * vec);
            d = _mm256_add_pd(d, _mm256_mul_pd(s, sc));
            _mm256_storeu_pd(dest + i * vec, d);
        }
#endif
    }
#elif defined(__ARM_NEON)
    {
        float64x2_t sc = vdupq_n_f64(scalar);
        for (int i = 0; i < vec_count; i++) {
            float64x2_t d = vld1q_f64(dest + i * vec);
            float64x2_t s = vld1q_f64(src + i * vec);
            vst1q_f64(dest + i * vec, vaddq_f64(d, vmulq_f64(s, sc)));
        }
    }
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
#if defined(HAVE_AVX512F)
        __m512d acc = _mm512_setzero_pd();
        for (int i = 0; i < vec_count; i++) {
            __m512d va = _mm512_loadu_pd(a + i * vec);
            __m512d vb = _mm512_loadu_pd(b + i * vec);
            acc = _mm512_add_pd(acc, _mm512_mul_pd(va, vb));
        }
        double result[8];
        _mm512_storeu_pd(result, acc);
        double sum = result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] + result[7];
        for (int j = vec_count * vec; j < n; j++)
            sum += a[j] * b[j];
        return sum;
#else
        __m256d acc = _mm256_setzero_pd();
        for (int i = 0; i < vec_count; i++) {
            __m256d va = _mm256_loadu_pd(a + i * vec);
            __m256d vb = _mm256_loadu_pd(b + i * vec);
            acc = _mm256_add_pd(acc, _mm256_mul_pd(va, vb));
        }
        double result[4];
        _mm256_storeu_pd(result, acc);
        double sum = result[0] + result[1] + result[2] + result[3];
        for (int j = vec_count * vec; j < n; j++)
            sum += a[j] * b[j];
        return sum;
#endif
    }
#elif defined(__ARM_NEON)
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
#endif

    double sum = 0.0;
    for (int j = 0; j < n; j++)
        sum += a[j] * b[j];
    return sum;
}

/* Vector subtraction: dest[j] = a[j] - b[j] */
static void vsub_simd(double* dest, const double* a, const double* b, int n)
{
    if (!simd_available() || n < 2) {
        for (int j = 0; j < n; j++)
            dest[j] = a[j] - b[j];
        return;
    }

    int vec = simd_vec_width();
    int vec_count = n / vec;

#if defined(HAVE_AVX512F) || defined(HAVE_AVX2)
    {
        for (int i = 0; i < vec_count; i++) {
#if defined(HAVE_AVX512F)
            __m512d va = _mm512_loadu_pd(a + i * vec);
            __m512d vb = _mm512_loadu_pd(b + i * vec);
            _mm512_storeu_pd(dest + i * vec, _mm512_sub_pd(va, vb));
#else
            __m256d va = _mm256_loadu_pd(a + i * vec);
            __m256d vb = _mm256_loadu_pd(b + i * vec);
            _mm256_storeu_pd(dest + i * vec, _mm256_sub_pd(va, vb));
#endif
        }
    }
#elif defined(__ARM_NEON)
    {
        for (int i = 0; i < vec_count; i++) {
            float64x2_t va = vld1q_f64(a + i * vec);
            float64x2_t vb = vld1q_f64(b + i * vec);
            vst1q_f64(dest + i * vec, vsubq_f64(va, vb));
        }
    }
#endif

    for (int j = vec_count * vec; j < n; j++)
        dest[j] = a[j] - b[j];
}

static int cmp_merged_desc(const void* a, const void* b) {
    double ta = ((const MergedSpike*)a)->time;
    double tb = ((const MergedSpike*)b)->time;
    return (ta < tb) - (ta > tb);
}

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

static double compute_i(double t, int n_spikes, const double* times,
                        const double* sum0, int target_nrn, int n_post, double tau_syn)
{
    int idx = 0;
    for (; idx < n_spikes; idx++) {
        if (times[idx] > t) break;
    }
    return sum0[idx * n_post + target_nrn] * exp(-t / tau_syn);
}

static int cmp_spike_asc(const void* a, const void* b) {
    double ta = ((const SpikeEntry*)a)->time;
    double tb = ((const SpikeEntry*)b)->time;
    return (ta > tb) - (ta < tb);
}

/* Inline voltage computation — eliminates callback indirection */
#define VOLTAGE(c, t) \
    (compute_v_factored(t, (c)->i, (c)->spike_count, (c)->post_spike_times, (c)->neuron_offset, \
                 (c)->sum0, (c)->sum1, (c)->target_nrn, (c)->n_post, \
                 (c)->k_prefactor, (c)->tau_mem_inv, (c)->tau_syn_inv, (c)->v_th, \
                 (c)->post_spike_exp_st) - (c)->v_th)

static double toms748_solve(const RootContext* ctx, double a, double b) {

    double fa = VOLTAGE(ctx, a), fb = VOLTAGE(ctx, b);
    if (!isfinite(fa) || !isfinite(fb)) return a;

    /* Precondition: bracket must contain sign change */
    if (fa * fb > 0) {
        return NAN;
    }

    if (a > b) { double tmp=a; a=b; b=tmp; tmp=fa; fa=fb; fb=tmp; }

    double c = a, fc = fa, d = 0.0;
    int iter = 0;
    while (++iter <= TOMS748_MAX_ITER) {
        double m = 0.5 * (b - a);
        if (fabs(m) <= TOMS748_TOL || fb == 0.0)
            return b;

        double s = 0.0;
        int useBisection = 1;

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

        if (useBisection) s = a + m;

        double f_oldB = fb;   // save before updating - avoids redundant fn call below
        double prevB = b; b = s; fb = VOLTAGE(ctx, b);
        if (!isfinite(fb)) return prevB;

        if (fabs(fb) <= TOMS748_TOL || fabs(b - prevB) <= TOMS748_TOL * fmax(1.0, fabs(prevB)))
            return b;

        /* Reuse known value instead of re-evaluating fn(ctx, prevB) */
        if (fa * fb < 0.0) { c = prevB; fc = f_oldB; }
        else if (fb * fc < 0.0) { a = prevB; fa = f_oldB; }
        else { a = prevB; fa = f_oldB; }

        if (!isfinite(fa) || !isfinite(fc)) return prevB;
        d = b - prevB;
    }
    return b;
}

SpikeOutput eventprop_compute_spikes(const EventPropInput input)
{
    SpikeOutput result;

    result.error_no = EVENTPROP_OK;   /* default: success */

    EventPropError err = eventprop_validate_input(input);
    if (err != EVENTPROP_OK) {
        result.n_spikes = 0;
        result.times    = NULL;
        result.sources  = NULL;
        result.currents = NULL;
        result.errors   = NULL;
        result.error_no = err;
        return result;
    }

    /* Use dynamic arrays internally, then copy to exact-size output */
    double k_prefactor = input.tau_syn / (input.tau_mem - input.tau_syn);
    double tmax_prefactor = 1.0 / (1.0 / input.tau_mem - 1.0 / input.tau_syn);
    double tmax_summand = log(input.tau_syn / input.tau_mem);
    double tau_mem_inv = 1.0 / input.tau_mem;
    double tau_syn_inv = 1.0 / input.tau_syn;

    /* Compute sums */
    SumsOutput sums = eventprop_compute_sums(input);

    if (sums.error_no != EVENTPROP_OK) {
        result.n_spikes = 0;
        result.times    = NULL;
        result.sources  = NULL;
        result.currents = NULL;
        result.errors   = NULL;
        result.error_no = sums.error_no;   /* propagate from compute_sums */
        return result;
    }

    if (!sums.sum0 || !sums.sum1) {
        result.n_spikes = 0;
        result.times    = NULL;
        result.sources  = NULL;
        result.currents = NULL;
        result.errors   = NULL;
        result.error_no = sums.error_no;

        return result;
    }

    const double* sum0 = sums.sum0;
    const double* sum1 = sums.sum1;

    /* Per-neuron dynamic buffers - growable to handle burst patterns */
    double** post_times_arr   = malloc(input.n_post * sizeof(double*));
    if (!post_times_arr) goto fwd_alloc_fail;
    double** post_currents_arr = malloc(input.n_post * sizeof(double*));
    if (!post_currents_arr) { free(post_times_arr); goto fwd_alloc_fail; }
    int* spike_caps  = calloc(input.n_post, sizeof(int));
    if (!spike_caps) { free(post_times_arr); free(post_currents_arr); goto fwd_alloc_fail; }
    int* spike_counts = calloc(input.n_post, sizeof(int));
    if (!spike_counts) { free(post_times_arr); free(post_currents_arr); free(spike_caps); goto fwd_alloc_fail; }

    /* Start with 3× estimated spikes to reduce realloc overhead */
    int initial_cap = input.n_spikes * 3 + 1;
    for (int n = 0; n < input.n_post; n++) {
        post_times_arr[n]   = calloc(initial_cap, sizeof(double));
        if (!post_times_arr[n]) goto fwd_alloc_fail_partial_neuron;
        post_currents_arr[n] = calloc(initial_cap, sizeof(double));
        if (!post_currents_arr[n]) { free(post_times_arr[n]); goto fwd_alloc_fail_partial_neuron; }
        spike_caps[n]       = initial_cap;
    }

    /* Shared error flag for OpenMP region: threads bail out gracefully on OOM */
#ifdef _OPENMP
    volatile int fwd_alloc_error = 0;
#endif
    int target_nrn;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (target_nrn = 0; target_nrn < input.n_post; target_nrn++) {
        if (target_nrn == 0) {
        }

        double* p_times   = post_times_arr[target_nrn];
        double* p_currents = post_currents_arr[target_nrn];
        int cap = spike_caps[target_nrn];
        int count = 0;
        int finished = 0, processed = 0;

        while (!finished) {
            int found = 0;
            for (int i = processed; i <= input.n_spikes; i++) {
                if (i == input.n_spikes) {
                    /* Last interval */
                    double s0 = sum0[input.n_spikes * input.n_post + target_nrn];
                    double s1 = sum1[input.n_spikes * input.n_post + target_nrn];
                    for (int s = 0; s < count; s++)
                        s1 += -input.v_th * exp(p_times[s] / input.tau_mem);

                    double tmax = tmax_prefactor * (tmax_summand + log(s1 / s0));
                    if (!isinf(tmax)) {
                        double v = compute_v_inline(tmax, i, count, p_times, 0, sum0, sum1,
                                             target_nrn, input, k_prefactor);
                        if (v > input.v_th + 1e-14) {
                            double t_lower = count > 0 ? p_times[count - 1] : 0.0;
                            RootContext rctx; memset(&rctx, 0, sizeof(rctx));
                            /* Precompute exp(st * tau_mem_inv) for each post-spike → factorization */
                            double* post_exp_st = NULL;
                            if (count > 0) {
                                post_exp_st = (double*)malloc(count * sizeof(double));
                                if (!post_exp_st) { fwd_alloc_error = 1; finished = 1; break; }
                                for (int s = 0; s < count; s++)
                                    post_exp_st[s] = exp(p_times[s] * tau_mem_inv);
                            }
                            rctx.sum0=sum0; rctx.sum1=sum1; rctx.i=i; rctx.spike_count=count;
                            rctx.post_spike_times=p_times; rctx.neuron_offset=0;
                            rctx.target_nrn=target_nrn; rctx.n_post=input.n_post;
                            rctx.k_prefactor=k_prefactor; rctx.tau_mem_inv=tau_mem_inv;
                            rctx.tau_syn_inv=tau_syn_inv; rctx.v_th=input.v_th;
                            rctx.post_spike_exp_st = post_exp_st;
                            double t_post = toms748_solve(&rctx, t_lower, tmax);
                            free(post_exp_st);
                            if (t_post > t_lower && t_post < tmax) {
                                if (count >= cap) {
                                    cap *= 3;
                                    double* new_times    = realloc(p_times, cap * sizeof(double));
                                    double* new_currents = realloc(p_currents, cap * sizeof(double));
                                    if (!new_times || !new_currents) { free(new_times); free(new_currents); fwd_alloc_error = 1; finished = 1; break; }
                                    p_times    = new_times;
                                    p_currents = new_currents;
                                }
                                p_times[count]   = t_post;
                                p_currents[count] = compute_i(t_post, input.n_spikes, input.times, sum0,
                                                              target_nrn, input.n_post, input.tau_syn);
                                count++; found = 1;
                            }
                        }
                    }
                    if (!found) finished = 1;
                    else processed = input.n_spikes;
                } else {
                    double t_input = input.times[i];
                    double tmax = compute_tmax(i, input.n_spikes, input.times, count, p_times, 0,
                                               sum0, sum1, target_nrn, input.n_post,
                                               tmax_prefactor, tmax_summand, input.tau_mem, input.v_th);

                    if (!isinf(tmax)) {
                        double v = compute_v_inline(tmax, i, count, p_times, 0, sum0, sum1,
                                             target_nrn, input, k_prefactor);
                        if (v > input.v_th + 1e-14) {
                            double t_before = (i > 0) ? input.times[i-1] : 0.0;
                            for (int s = 0; s < count; s++) {
                                double st = p_times[s];
                                if (st > t_input) break;
                                if (st > t_before) t_before = st;
                            }
                            RootContext rctx; memset(&rctx, 0, sizeof(rctx));
                            /* Precompute exp(st * tau_mem_inv) for each post-spike → factorization */
                            double* post_exp_st = NULL;
                            if (count > 0) {
                                post_exp_st = (double*)malloc(count * sizeof(double));
                                if (!post_exp_st) { fwd_alloc_error = 1; finished = 1; break; }
                                for (int s = 0; s < count; s++)
                                    post_exp_st[s] = exp(p_times[s] * tau_mem_inv);
                            }
                            rctx.sum0=sum0; rctx.sum1=sum1; rctx.i=i; rctx.spike_count=count;
                            rctx.post_spike_times=p_times; rctx.neuron_offset=0;
                            rctx.target_nrn=target_nrn; rctx.n_post=input.n_post;
                            rctx.k_prefactor=k_prefactor; rctx.tau_mem_inv=tau_mem_inv;
                            rctx.tau_syn_inv=tau_syn_inv; rctx.v_th=input.v_th;
                            rctx.post_spike_exp_st = post_exp_st;
                            double t_post = toms748_solve(&rctx, t_before, tmax);
                            free(post_exp_st);
                            if (t_post > t_before && t_post < tmax) {
                                if (count >= cap) {
                                    cap *= 3;
                                    double* new_times    = realloc(p_times, cap * sizeof(double));
                                    double* new_currents = realloc(p_currents, cap * sizeof(double));
                                    if (!new_times || !new_currents) { free(new_times); free(new_currents); fwd_alloc_error = 1; finished = 1; break; }
                                    p_times    = new_times;
                                    p_currents = new_currents;
                                }
                                p_times[count]   = t_post;
                                p_currents[count] = compute_i(t_post, input.n_spikes, input.times, sum0,
                                                              target_nrn, input.n_post, input.tau_syn);
                                count++; found = 1;
                            }
                            processed = i; break;
                        }
                    }

                    /* Check at input spike time */
                    double v_at_input = compute_v_inline(t_input, i, count, p_times, 0,
                                                          sum0, sum1, target_nrn, input,
                                                          k_prefactor);
                    if (v_at_input > input.v_th + 1e-14) {
                        double t_before = (i > 0) ? input.times[i-1] : 0.0;
                        for (int s = 0; s < count; s++) {
                            double st = p_times[s];
                            if (st > t_input) break;
                            if (st > t_before) t_before = st;
                        }
                        RootContext rctx; memset(&rctx, 0, sizeof(rctx));
                        /* Precompute exp(st * tau_mem_inv) for each post-spike → factorization */
                        double* post_exp_st = NULL;
                        if (count > 0) {
                            post_exp_st = (double*)malloc(count * sizeof(double));
                            if (!post_exp_st) { fwd_alloc_error = 1; finished = 1; break; }
                            for (int s = 0; s < count; s++)
                                post_exp_st[s] = exp(p_times[s] * tau_mem_inv);
                        }
                        rctx.sum0=sum0; rctx.sum1=sum1; rctx.i=i; rctx.spike_count=count;
                        rctx.post_spike_times=p_times; rctx.neuron_offset=0;
                        rctx.target_nrn=target_nrn; rctx.n_post=input.n_post;
                        rctx.k_prefactor=k_prefactor; rctx.tau_mem_inv=tau_mem_inv;
                        rctx.tau_syn_inv=tau_syn_inv; rctx.v_th=input.v_th;
                        rctx.post_spike_exp_st = post_exp_st;
                        double t_post = toms748_solve(&rctx, t_before, t_input);
                        free(post_exp_st);
                        if (t_post > t_before && t_post < t_input) {
                            if (count >= cap) {
                                cap *= 3;
                                double* new_times    = realloc(p_times, cap * sizeof(double));
                                double* new_currents = realloc(p_currents, cap * sizeof(double));
                                if (!new_times || !new_currents) { free(new_times); free(new_currents); fwd_alloc_error = 1; finished = 1; break; }
                                p_times    = new_times;
                                p_currents = new_currents;
                            }
                            p_times[count]   = t_post;
                            p_currents[count] = compute_i(t_post, input.n_spikes, input.times, sum0,
                                                          target_nrn, input.n_post, input.tau_syn);
                            count++; found = 1;
                        }
                        processed = i; break;
                    }
                }
            }
            if (!found) processed = (processed + 1 < input.n_spikes) ? processed + 1 : input.n_spikes;
        }

        /* Write back updated pointers */
        post_times_arr[target_nrn]   = p_times;
        post_currents_arr[target_nrn] = p_currents;
        spike_caps[target_nrn]       = cap;
        spike_counts[target_nrn]     = count;
    }


    /* Count total spikes and allocate exact-size output */
    result.n_spikes = 0;
    for (int n = 0; n < input.n_post; n++)
        result.n_spikes += spike_counts[n];

#ifdef _OPENMP
    if (fwd_alloc_error) goto fwd_alloc_fail;
#endif

    if (result.n_spikes == 0) {
        result.times = NULL;
        result.sources = NULL;
        result.currents = NULL;
        result.errors = NULL;

        for (int n = 0; n < input.n_post; n++) {
            free(post_times_arr[n]);
            free(post_currents_arr[n]);
        }
        eventprop_free_sums_output(&sums);
        free(post_times_arr); free(post_currents_arr);
        free(spike_caps); free(spike_counts);
        return result;
    }

    /* Collect all spikes into sortable array */
    SpikeEntry* entries = malloc(result.n_spikes * sizeof(SpikeEntry));
    if (!entries) goto fwd_alloc_fail;
    int idx = 0;
    for (int n = 0; n < input.n_post; n++) {
        int cnt = spike_counts[n];
        double* pt   = post_times_arr[n];
        double* pc   = post_currents_arr[n];
        for (int k = 0; k < cnt; k++) {
            entries[idx].time    = pt[k];
            entries[idx].source  = n;
            entries[idx].current = pc[k];
            idx++;
        }
    }

    /* Sort by time ascending */
    qsort(entries, result.n_spikes, sizeof(SpikeEntry), cmp_spike_asc);

    /* Allocate and copy to output */
    result.times = malloc(result.n_spikes * sizeof(double));
    if (!result.times) goto fwd_alloc_fail_entries;
    result.sources = malloc(result.n_spikes * sizeof(int));
    if (!result.sources) { free(result.times); goto fwd_alloc_fail_entries; }
    result.currents = malloc(result.n_spikes * sizeof(double));
    if (!result.currents) { free(result.times); free(result.sources); goto fwd_alloc_fail_entries; }
    result.errors = calloc(result.n_spikes, sizeof(double));
    if (!result.errors) { free(result.times); free(result.sources); free(result.currents); goto fwd_alloc_fail_entries; }

    for (int i = 0; i < result.n_spikes; i++) {
        result.times[i]     = entries[i].time;
        result.sources[i]   = entries[i].source;
        result.currents[i]  = entries[i].current;
    }

    /* Cleanup */
    for (int n = 0; n < input.n_post; n++) {
        free(post_times_arr[n]);
        free(post_currents_arr[n]);
    }
    eventprop_free_sums_output(&sums);
    free(post_times_arr); free(post_currents_arr);
    free(spike_caps); free(spike_counts);
    free(entries);

    return result;

fwd_alloc_fail_entries:
    free(entries);
fwd_alloc_fail_partial_neuron:
    /* Free any neuron buffers that were allocated before failure */
    for (int n = 0; n < input.n_post && post_times_arr != NULL; n++) {
        if (post_currents_arr) { free(post_times_arr[n]); free(post_currents_arr[n]); }
    }
fwd_alloc_fail:
    /* Cleanup on allocation failure */
    eventprop_free_sums_output(&sums);
    free(post_times_arr); free(post_currents_arr);
    free(spike_caps); free(spike_counts);
    result.n_spikes = 0;
    result.times    = NULL;
    result.sources  = NULL;
    result.currents = NULL;
    result.errors   = NULL;
    result.error_no = EVENTPROP_ALLOCATION_FAILED;
    return result;
}

SumsOutput eventprop_compute_sums(const EventPropInput input)
{
    SumsOutput result;
    result.error_no = EVENTPROP_OK;   /* default: success */

    EventPropError err = eventprop_validate_input(input);
    if (err != EVENTPROP_OK) {
        result.sum0 = NULL;
        result.sum1 = NULL;
        result.error_no = err;
        return result;
    }

    int n_rows = input.n_spikes + 1;

    /* Check for integer overflow in allocation size */
    if (n_rows > 0 && input.n_post > INT_MAX / n_rows) {
        result.sum0 = NULL;
        result.sum1 = NULL;
        result.error_no = EVENTPROP_ALLOCATION_FAILED;
        return result;
    }

    /* Allocate output - exact size known upfront */
    result.sum0 = calloc(n_rows * input.n_post, sizeof(double));
    result.sum1 = calloc(n_rows * input.n_post, sizeof(double));

    if (!result.sum0 || !result.sum1) {
        free(result.sum0);
        free(result.sum1);
        result.sum0 = NULL;
        result.sum1 = NULL;
        result.error_no = EVENTPROP_ALLOCATION_FAILED;
        return result;
    }

    if (input.n_spikes == 0) return result;

    /* Pre-compute exponential lookup tables */
    double* exp_mem = calloc(input.n_spikes, sizeof(double));
    double* exp_syn = calloc(input.n_spikes, sizeof(double));

    for (int i = 0; i < input.n_spikes; i++) {
        exp_mem[i] = exp(input.times[i] / input.tau_mem);
        exp_syn[i] = exp(input.times[i] / input.tau_syn);
    }

    /* Accumulate sums using SIMD-accelerated FMA */
    MutableMatrixView s0 = mv_mut(result.sum0, n_rows, input.n_post);
    MutableMatrixView s1 = mv_mut(result.sum1, n_rows, input.n_post);
    MatrixView w         = mv(input.w, input.n_pre, input.n_post);

    for (int i = 0; i < input.n_spikes; i++) {
        int src = input.sources[i];
        fma_row_simd(mv_row_mut(s0, i + 1), mv_row_mut(s0, i), mv_row(w, src), exp_syn[i], input.n_post);
        fma_row_simd(mv_row_mut(s1, i + 1), mv_row_mut(s1, i), mv_row(w, src), exp_mem[i], input.n_post);
    }

    free(exp_mem);
    free(exp_syn);

    return result;
}

MaximaOutput eventprop_compute_maxima(const EventPropInput input)
{
    MaximaOutput result;
    result.error_no = EVENTPROP_OK;   /* default: success */

    EventPropError err = eventprop_validate_input(input);
    if (err != EVENTPROP_OK) {
        result.times   = NULL;
        result.values  = NULL;
        result.errors  = NULL;
        result.error_no = err;
        return result;
    }

    /* Allocate output - always exactly n_post elements */
    result.times   = calloc(input.n_post, sizeof(double));
    result.values  = calloc(input.n_post, sizeof(double));
    result.errors  = calloc(input.n_post, sizeof(double));

    if (!result.times || !result.values || !result.errors) {
        free(result.times);
        free(result.values);
        free(result.errors);
        result.times   = NULL;
        result.values  = NULL;
        result.errors  = NULL;
        result.error_no = EVENTPROP_ALLOCATION_FAILED;
        return result;
    }

    /* Initialize maxima to "not found" */
    for (int i = 0; i < input.n_post; i++) {
        result.times[i]   = NAN;
        result.values[i]  = 0.0;
    }

    /* Use our own alloc function */
    SumsOutput sums = eventprop_compute_sums(input);

    if (sums.error_no != EVENTPROP_OK) {
        free(result.times);
        free(result.values);
        free(result.errors);
        result.times   = NULL;
        result.values  = NULL;
        result.errors  = NULL;
        result.error_no = sums.error_no;   /* propagate from compute_sums */
        return result;
    }

    if (!sums.sum0 || !sums.sum1) {
        free(result.times);
        free(result.values);
        free(result.errors);
        result.times   = NULL;
        result.values  = NULL;
        result.errors  = NULL;
        result.error_no = EVENTPROP_ALLOCATION_FAILED;
        return result;
    }

    int n_rows = input.n_spikes + 1;
    double* sum0 = sums.sum0;
    double* sum1 = sums.sum1;

    /* Find maxima for each post-synaptic neuron */
    double k_pref = input.tau_syn / (input.tau_mem - input.tau_syn);
    double tmax_pref = 1.0 / (1.0/input.tau_mem - 1.0/input.tau_syn);
    double tmax_sum = log(input.tau_syn / input.tau_mem);

    /* No spikes → nothing to compute, return NaN/-INF (matches Eigen) */
    if (input.n_spikes == 0) {
        eventprop_free_sums_output(&sums);
        return result;
    }

    MatrixView s0 = mv(sum0, n_rows, input.n_post);
    MatrixView s1 = mv(sum1, n_rows, input.n_post);

    int target_nrn;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (target_nrn = 0; target_nrn < input.n_post; target_nrn++) {
        for (int i = 0; i <= input.n_spikes; i++) {
            double t_input = (i == input.n_spikes) ? INFINITY : input.times[i];
            double s0v = *mv_at(s0, i, target_nrn);
            double s1v = *mv_at(s1, i, target_nrn);

            /* Analytical tmax from log(s1v/s0v) */
            if (s0v > 0 && s1v > 0) {
                double tmax = tmax_pref * (tmax_sum + log(s1v / s0v));

                if (!isinf(tmax) && (i == 0 || input.times[i-1] <= tmax) &&
                    (i == input.n_spikes || tmax <= t_input)) {
                    double vmax = k_pref * (s1v * exp(-tmax/input.tau_mem) -
                                            s0v * exp(-tmax/input.tau_syn));
                    if (vmax > result.values[target_nrn]) {
                        result.values[target_nrn] = vmax;
                        result.times[target_nrn]  = tmax;
                    }
                }
            }

            /* Also check voltage at input spike time */
            if (t_input < INFINITY && (s0v != 0 || s1v != 0)) {
                double v_at_input = k_pref * (s1v * exp(-t_input/input.tau_mem) -
                                              s0v * exp(-t_input/input.tau_syn));
                if (v_at_input > result.values[target_nrn]) {
                    result.values[target_nrn] = v_at_input;
                    result.times[target_nrn]  = t_input;
                }
            }
        }
    }

    eventprop_free_sums_output(&sums);
    return result;
}

BackwardOutput eventprop_backward_maxima(const EventPropInput input,
                                         const double* maxima_times,
                                         const double* maxima_errors) {

    BackwardOutput result;
    result.error_no = EVENTPROP_OK;   /* default: success */

    /* Validate shared input parameters */
    EventPropError err = eventprop_validate_input(input);
    if (err != EVENTPROP_OK) {
        result.grad         = NULL;
        result.spike_errors = NULL;
        result.error_no     = err;
        return result;
    }

    /* Validate maxima pointers */
    if (!maxima_times || !maxima_errors) {
        result.grad         = NULL;
        result.spike_errors = NULL;
        result.error_no     = EVENTPROP_NULL_POINTER;
        return result;
    }

    /* Check for integer overflow in allocation size */
    if (input.n_pre > 0 && input.n_post > INT_MAX / input.n_pre) {
        result.grad         = NULL;
        result.spike_errors = NULL;
        result.error_no     = EVENTPROP_ALLOCATION_FAILED;
        return result;
    }

    /* Allocate output */
    result.grad         = calloc(input.n_pre * input.n_post, sizeof(double));
    result.spike_errors = calloc(input.n_spikes, sizeof(double));

    if (!result.grad || !result.spike_errors) {
        free(result.grad);
        free(result.spike_errors);
        result.grad         = NULL;
        result.spike_errors = NULL;
        result.error_no     = EVENTPROP_ALLOCATION_FAILED;
        return result;
    }

    if (input.n_spikes == 0 || input.n_post == 0) {
        return result;
    }

    double largest_time = input.times[input.n_spikes - 1];
    double k_bwd_prefactor = input.tau_mem / (input.tau_mem - input.tau_syn);

    /* Precompute inverse time constants */
    double tau_mem_inv = 1.0 / input.tau_mem;
    double tau_syn_inv = 1.0 / input.tau_syn;

    /* Sweep backward through input spikes */
    for (int spike_idx = input.n_spikes - 1; spike_idx >= 0; spike_idx--) {
        double t_bwd = largest_time - input.times[spike_idx];

        for (int max_idx = 0; max_idx < input.n_post; max_idx++) {
            if (isnan(maxima_times[max_idx]))
                continue;

            double t_vmax_bwd = largest_time - maxima_times[max_idx];
            if (t_vmax_bwd > t_bwd)
                continue;

            double dt_mem = -(t_bwd - t_vmax_bwd) * tau_mem_inv;
            double dt_syn = -(t_bwd - t_vmax_bwd) * tau_syn_inv;

            double lambda_v = -exp(dt_mem) * maxima_errors[max_idx] * tau_mem_inv;
            double k_bwd_val = k_bwd_prefactor * (exp(dt_mem) - exp(dt_syn));
            double lambda_i = -k_bwd_val * maxima_errors[max_idx] * tau_mem_inv;

            int source_nrn = input.sources[spike_idx];
            result.grad[source_nrn * input.n_post + max_idx] -= input.tau_syn * lambda_i;

            /* Spike error weighted by connection strength (matching Eigen/pybind11) */
            result.spike_errors[spike_idx] += input.w[source_nrn * input.n_post + max_idx] * (lambda_v - lambda_i);
        }
    }

    return result;
}

BackwardOutput eventprop_backward_spikes(const EventPropInput input, const SpikeOutput spike){

    BackwardOutput result;
    result.error_no = EVENTPROP_OK;   /* default: success */

    /* Validate shared input parameters */
    EventPropError err = eventprop_validate_input(input);
    if (err != EVENTPROP_OK) {
        result.grad         = NULL;
        result.spike_errors = NULL;
        result.error_no     = err;
        return result;
    }

    /* Validate output spike count */
    if (spike.n_spikes < 0) {
        result.grad         = NULL;
        result.spike_errors = NULL;
        result.error_no     = EVENTPROP_NEGATIVE_SPIKES;
        return result;
    }

    /* Check for integer overflow in allocation sizes */
    if (input.n_pre > 0 && input.n_post > INT_MAX / input.n_pre) {
        result.grad         = NULL;
        result.spike_errors = NULL;
        result.error_no     = EVENTPROP_ALLOCATION_FAILED;
        return result;
    }

    /* Validate output spike source bounds: post-sources index post-neurons */
    for (int i = 0; i < spike.n_spikes; ++i) {
        if (spike.sources[i] < 0 || spike.sources[i] >= input.n_post) {
            result.grad         = NULL;
            result.spike_errors = NULL;
            result.error_no     = EVENTPROP_INVALID_SOURCE;
            return result;
        }
    }

    /* Validate currents: must not equal v_th (division by zero in inv_ic) */
    for (int i = 0; i < spike.n_spikes; ++i) {
        if (spike.currents[i] == input.v_th) {
            result.grad         = NULL;
            result.spike_errors = NULL;
            result.error_no     = EVENTPROP_INVALID_CURRENT;
            return result;
        }
    }

    /* Allocate output */
    result.grad         = calloc(input.n_pre * input.n_post, sizeof(double));
    result.spike_errors = calloc(input.n_spikes, sizeof(double));

    if (!result.grad || !result.spike_errors) {
        free(result.grad);
        free(result.spike_errors);
        result.grad         = NULL;
        result.spike_errors = NULL;
        result.error_no     = EVENTPROP_ALLOCATION_FAILED;
        return result;
    }

    // No output spikes to process
    if(spike.n_spikes == 0) {
        return result;
    }

    int total = input.n_spikes + spike.n_spikes;

    /* Check for integer overflow in merged array size */
    if (input.n_spikes > INT_MAX - spike.n_spikes) {
        free(result.grad);
        free(result.spike_errors);
        result.grad         = NULL;
        result.spike_errors = NULL;
        result.error_no     = EVENTPROP_ALLOCATION_FAILED;
        return result;
    }

    MergedSpike* merged = malloc(total * sizeof(MergedSpike));
    if (!merged) {
        free(result.grad);
        free(result.spike_errors);
        result.grad         = NULL;
        result.spike_errors = NULL;
        result.error_no     = EVENTPROP_ALLOCATION_FAILED;
        return result;
    }

    for (int i = 0; i < input.n_spikes; i++) {
        merged[i].time=input.times[i];
        merged[i].idx=i;
        merged[i].is_input=1;
    }

    for (int i = 0; i < spike.n_spikes; i++) {
        merged[input.n_spikes+i].time = spike.times[i];
        merged[input.n_spikes+i].idx = i;
        merged[input.n_spikes+i].is_input=0;
    }

      /* Sort descending by time: O(n log n) via qsort */
    qsort(merged, total, sizeof(MergedSpike), cmp_merged_desc);

    double* all_errors = calloc(total, sizeof(double));
    for (int i = 0; i < spike.n_spikes; i++) {
        all_errors[input.n_spikes + i] = spike.errors[i];
    }

    /* ── Allocate intermediate buffers (with NULL checks) ─────────────── */

    double* lambda_v = calloc(input.n_post, sizeof(double));
    if (!lambda_v) goto bwd_alloc_fail;
    double* lambda_i = malloc(input.n_post * sizeof(double));
    if (!lambda_i) { free(lambda_v); goto bwd_alloc_fail; }

    /* Per-neuron jump storage (matching Eigen's vector<vector<pair<double,double>>> structure) */
    double** jump_time = calloc(input.n_post, sizeof(double*));
    if (!jump_time) { free(lambda_v); free(lambda_i); goto bwd_alloc_fail; }
    double** jump_mag  = calloc(input.n_post, sizeof(double*));
    if (!jump_mag)  { free(lambda_v); free(lambda_i); free(jump_time); goto bwd_alloc_fail; }
    int*   jump_count  = calloc(input.n_post, sizeof(int));
    if (!jump_count) { free(lambda_v); free(lambda_i); free(jump_time); free(jump_mag); goto bwd_alloc_fail; }
    int*   jump_cap    = calloc(input.n_post, sizeof(int));
    if (!jump_cap)     { free(lambda_v); free(lambda_i); free(jump_time); free(jump_mag); free(jump_count); goto bwd_alloc_fail; }

    for (int i = 0; i < input.n_post; i++) {
        jump_cap[i] = 8;  /* initial capacity */
        jump_time[i] = malloc(jump_cap[i] * sizeof(double));
        if (!jump_time[i]) goto bwd_alloc_fail_partial;
        jump_mag[i]  = malloc(jump_cap[i] * sizeof(double));
        if (!jump_mag[i]) { free(jump_time[i]); goto bwd_alloc_fail_partial; }
    }

    MutableMatrixView grad = mv_mut(result.grad, input.n_pre, input.n_post);
    MatrixView w           = mv(input.w, input.n_pre, input.n_post);

    double largest_time = merged[0].time, previous_t = largest_time;
    double k_bwd = input.tau_mem / (input.tau_mem - input.tau_syn);

    for (int s = 0; s < total; s++) {
        int spike_idx = merged[s].idx, is_input = merged[s].is_input;
        double spike_time = merged[s].time;
        int spike_source = is_input ? input.sources[spike_idx] : spike.sources[spike_idx];

        double t_bwd = largest_time - spike_time;
        double dt = previous_t - t_bwd;
        scale_inplace_simd(lambda_v, exp(dt / input.tau_mem), input.n_post);

        if (is_input) {
            /* Compute lambda_i for ALL post-synaptic neurons (matching Eigen structure) */
            memset(lambda_i, 0, input.n_post * sizeof(double));
            for (int nrn = 0; nrn < input.n_post; nrn++) {
                for (int j = 0; j < jump_count[nrn]; j++) {
                    double dt_jump = t_bwd - jump_time[nrn][j];
                    lambda_i[nrn] += jump_mag[nrn][j] * k_bwd * (exp(-dt_jump / input.tau_mem) - exp(-dt_jump / input.tau_syn));
                }
            }

            vadd_scaled_simd(mv_row_mut(grad, spike_source), lambda_i, -input.tau_syn, input.n_post);

            /* Compute dot product: w[row] · (lambda_v - lambda_i)
               Use stack allocation for small n_post, heap otherwise. */
#define BACKWARD_DIFF_STACK_SIZE 512
            double stack_diff[BACKWARD_DIFF_STACK_SIZE];
            int use_stack = (input.n_post <= BACKWARD_DIFF_STACK_SIZE);
            double* diff = use_stack ? stack_diff : malloc(input.n_post * sizeof(double));

            vsub_simd(diff, lambda_v, lambda_i, input.n_post);
            double outbound = dot_product_simd(mv_row(w, spike_source), diff, input.n_post);
            all_errors[spike_idx] += outbound;

            if (!use_stack) free(diff);

            /* Write spike error to alloc buffer */
            result.spike_errors[spike_idx] = all_errors[spike_idx];
        } else {
            double current = spike.currents[spike_idx];
            double inv_ic = 1.0 / (current - input.v_th);
            int nrn = spike_source;  /* post-spike: spike_source is the post-synaptic neuron */
            double post_err = all_errors[input.n_spikes + spike_idx];

            /* Record jump for this neuron (with capacity management) */
            if (jump_count[nrn] >= jump_cap[nrn]) {
                jump_cap[nrn] *= 2;
                double* new_time = realloc(jump_time[nrn], jump_cap[nrn] * sizeof(double));
                if (!new_time) goto bwd_alloc_fail_partial;
                jump_time[nrn] = new_time;
                double* new_mag = realloc(jump_mag[nrn],  jump_cap[nrn] * sizeof(double));
                if (!new_mag) { free(new_time); goto bwd_alloc_fail_partial; }
                jump_mag[nrn] = new_mag;
            }
            jump_time[nrn][jump_count[nrn]] = t_bwd;
            jump_mag[nrn][jump_count[nrn]]  = inv_ic * (input.v_th * lambda_v[nrn] + post_err);
            jump_count[nrn]++;

            lambda_v[nrn] = (current / (current - input.v_th)) * lambda_v[nrn] + inv_ic * post_err;
        }
        previous_t = t_bwd;
    }

    for (int i = 0; i < input.n_post; i++) {
        free(jump_time[i]); free(jump_mag[i]);
    }
    free(merged); free(all_errors); free(lambda_v); free(lambda_i);
    free(jump_time); free(jump_mag); free(jump_count); free(jump_cap);

    return result;

/* ── Allocation failure cleanup paths ─────────────────────────────── */
bwd_alloc_fail_partial:
    for (int j = 0; j < input.n_post; j++) {
        if (jump_time) free(jump_time[j]);
        if (jump_mag)  free(jump_mag[j]);
    }
bwd_alloc_fail:
    free(lambda_v);   free(lambda_i);
    free(jump_time);  free(jump_mag);
    free(jump_count); free(jump_cap);
    free(all_errors); free(merged);
    free(result.grad);     result.grad = NULL;
    free(result.spike_errors); result.spike_errors = NULL;
    result.error_no = EVENTPROP_ALLOCATION_FAILED;
    return result;
}
