/**
 * @file eventprop.c
 * @brief EventProp: Event-based backpropagation for SNN training.
 *
 * Consolidated implementation with automatic memory allocation (alloc versions).
 * Includes legacy diagnostic functions for compatibility.
 */

#include "eventprop.h"
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "matrixview.inc"  /* inline MatrixView helpers */

#ifdef _OPENMP
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"
#endif
#include <omp.h>
#ifdef __clang__
#pragma clang diagnostic pop
#endif
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
#define BACKWARD_DIFF_STACK_SIZE 512

#define TOMS748_MAX_ITER 1024
#define TOMS748_TOL 1e-14
#define TOMS748_NEAR_ZERO 1e-300   /* near-zero guard to avoid div-by-zero in curvature checks */

/* Voltage threshold crossing tolerance — has v genuinely exceeded v_th? */
#define SPIKE_THRESHOLD_EPS 1e-14

/* Evaluate v(t) - v_th for root-finding */
#define VOLTAGE(c, t) \
    (compute_v_factored(t, (c)->i, (c)->spike_count, (c)->post_spike_times, (c)->neuron_offset, \
                 (c)->sum0, (c)->sum1, (c)->target_nrn, (c)->n_post, \
                 (c)->k_prefactor, (c)->tau_mem_inv, (c)->tau_syn_inv, (c)->v_th, \
                 (c)->post_spike_exp_st) - (c)->v_th)

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

/** Context for eventprop_backward_spikes — groups all heap allocations. */
typedef struct BackwardSpikesCtx {
    MergedSpike* merged;
    double* all_errors;
    double* lambda_v;
    double* lambda_i;
    double** jump_time;   // array of pointers
    double** jump_mag;
    int* jump_count;
    int* jump_cap;
    int n_post_alloc;     // how many per-neuron arrays were actually allocated
} BackwardSpikesCtx;

// ── Internal context for compute_spikes memory management ─────────────────────
typedef struct ComputeSpikesCtx {
    int n_post;  // needed for per-neuron clean-up loop
    SumsOutput sums;              // eventprop_free_sums_output
    double** post_times_arr;      // array + each [n]
    double** post_currents_arr;   // array + each [n]
    int* spike_caps;
    int* spike_counts;
    SpikeEntry* entries;          // sorted collection buffer
    /* Output fields (set by collect_spikes_result) */
    double* result_times;
    int*    result_sources;
    double* result_currents;
    double* result_errors;
    int     result_n_spikes;
} ComputeSpikesCtx;

/** Per-neuron mutable state during spike detection. */
typedef struct NeuronState {
    double* p_times;
    double* p_currents;
    int cap;
    int count;
} NeuronState;

// Inline functions replacing GCC statement expressions ({ ... }) for MSVC compatibility.
// compute_v_inline — direct evaluation, no precomputed exponential.
static inline double compute_v_inline(const double t, const int i, const int spike_count,
    const double* post_spike_times, const int neuron_offset,
    const double* sum0, const double* sum1, const int target_nrn,
    const EventPropInput input, const double k_prefactor)
{
    double mem = k_prefactor * (sum1[i * input.n_post + target_nrn] * exp(-t / input.tau_mem)
                    - sum0[i * input.n_post + target_nrn] * exp(-t / input.tau_syn));
    for (int s = 0; s < spike_count; s++) {
        const double st = post_spike_times[neuron_offset + s];
        if (st <= t) {  /* Include all spikes where st <= t, don't break early */
            mem -= input.v_th * exp(-(t - st) / input.tau_mem);
        }
    }
    return mem;
}

// compute_v_factored — uses precomputed exp(st*tau-1) to share one exp() call.
static inline double compute_v_factored(const double t, const int i, const int spike_count,
    const double* post_spike_times, const int neuron_offset,
    const double* sum0, const double* sum1, const int target_nrn,const  int n_post,
    const double k_prefactor, const double tau_mem_inv, const double tau_syn_inv,
    const double v_th, const double* post_spike_exp_st)
{
    const double e_t = exp(-(t) * tau_mem_inv);
    double mem = k_prefactor * (sum1[i * n_post + target_nrn] * e_t
                    - sum0[i * n_post + target_nrn] * exp(-(t) * tau_syn_inv));
    if (post_spike_exp_st) {
        for (int s = 0; s < spike_count; s++) {
            const double st = post_spike_times[neuron_offset + s];
            if (st <= t) {
                mem -= v_th * e_t * post_spike_exp_st[s];
            }
        }
    } else {
        for (int s = 0; s < spike_count; s++) {
            const double st = post_spike_times[neuron_offset + s];
            if (st <= t) {
                mem -= v_th * exp(-(t - st) * tau_mem_inv);
            }
        }
    }
    return mem;
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
                               const double scalar, const int n)
{
    if (!simd_available() || n < 2) {
        for (int j = 0; j < n; j++)
            dest[j] = src[j] + scalar * weights[j];
        return;
    }

    const int vec = simd_vec_width();
    const int vec_count = n / vec;

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
        const float64x2_t sc = vdupq_n_f64(scalar);
        for (int i = 0; i < vec_count; i++) {
            const float64x2_t s = vld1q_f64(src + i * vec);
            const float64x2_t w = vld1q_f64(weights + i * vec);
            vst1q_f64(dest + i * vec, vaddq_f64(s, vmulq_f64(w, sc)));
        }
    }
#endif

    /* Remainder */
    for (int j = vec_count * vec; j < n; j++)
        dest[j] = src[j] + scalar * weights[j];
}

/* Scale in-place: span[j] *= scalar */
static void scale_inplace_simd(double* span, const double scalar, const int n)
{
    if (!simd_available() || n < 2) {
        for (int j = 0; j < n; j++)
            span[j] *= scalar;
        return;
    }

    const int vec = simd_vec_width();
    const int vec_count = n / vec;

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
        const float64x2_t sc = vdupq_n_f64(scalar);
        for (int i = 0; i < vec_count; i++) {
            const float64x2_t s = vld1q_f64(span + i * vec);
            vst1q_f64(span + i * vec, vmulq_f64(s, sc));
        }
    }
#endif

    for (int j = vec_count * vec; j < n; j++)
        span[j] *= scalar;
}

/* Vector-scalar multiply-accumulate: dest[j] += src[j] * scalar */
static void vadd_scaled_simd(double* dest, const double* src, const double scalar, const int n)
{
    if (!simd_available() || n < 2) {
        for (int j = 0; j < n; j++)
            dest[j] += src[j] * scalar;
        return;
    }

    const int vec = simd_vec_width();
    const int vec_count = n / vec;

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
        const float64x2_t sc = vdupq_n_f64(scalar);
        for (int i = 0; i < vec_count; i++) {
            const float64x2_t d = vld1q_f64(dest + i * vec);
            const float64x2_t s = vld1q_f64(src + i * vec);
            vst1q_f64(dest + i * vec, vaddq_f64(d, vmulq_f64(s, sc)));
        }
    }
#endif

    for (int j = vec_count * vec; j < n; j++)
        dest[j] += src[j] * scalar;
}

/* Dot product: sum(a[j] * b[j]) for j = 0...n-1 */
static double dot_product_simd(const double* a, const double* b, const int n)
{
    if (!simd_available() || n < 2) {
        double sum = 0.0;
        for (int j = 0; j < n; j++)
            sum += a[j] * b[j];
        return sum;
    }

    const int vec = simd_vec_width();
    const int vec_count = n / vec;

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
            const float64x2_t va = vld1q_f64(a + i * vec);
            const float64x2_t vb = vld1q_f64(b + i * vec);
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

    const int vec = simd_vec_width();
    const int vec_count = n / vec;

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
            const float64x2_t va = vld1q_f64(a + i * vec);
            const float64x2_t vb = vld1q_f64(b + i * vec);
            vst1q_f64(dest + i * vec, vsubq_f64(va, vb));
        }
    }
#endif

    for (int j = vec_count * vec; j < n; j++)
        dest[j] = a[j] - b[j];
}

static int cmp_merged_desc(const void* a, const void* b) {
    const double ta = ((const MergedSpike*)a)->time;
    const double tb = ((const MergedSpike*)b)->time;
    return (ta < tb) - (ta > tb);
}

static double compute_tmax(const int i, const int n_spikes, const double* times,
                           const int spike_count, const double* post_spike_times, const int neuron_offset,
                           const double* sum0, const double* sum1,
                           const int target_nrn, const int n_post,
                           const double tmax_prefactor, const double tmax_summand, const double tau_mem, const double v_th)
{
    if (i == 0) return 0.0;

    const double t_input = (i == n_spikes) ? INFINITY : times[i];
    double t_before = (i > 0) ? times[i - 1] : 0.0;

    for (int s = 0; s < spike_count; s++) {
        const double st = post_spike_times[neuron_offset + s];
        if (st > t_input) break;
        if (st > t_before) t_before = st;
    }

    const double sum0_elem = sum0[i * n_post + target_nrn];
    const double sum1_elem = sum1[i * n_post + target_nrn];

    if (sum0_elem <= 0 || sum1_elem <= 0) return INFINITY;

    double sum1_adj = sum1_elem;
    for (int s = 0; s < spike_count; s++) {
        const double st = post_spike_times[neuron_offset + s];
        if (st > t_input) break;
        sum1_adj += -v_th * exp(st / tau_mem);
    }

    const double tmax = tmax_prefactor * (tmax_summand + log(sum1_adj / sum0_elem));

    if (t_before <= tmax && (i == n_spikes || tmax <= t_input))
        return tmax;
    return INFINITY;
}

static double compute_i(const double t, const int n_spikes, const double* times,
                        const double* sum0, const int target_nrn, const int n_post, const double tau_syn)
{
    int idx = 0;
    for (; idx < n_spikes; idx++) {
        if (times[idx] > t) break;
    }
    return sum0[idx * n_post + target_nrn] * exp(-t / tau_syn);
}

static int cmp_spike_asc(const void* a, const void* b) {
    const double ta = ((const SpikeEntry*)a)->time;
    const double tb = ((const SpikeEntry*)b)->time;
    return (ta > tb) - (ta < tb);
}

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
        const double m = 0.5 * (b - a);
        if (fabs(m) <= TOMS748_TOL || fb == 0.0)
            return b;

        double s = 0.0;
        int useBisection = 1;

        if (fabs(fc) > fabs(fb) && c != a && c != b) {
            const double qba = (fb - fa) / (b - a);
            const double qcb = (fc - fb) / (c - b);
            const double qcbba = (qcb - qba) / (c - a);

            if (fabs(qcbba) > TOMS748_NEAR_ZERO) {
                const double denom = qba + qcb;
                if (fabs(denom) > TOMS748_NEAR_ZERO) {
                    s = b - fb / (qba + qcb * (b - a) * qcbba / denom);
                    const int inBracket = (s > a && s < b);
                    const double mVal = 0.5 * (b - a);
                    const int smallStep = fabs(s - b) < mVal;
                    const int prevWasBisect = fabs(d - mVal) < TOMS748_TOL * 100;
                    if (inBracket && smallStep && !prevWasBisect)
                        useBisection = 0;
                }
            }
        }

        if (useBisection) s = a + m;

        const double f_oldB = fb;   // save before updating - avoids redundant fn call below
        const double prevB = b; b = s; fb = VOLTAGE(ctx, b);
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

/** Grow both times and currents buffers if needed.
 *  Returns: EVENTPROP_OK on success, EVENTPROP_ALLOC_NEURON_BUF on failure. */
static inline EventPropError grow_spike_buffers(double** p_times,
                                                double** p_currents,
                                                int* cap,
                                                int count)
{
    if (count < (*cap)) return EVENTPROP_OK;

    (*cap) *= 3;
    const size_t new_size = (size_t)(*cap) * sizeof(double);

    /* Sequential realloc: undo times if currents fails */
    double* new_times = realloc(*p_times, new_size);
    if (!new_times) return EVENTPROP_ALLOC_NEURON_BUF;

    double* new_currents = realloc(*p_currents, new_size);
    if (!new_currents) {
        free(new_times);  /* rollback */
        return EVENTPROP_ALLOC_NEURON_BUF;
    }

    *p_times = new_times;
    *p_currents = new_currents;
    return EVENTPROP_OK;
}

/** Try to find a spike root in [t_lower, t_upper] and record it.
 *
 *  Returns:
 *    0  — spike found and recorded
 *   -1  — allocation error (caller should abort)
 *    1  — no valid spike in bracket
 */
static inline int try_find_root_spike(const RootContext* rctx_base,
                                      const EventPropInput input,
                                      double t_lower,
                                      double t_upper,
                                      NeuronState* state)
{
    /* Precompute exp(st * tau_mem_inv) for factorization */
    double* post_exp_st = NULL;
    if (state->count > 0) {
        post_exp_st = malloc((size_t)state->count * sizeof(double));
        if (!post_exp_st) return -1;

        const double tau_mem_inv = 1.0 / input.tau_mem;
        for (int s = 0; s < state->count; s++) {
            post_exp_st[s] = exp(state->p_times[s] * tau_mem_inv);
        }
    }

    RootContext rctx = *rctx_base;
    /* Caller sets rctx_base.i to the current interval index */
    rctx.spike_count = state->count;
    rctx.post_spike_exp_st = post_exp_st;

    const double t_post = toms748_solve(&rctx, t_lower, t_upper);
    free(post_exp_st);

    /* Validate root: must be strictly inside bracket */
    if (!(t_post > t_lower && t_post < t_upper)) return 1;

    /* Grow buffers if needed */
    if (grow_spike_buffers(&state->p_times, &state->p_currents,
                           &state->cap, state->count) != EVENTPROP_OK)
        return -1;

    /* Record spike */
    state->p_times[state->count] = t_post;
    state->p_currents[state->count] = compute_i(t_post,
                                                 input.n_spikes,
                                                 input.times,
                                                 rctx_base->sum0,
                                                 rctx_base->target_nrn,
                                                 rctx_base->n_post,
                                                 input.tau_syn);
    state->count++;
    return 0;
}

/* ── Collect per-neuron spikes into sorted SpikeOutput (shared by v1 and v2) ── */
static inline EventPropError collect_spikes_result(ComputeSpikesCtx* ctx,
                                                   int n_post)
{
    /* Count total spikes */
    int total = 0;
    for (int n = 0; n < n_post; n++)
        total += ctx->spike_counts[n];

    if (total == 0) return EVENTPROP_OK;  /* no spikes, nothing to do */

    /* Collect into sortable entries */
    SpikeEntry* entries = malloc((size_t)total * sizeof(SpikeEntry));
    if (!entries) return EVENTPROP_ALLOC_ENTRIES;

    int idx = 0;
    for (int n = 0; n < n_post; n++) {
        const int cnt = ctx->spike_counts[n];
        const double* pt = ctx->post_times_arr[n];
        const double* pc = ctx->post_currents_arr[n];
        for (int k = 0; k < cnt; k++) {
            entries[idx].time    = pt[k];
            entries[idx].source  = n;
            entries[idx].current = pc[k];
            idx++;
        }
    }

    qsort(entries, (size_t)total, sizeof(SpikeEntry), cmp_spike_asc);

    /* Allocate output arrays */
    double* times    = malloc((size_t)total * sizeof(double));
    if (!times) { free(entries); return EVENTPROP_ALLOC_RESULT_TIMES; }

    int* sources     = malloc((size_t)total * sizeof(int));
    if (!sources) { free(times); free(entries); return EVENTPROP_ALLOC_RESULT_SOURCES; }

    double* currents = malloc((size_t)total * sizeof(double));
    if (!currents) { free(sources); free(times); free(entries);
                     return EVENTPROP_ALLOC_RESULT_CURRENTS; }

    double* errors   = calloc((size_t)total, sizeof(double));
    if (!errors) { free(currents); free(sources); free(times); free(entries);
                   return EVENTPROP_ALLOC_RESULT_ERRORS; }

    /* Copy sorted data into output */
    for (int i = 0; i < total; i++) {
        times[i]    = entries[i].time;
        sources[i]  = entries[i].source;
        currents[i] = entries[i].current;
    }

    /* Store in ctx so caller can assign to result */
    ctx->result_times   = times;
    ctx->result_sources = sources;
    ctx->result_currents = currents;
    ctx->result_errors  = errors;
    ctx->result_n_spikes = total;

    free(entries);
    return EVENTPROP_OK;
}

/* ── Allocate per-neuron dynamic spike buffers (used by v2) ── */
static inline EventPropError allocate_neuron_buffers(ComputeSpikesCtx* ctx,
                                                     const EventPropInput input)
{
    const size_t sz = (size_t)input.n_post;

    ctx->post_times_arr = malloc(sz * sizeof(double*));
    if (!ctx->post_times_arr) return EVENTPROP_ALLOC_TIMES_ARR;

    ctx->post_currents_arr = malloc(sz * sizeof(double*));
    if (!ctx->post_currents_arr) return EVENTPROP_ALLOC_CURRENTS_ARR;

    ctx->spike_caps = calloc(sz, sizeof(int));
    if (!ctx->spike_caps) return EVENTPROP_ALLOC_SPIKE_CAPS;

    ctx->spike_counts = calloc(sz, sizeof(int));
    if (!ctx->spike_counts) return EVENTPROP_ALLOC_SPIKE_COUNTS;

    const size_t initial_cap = (size_t)(3 * input.n_spikes + 1);
    int failed = 0;
    for (int n = 0; n < input.n_post; n++) {
        ctx->post_times_arr[n]   = calloc(initial_cap, sizeof(double));
        if (!ctx->post_times_arr[n]) { failed = n + 1; break; }
        ctx->post_currents_arr[n] = calloc(initial_cap, sizeof(double));
        if (!ctx->post_currents_arr[n]) {
            free(ctx->post_times_arr[n]);
            failed = n + 1;
            break;
        }
        ctx->spike_caps[n] = (int)initial_cap;
    }

    if (failed > 0) {
        for (int n = 0; n < failed - 1; n++) {
            free(ctx->post_times_arr[n]);
            free(ctx->post_currents_arr[n]);
        }
        return EVENTPROP_ALLOC_NEURON_BUF;
    }

    return EVENTPROP_OK;
}

/** Generate candidate upper bounds for interval i.
 *  Returns the number of candidates (0, 1, or 2). */
static inline int generate_bracket_candidates(const RootContext* rctx,
                                              int count,
                                              const double* p_times,
                                              const EventPropInput input,
                                              double tmax_prefactor,
                                              double tmax_summand,
                                              double t_before,
                                              double candidates[2])
{
    int n_candidates = 0;

    if (rctx->i == input.n_spikes) {
        /* ── Last interval: one candidate — analytical tmax from adjusted sums ── */
        double s0 = rctx->sum0[input.n_spikes * input.n_post + rctx->target_nrn];
        double s1 = rctx->sum1[input.n_spikes * input.n_post + rctx->target_nrn];
        for (int s = 0; s < count; s++)
            s1 += -input.v_th * exp(p_times[s] / input.tau_mem);

        if (s0 > 0) {
            const double tmax = tmax_prefactor * (tmax_summand + log(s1 / s0));
            /* Validate bracket: tmax must be >= t_before (matches compute_tmax bounds check) */
            if (!isinf(tmax) && t_before <= tmax) {
                const double v = compute_v_inline(tmax, rctx->i, count, p_times, 0,
                                            rctx->sum0, rctx->sum1, rctx->target_nrn, input, rctx->k_prefactor);
                if (v > input.v_th + SPIKE_THRESHOLD_EPS)
                    candidates[n_candidates++] = tmax;
            }
        }
    } else {
        /* ── Regular interval: up to 2 candidates ── */
        const double t_input = input.times[rctx->i];

        /* Candidate 1: analytical tmax */
        const double tmax = compute_tmax(rctx->i, input.n_spikes, input.times, count, p_times, 0,
                                   rctx->sum0, rctx->sum1, rctx->target_nrn, rctx->n_post,
                                   tmax_prefactor, tmax_summand, input.tau_mem, input.v_th);
        if (!isinf(tmax)) {
            const double v = compute_v_inline(tmax, rctx->i, count, p_times, 0,
                                        rctx->sum0, rctx->sum1, rctx->target_nrn, input, rctx->k_prefactor);
            if (v > input.v_th + SPIKE_THRESHOLD_EPS)
                candidates[n_candidates++] = tmax;
        }

        /* Candidate 2: voltage at input spike time */
        const double v_at_input = compute_v_inline(t_input, rctx->i, count, p_times, 0,
                                             rctx->sum0, rctx->sum1, rctx->target_nrn, input, rctx->k_prefactor);
        if (v_at_input > input.v_th + SPIKE_THRESHOLD_EPS)
            candidates[n_candidates++] = t_input;
    }

    return n_candidates;
}


static void free_backward_spikes_ctx(const BackwardSpikesCtx* ctx) {
    if (!ctx) return;
    for (int i = 0; i < ctx->n_post_alloc; i++) {
        free(ctx->jump_time[i]);
        free(ctx->jump_mag[i]);
    }
    free(ctx->merged);
    free(ctx->all_errors);
    free(ctx->lambda_v);
    free(ctx->lambda_i);
    free(ctx->jump_time);
    free(ctx->jump_mag);
    free(ctx->jump_count);
    free(ctx->jump_cap);
}

static void free_compute_spikes_ctx(const ComputeSpikesCtx* ctx) {
    eventprop_free_sums_output(ctx->sums);
    if (ctx->post_times_arr) {
        for (int n = 0; n < ctx->n_post; n++) {
            free(ctx->post_times_arr[n]);
            free(ctx->post_currents_arr[n]);
        }
        free(ctx->post_times_arr);
        free(ctx->post_currents_arr);
    }
    free(ctx->spike_caps);
    free(ctx->spike_counts);
    free(ctx->entries);
}

void eventprop_free_spike_output(const SpikeOutput output)
{
    free(output.times);
    free(output.sources);
    free(output.currents);
    free(output.errors);
}

void eventprop_free_sums_output(const SumsOutput output)
{
    free(output.sum0);
    free(output.sum1);
}

void eventprop_free_maxima_output(const MaximaOutput output)
{
    free(output.times);
    free(output.values);
    free(output.errors);
}

void eventprop_free_backward_output(const BackwardOutput output){
    free(output.grad);
    free(output.spike_errors);
}

EventPropError eventprop_validate_input(const EventPropInput input)
{
    /* ── Dimension checks ─────────────────────────────────────── */
    if (input.n_pre <= 0 || input.n_post <= 0)
        return EVENTPROP_INVALID_DIMENSION;
    if (input.n_pre > EVENTPROP_MAX_DIM || input.n_post > EVENTPROP_MAX_DIM)
        return EVENTPROP_DIMENSION_TOO_LARGE;
    /* n_pre * n_post can overflow int even when both pass MAX_DIM */
    if ((size_t)input.n_pre * (size_t)input.n_post > (size_t)INT_MAX)
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
        const double min_tau = fmin(input.tau_mem, input.tau_syn);
        const double rel_diff = fabs(input.tau_mem - input.tau_syn) / min_tau;
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

        const double t = input.times[i];
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
        const double min_tau = fmin(input.tau_mem, input.tau_syn);
        if (max_time > 0.0 && max_time / min_tau >= EVENTPROP_EXP_OVERFLOW_THRESHOLD)
            return EVENTPROP_TIMES_TAU_MISMATCH;
    }

    return EVENTPROP_OK;
}

SpikeOutput eventprop_compute_spikes(const EventPropInput input)
{
    /* Context tracks all heap allocations for unified clean-up */
    ComputeSpikesCtx ctx = { .n_post = input.n_post };
    SpikeOutput result = (SpikeOutput){ .error_no = eventprop_validate_input(input) };

    if (result.error_no != EVENTPROP_OK)
        goto cleanup_and_return;

    const double k_prefactor = input.tau_syn / (input.tau_mem - input.tau_syn);
    const double tmax_prefactor = 1.0 / (1.0 / input.tau_mem - 1.0 / input.tau_syn);
    const double tmax_summand = log(input.tau_syn / input.tau_mem);
    const double tau_mem_inv = 1.0 / input.tau_mem;
    const double tau_syn_inv = 1.0 / input.tau_syn;

    /* Compute sums */
    ctx.sums = eventprop_compute_sums(input);
    if (ctx.sums.error_no != EVENTPROP_OK) {
        result.error_no = ctx.sums.error_no;
        goto cleanup_and_return;
    }

    const double* sum0 = ctx.sums.sum0;
    const double* sum1 = ctx.sums.sum1;

    /* Allocate per-neuron dynamic buffers */
    result.error_no = allocate_neuron_buffers(&ctx, input);
    if (result.error_no != EVENTPROP_OK)
        goto cleanup_and_return;

    /* Shared error flag for OpenMP region */
#ifdef _OPENMP
    volatile
#endif
    int fwd_alloc_error = 0;

    int target_nrn;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (target_nrn = 0; target_nrn < input.n_post; target_nrn++) {

        double* p_times    = ctx.post_times_arr[target_nrn];
        double* p_currents = ctx.post_currents_arr[target_nrn];
        int cap = ctx.spike_caps[target_nrn];
        int count = 0;
        int finished = 0, processed = 0;

        /* Root context — constant fields set once per neuron */
        RootContext rctx_base = (RootContext){
            .sum0 = sum0, .sum1 = sum1,
            .post_spike_times = p_times, .neuron_offset = 0,
            .target_nrn = target_nrn, .n_post = input.n_post,
            .k_prefactor = k_prefactor, .tau_mem_inv = tau_mem_inv,
            .tau_syn_inv = tau_syn_inv, .v_th = input.v_th,
        };

        while (!finished) {
            int found = 0;
            for (int i = processed; i <= input.n_spikes; i++) {

                /* ── Compute t_lower (bracket start) ── */
                const double t_input = (i == input.n_spikes) ? INFINITY : input.times[i];

                double t_before;
                if (i == input.n_spikes) {
                    /* Last interval: lower bound is last post-spike or 0 */
                    t_before = count > 0 ? p_times[count - 1] : 0.0;
                } else {
                    /* Regular interval: latest of (prev input time, post spikes before current input) */
                    t_before = (i > 0) ? input.times[i - 1] : 0.0;
                    for (int s = 0; s < count; s++) {
                        double st = p_times[s];
                        if (st > t_input) break;
                        if (st > t_before) t_before = st;
                    }
                }

                /* ── Generate candidate upper bounds ── */
                rctx_base.i = i;
                double candidates[2];
                int n_candidates = generate_bracket_candidates(
                    &rctx_base, count, p_times,
                    input, tmax_prefactor, tmax_summand, t_before, candidates);

                /* ── Try each candidate bracket with shared solver ── */
                for (int c = 0; c < n_candidates; c++) {
                    NeuronState state = { .p_times = p_times,
                                          .p_currents = p_currents,
                                          .cap = cap, .count = count };

                    const int rc = try_find_root_spike(&rctx_base, input, t_before,
                                                 candidates[c], &state);

                    if (rc == -1) {
                        fwd_alloc_error = 1;
                        finished = 1;
                        break;
                    }
                    if (rc == 0) {
                        /* Spike found — update local state from NeuronState */
                        p_times    = state.p_times;
                        p_currents = state.p_currents;
                        cap        = state.cap;
                        count      = state.count;
                        found = 1;
                        break;  /* short-circuit: don't try remaining candidates */
                    }
                }

                if (found) {
                    processed = i;
                    break;  /* re-scan this interval next while iteration */
                }
                if (i == input.n_spikes) {
                    finished = 1;  /* exhausted last interval with no spike */
                }
            }

            if (!found)
                processed = (processed + 1 < input.n_spikes) ? processed + 1 : input.n_spikes;
        }

        /* Write back updated pointers */
        ctx.post_times_arr[target_nrn]   = p_times;
        ctx.post_currents_arr[target_nrn] = p_currents;
        ctx.spike_caps[target_nrn]       = cap;
        ctx.spike_counts[target_nrn]     = count;
    }

    if (fwd_alloc_error) {
        result.n_spikes = 0;
        result.error_no = EVENTPROP_ALLOC_NEURON_BUF;
        goto cleanup_and_return;
    }

    /* Collect, sort, and copy output */
    const EventPropError err = collect_spikes_result(&ctx, input.n_post);
    if (err != EVENTPROP_OK) {
        result.error_no = err;
        goto cleanup_and_return;
    }

    result.n_spikes   = ctx.result_n_spikes;
    result.times      = ctx.result_times;
    result.sources    = ctx.result_sources;
    result.currents   = ctx.result_currents;
    result.errors     = ctx.result_errors;

cleanup_and_return:
    if (result.error_no != EVENTPROP_OK)
        eventprop_free_spike_output(result);
    free_compute_spikes_ctx(&ctx);
    return result;
}

SumsOutput eventprop_compute_sums(const EventPropInput input)
{
    SumsOutput result = (SumsOutput){ .error_no = eventprop_validate_input(input) };
    double* exp_mem = NULL, *exp_syn = NULL;

    result.n_rows = input.n_spikes + 1;
    result.n_post = input.n_post;

    if (result.error_no != EVENTPROP_OK){
        goto cleanup_and_return;
    }

    size_t count = (size_t)result.n_rows * (size_t)input.n_post;
    
    result.sum0 = calloc(count, sizeof(double));
    if(!result.sum0) {
        result.error_no = EVENTPROP_ALLOC_SUM0;
        goto cleanup_and_return;
    }

    result.sum1 = calloc(count, sizeof(double));
    if (!result.sum1) {
        result.error_no = EVENTPROP_ALLOC_SUM1;
        goto cleanup_and_return;
    }

    if (input.n_spikes == 0) {
        goto cleanup_and_return;
    }

    /* Pre-compute exponential lookup tables */
    exp_mem = calloc((size_t)input.n_spikes, sizeof(double));
    if (!exp_mem) {
        result.error_no = EVENTPROP_ALLOC_EXP_MEM;
        goto cleanup_and_return;
    }
    exp_syn = calloc((size_t)input.n_spikes, sizeof(double));
    if (!exp_syn) {
        result.error_no = EVENTPROP_ALLOC_EXP_SYN;
        goto cleanup_and_return;
    }

    for (int i = 0; i < input.n_spikes; i++) {
        exp_mem[i] = exp(input.times[i] / input.tau_mem);
        exp_syn[i] = exp(input.times[i] / input.tau_syn);
    }

    /* Accumulate sums using SIMD-accelerated FMA */
    MutableMatrixView s0 = mv_mut(result.sum0, result.n_rows, input.n_post);
    MutableMatrixView s1 = mv_mut(result.sum1, result.n_rows, input.n_post);
    MatrixView w         = mv(input.w, input.n_pre, input.n_post);

    for (int i = 0; i < input.n_spikes; i++) {
        int src = input.sources[i];
        fma_row_simd(mv_row_mut(s0, i + 1), mv_row_mut(s0, i), mv_row(w, src), exp_syn[i], input.n_post);
        fma_row_simd(mv_row_mut(s1, i + 1), mv_row_mut(s1, i), mv_row(w, src), exp_mem[i], input.n_post);
    }

cleanup_and_return:
    if (result.error_no != EVENTPROP_OK) {
        eventprop_free_sums_output(result);
    }
    free(exp_mem);
    free(exp_syn);
    return result;
}

MaximaOutput eventprop_compute_maxima(const EventPropInput input)
{
    MaximaOutput result = (MaximaOutput){ .error_no = eventprop_validate_input(input) } ;
    SumsOutput sums     = {0};

    if (result.error_no != EVENTPROP_OK) {
        goto cleanup_and_return;
    }

    size_t sz = (size_t)input.n_post;
    /* Allocate output - always exactly n_post elements */
    result.times   = calloc(sz, sizeof(double));
    if (!result.times) {
        result.error_no = EVENTPROP_ALLOC_MAXIMA_TIMES;
        goto cleanup_and_return;
    }
    result.values  = calloc(sz, sizeof(double));
    if (!result.values) {
        result.error_no = EVENTPROP_ALLOC_MAXIMA_VALUES;
        goto cleanup_and_return;
    }
    result.errors  = calloc(sz, sizeof(double));
    if (!result.errors) {
        result.error_no = EVENTPROP_ALLOC_MAXIMA_ERRORS;
        goto cleanup_and_return;
    }

    result.n_post = input.n_post;

    /* Initialize maxima to "not found" */
    for (int i = 0; i < input.n_post; i++) {
        result.times[i]   = NAN;
        result.values[i]  = 0.0;
    }

    /* Compute sums */
    sums = eventprop_compute_sums(input);
    if (sums.error_no != EVENTPROP_OK) {
        result.error_no = sums.error_no;   /* propagate from compute_sums */
        goto cleanup_and_return;
    }

    int n_rows = input.n_spikes + 1;
    double* sum0 = sums.sum0;
    double* sum1 = sums.sum1;

    /* Find maxima for each post-synaptic neuron */
    const double k_pref = input.tau_syn / (input.tau_mem - input.tau_syn);
    const double tmax_pref = 1.0 / (1.0/input.tau_mem - 1.0/input.tau_syn);
    const double tmax_sum = log(input.tau_syn / input.tau_mem);

    /* No spikes → nothing to compute, return NaN/0 (matches Eigen) */
    if (input.n_spikes == 0) {
        goto cleanup_and_return;
    }

    MatrixView s0 = mv(sum0, n_rows, input.n_post);
    MatrixView s1 = mv(sum1, n_rows, input.n_post);

    int target_nrn;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    // ReSharper disable once CppJoinDeclarationAndAssignment
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

cleanup_and_return:
    eventprop_free_sums_output(sums);
    if (result.error_no != EVENTPROP_OK) {
        eventprop_free_maxima_output(result);
    }
    return result;
}

BackwardOutput eventprop_backward_maxima(const EventPropInput input,
                                         const double* maxima_times,
                                         const double* maxima_errors) {

    BackwardOutput result = (BackwardOutput) { .error_no = eventprop_validate_input(input) };

    if (result.error_no != EVENTPROP_OK) {
        goto cleanup_and_return;
    }

    /* Validate maxima pointers */
    if (!maxima_times || !maxima_errors) {
        result.error_no     = EVENTPROP_NULL_POINTER;
        goto cleanup_and_return;
    }

    /* Allocate output */
    result.grad         = calloc((size_t)(input.n_pre * input.n_post), sizeof(double));
    if (!result.grad) {
        result.error_no     = EVENTPROP_ALLOC_BWD_GRAD;
        goto cleanup_and_return;
    }
    result.spike_errors = calloc((size_t)input.n_spikes, sizeof(double));
    if (!result.spike_errors) {
        result.error_no     = EVENTPROP_ALLOC_BWD_SPIKE_ERRORS;
        goto cleanup_and_return;
    }

    result.n_pre = input.n_pre;
    result.n_post = input.n_post;
    result.n_spikes = input.n_spikes;

    if (input.n_spikes == 0 || input.n_post == 0) {
        goto cleanup_and_return;
    }

    const double largest_time = input.times[input.n_spikes - 1];
    const double k_bwd_prefactor = input.tau_mem / (input.tau_mem - input.tau_syn);

    /* Precompute inverse time constants */
    const double tau_mem_inv = 1.0 / input.tau_mem;
    const double tau_syn_inv = 1.0 / input.tau_syn;

    /* Sweep backward through input spikes */
    for (int spike_idx = input.n_spikes - 1; spike_idx >= 0; spike_idx--) {
        const double t_bwd = largest_time - input.times[spike_idx];

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

cleanup_and_return:
    if (result.error_no != EVENTPROP_OK) {
        eventprop_free_backward_output(result);
    }
    return result;
}

BackwardOutput eventprop_backward_spikes(const EventPropInput input, const SpikeOutput spike){

    BackwardOutput result = (BackwardOutput){ 
        .error_no = eventprop_validate_input(input),
        .n_pre = input.n_pre, 
        .n_post = input.n_post,   
        .n_spikes = input.n_spikes
    };
    BackwardSpikesCtx ctx = {0};

    if (result.error_no != EVENTPROP_OK) {
        goto cleanup_and_return;
    }

    /* Validate output spike count */
    if (spike.n_spikes < 0) {
        result.error_no     = EVENTPROP_NEGATIVE_SPIKES;
        goto cleanup_and_return;
    }

    int total = input.n_spikes + spike.n_spikes;

    /* Check for integer overflow in merged array size */
    if (input.n_spikes > INT_MAX - spike.n_spikes) {
        result.error_no     = EVENTPROP_ALLOC_BWD_MERGED; /* overflow guard */
        goto cleanup_and_return;
    }

    /* Validate output spike source bounds: post-sources index post-neurons */
    for (int i = 0; i < spike.n_spikes; ++i) {
        if (spike.sources[i] < 0 || spike.sources[i] >= input.n_post) {
            result.error_no     = EVENTPROP_INVALID_SOURCE;
            goto cleanup_and_return;
        }
    }

    /* Validate currents: must not equal v_th (division by zero in inv_ic) */
    for (int i = 0; i < spike.n_spikes; ++i) {
        if (spike.currents[i] == input.v_th) {
            result.error_no     = EVENTPROP_INVALID_CURRENT;
            goto cleanup_and_return;
        }
    }

    /* Allocate output */
    result.grad         = calloc((size_t)(input.n_pre * input.n_post), sizeof(double));
    if (!result.grad) {
        result.error_no     = EVENTPROP_ALLOC_BWD_GRAD;
        goto cleanup_and_return;
    }
    
    result.spike_errors = calloc((size_t)input.n_spikes, sizeof(double));
    if (!result.spike_errors) {
        result.error_no     = EVENTPROP_ALLOC_BWD_SPIKE_ERRORS;
        goto cleanup_and_return;
    }

    /* No output spikes to process */
    if (spike.n_spikes == 0) {
        goto cleanup_and_return;
    }

    ctx.merged = malloc((size_t)total * sizeof(MergedSpike));
    if (!ctx.merged) {
        result.error_no     = EVENTPROP_ALLOC_BWD_MERGED;
        goto cleanup_and_return;
    }

    for (int i = 0; i < input.n_spikes; i++) {
        ctx.merged[i].time   = input.times[i];
        ctx.merged[i].idx    = i;
        ctx.merged[i].is_input = 1;
    }

    for (int i = 0; i < spike.n_spikes; i++) {
        ctx.merged[input.n_spikes+i].time     = spike.times[i];
        ctx.merged[input.n_spikes+i].idx      = i;
        ctx.merged[input.n_spikes+i].is_input = 0;
    }

    /* Sort descending by time: O(n log n) via qsort */
    qsort(ctx.merged, (size_t)total, sizeof(MergedSpike), cmp_merged_desc);

    ctx.all_errors = calloc((size_t)total, sizeof(double));
    if (!ctx.all_errors) {
        result.error_no     = EVENTPROP_ALLOC_BWD_ALL_ERRORS;
        goto cleanup_and_return;
    }

    for (int i = 0; i < spike.n_spikes; i++) {
        ctx.all_errors[input.n_spikes + i] = spike.errors[i];
    }

    /* ── Allocate intermediate buffers (sequential checks) ─────────────── */

    ctx.lambda_v = calloc((size_t)input.n_post, sizeof(double));
    if (!ctx.lambda_v) {
        result.error_no     = EVENTPROP_ALLOC_BWD_LAMBDA_V;
        goto cleanup_and_return;
    }
   
    ctx.lambda_i = malloc((size_t)input.n_post * sizeof(double));
    if (!ctx.lambda_i) {
        result.error_no     = EVENTPROP_ALLOC_BWD_LAMBDA_I;
        goto cleanup_and_return;
    }

    /* Per-neuron jump storage (matching Eigen's vector<vector<pair<double,double>>> structure) */
    ctx.jump_time  = calloc((size_t)input.n_post, sizeof(double*));
    if (!ctx.jump_time) {
        result.error_no     = EVENTPROP_ALLOC_BWD_JUMP_TIME;
        goto cleanup_and_return;
    }
    
    ctx.jump_mag   = calloc((size_t)input.n_post, sizeof(double*));
    if (!ctx.jump_mag) {
        result.error_no     = EVENTPROP_ALLOC_BWD_JUMP_MAG;
        goto cleanup_and_return;
    }
    
    ctx.jump_count = calloc((size_t)input.n_post, sizeof(int));
    if (!ctx.jump_count) {
        result.error_no     = EVENTPROP_ALLOC_BWD_JUMP_COUNT;
        goto cleanup_and_return;
    }
    
    ctx.jump_cap   = calloc((size_t)input.n_post, sizeof(int));
    if (!ctx.jump_cap) {
        result.error_no     = EVENTPROP_ALLOC_BWD_JUMP_CAP;
        goto cleanup_and_return;
    }

    /* Allocate per-neuron buffers — track how many succeeded for safe cleanup */
    ctx.n_post_alloc = 0;
    for (int i = 0; i < input.n_post; i++) {
        ctx.jump_cap[i] = 8;  /* initial capacity */
        ctx.jump_time[i] = malloc((size_t)ctx.jump_cap[i] * sizeof(double));
        if (!ctx.jump_time[i]) {
            result.error_no     = EVENTPROP_ALLOC_BWD_PER_NEURON;
            goto cleanup_and_return;
        }
        ctx.jump_mag[i]  = malloc((size_t)ctx.jump_cap[i] * sizeof(double));
        if (!ctx.jump_mag[i]) {
            free(ctx.jump_time[i]);
            result.error_no     = EVENTPROP_ALLOC_BWD_PER_NEURON;
            goto cleanup_and_return;
        }
        ctx.n_post_alloc++;
    }

    MutableMatrixView grad = mv_mut(result.grad, input.n_pre, input.n_post);
    MatrixView w           = mv(input.w, input.n_pre, input.n_post);

    const double largest_time = ctx.merged[0].time;
    double previous_t = largest_time;
    const double k_bwd = input.tau_mem / (input.tau_mem - input.tau_syn);

    for (int s = 0; s < total; s++) {
        int spike_idx = ctx.merged[s].idx, is_input = ctx.merged[s].is_input;
        double spike_time = ctx.merged[s].time;
        int spike_source = is_input ? input.sources[spike_idx] : spike.sources[spike_idx];

        double t_bwd = largest_time - spike_time;
        double dt = previous_t - t_bwd;
        scale_inplace_simd(ctx.lambda_v, exp(dt / input.tau_mem), input.n_post);

        if (is_input) {
            /* Compute lambda_i for ALL post-synaptic neurons (matching Eigen structure) */
            memset(ctx.lambda_i, 0, (size_t)input.n_post * sizeof(double));
            for (int nrn = 0; nrn < input.n_post; nrn++) {
                for (int j = 0; j < ctx.jump_count[nrn]; j++) {
                    double dt_jump = t_bwd - ctx.jump_time[nrn][j];
                    ctx.lambda_i[nrn] += ctx.jump_mag[nrn][j] * k_bwd *
                        (exp(-dt_jump / input.tau_mem) - exp(-dt_jump / input.tau_syn));
                }
            }

            vadd_scaled_simd(mv_row_mut(grad, spike_source), ctx.lambda_i,
                             -input.tau_syn, input.n_post);

            /* Compute dot product: w[row] · (lambda_v - lambda_i)
               Use stack allocation for small n_post, heap otherwise. */
            double stack_diff[BACKWARD_DIFF_STACK_SIZE];
            int use_stack = (input.n_post <= BACKWARD_DIFF_STACK_SIZE);
            double* diff = use_stack ? stack_diff : malloc((size_t)input.n_post * sizeof(double));

            vsub_simd(diff, ctx.lambda_v, ctx.lambda_i, input.n_post);
            double outbound = dot_product_simd(mv_row(w, spike_source), diff, input.n_post);
            ctx.all_errors[spike_idx] += outbound;

            if (!use_stack) free(diff);

            /* Write spike error to alloc buffer */
            result.spike_errors[spike_idx] = ctx.all_errors[spike_idx];
        } else {
            double current = spike.currents[spike_idx];
            double inv_ic = 1.0 / (current - input.v_th);
            int nrn = spike_source;  /* post-spike: spike_source is the post-synaptic neuron */
            double post_err = ctx.all_errors[input.n_spikes + spike_idx];

            /* Record jump for this neuron (with capacity management) */
            if (ctx.jump_count[nrn] >= ctx.jump_cap[nrn]) {
                ctx.jump_cap[nrn] *= 2;
                double* new_time = realloc(ctx.jump_time[nrn], (size_t)ctx.jump_cap[nrn] * sizeof(double));
                if (!new_time) goto cleanup_and_return;
                double* new_mag = realloc(ctx.jump_mag[nrn], (size_t)ctx.jump_cap[nrn] * sizeof(double));
                if (!new_mag) {
                    free(new_time);
                    goto cleanup_and_return;
                }
                /* Only update ctx pointers after both succeed */
                ctx.jump_time[nrn] = new_time;
                ctx.jump_mag[nrn] = new_mag;
            }
            ctx.jump_time[nrn][ctx.jump_count[nrn]] = t_bwd;
            ctx.jump_mag[nrn][ctx.jump_count[nrn]]  = inv_ic * (input.v_th * ctx.lambda_v[nrn] + post_err);
            ctx.jump_count[nrn]++;

            ctx.lambda_v[nrn] = (current / (current - input.v_th)) * ctx.lambda_v[nrn] + inv_ic * post_err;
        }
        previous_t = t_bwd;
    }

cleanup_and_return:
    free_backward_spikes_ctx(&ctx);
    if (result.error_no != EVENTPROP_OK) {
        eventprop_free_backward_output(result);
    }
    return result;
}

/** ── Error string helper ─────────────────────────────────────────────── */

const char* eventprop_error_string(const int error_no)
{
    switch (error_no) {
        case EVENTPROP_OK:                      return "OK";
        case EVENTPROP_INVALID_DIMENSION:       return "Invalid dimension: n_pre or n_post must be positive";
        case EVENTPROP_NEGATIVE_SPIKES:         return "Negayestive spike count";
        case EVENTPROP_NULL_POINTER:            return "NULL pointer: w, times, or sources is NULL when required";
        case EVENTPROP_INVALID_SOURCE:          return "Invalid source neuron index (out of [0, n_pre) range)";
        case EVENTPROP_INVALID_TAU:             return "Invalid tau: tau_mem and tau_syn must be positive and distinct";
        case EVENTPROP_INVALID_CURRENT:         return "Invalid current: spike current equals threshold (division by zero)";
        case EVENTPROP_NEGATIVE_SPIKE_TIME:     return "Negative spike time";
        case EVENTPROP_EXCESSIVE_SPIKE_TIME:    return "Spike time exceeds maximum allowed value";
        case EVENTPROP_TAU_TOO_SMALL:           return "Tau too small (below minimum threshold)";
        case EVENTPROP_TAU_TOO_LARGE:           return "Tau too large (above maximum threshold)";
        case EVENTPROP_INVALID_VTH:             return "Invalid spike threshold: v_th must be positive";
        case EVENTPROP_DIMENSION_TOO_LARGE:     return "Dimension exceeds maximum allowed size";
        case EVENTPROP_TIMES_TAU_MISMATCH:      return "Spike time and tau mismatch would cause exponential overflow";
        case EVENTPROP_TAU_TOO_CLOSE:           return "tau_mem and tau_syn are too close together";
        case EVENTPROP_BATCHSIZE_ZERO_OR_NEG:   return "Batch size must be positive";
        case EVENTPROP_ALLOC_SUM0:              return "Allocation failed: sum0 array";
        case EVENTPROP_ALLOC_SUM1:              return "Allocation failed: sum1 array";
        case EVENTPROP_ALLOC_EXP_MEM:           return "Allocation failed: exp_mem lookup table";
        case EVENTPROP_ALLOC_EXP_SYN:           return "Allocation failed: exp_syn lookup table";
        case EVENTPROP_ALLOC_TIMES_ARR:         return "Allocation failed: post_times array";
        case EVENTPROP_ALLOC_CURRENTS_ARR:      return "Allocation failed: post_currents array";
        case EVENTPROP_ALLOC_SPIKE_CAPS:        return "Allocation failed: spike_caps array";
        case EVENTPROP_ALLOC_SPIKE_COUNTS:      return "Allocation failed: spike_counts array";
        case EVENTPROP_ALLOC_NEURON_BUF:        return "Allocation failed: per-neuron times/currents buffer";
        case EVENTPROP_ALLOC_ENTRIES:           return "Allocation failed: entries (sorted collection)";
        case EVENTPROP_ALLOC_RESULT_TIMES:      return "Allocation failed: result spike times";
        case EVENTPROP_ALLOC_RESULT_SOURCES:    return "Allocation failed: result spike sources";
        case EVENTPROP_ALLOC_RESULT_CURRENTS:   return "Allocation failed: result spike currents";
        case EVENTPROP_ALLOC_RESULT_ERRORS:     return "Allocation failed: result errors buffer";
        case EVENTPROP_ALLOC_MAXIMA_TIMES:      return "Allocation failed: maxima times array";
        case EVENTPROP_ALLOC_MAXIMA_VALUES:     return "Allocation failed: maxima values array";
        case EVENTPROP_ALLOC_MAXIMA_ERRORS:     return "Allocation failed: maxima errors array";
        case EVENTPROP_ALLOC_BWD_GRAD:          return "Allocation failed: backward gradient matrix";
        case EVENTPROP_ALLOC_BWD_SPIKE_ERRORS:  return "Allocation failed: backward spike errors buffer";
        case EVENTPROP_ALLOC_BWD_MERGED:        return "Allocation failed: backward merged spike array";
        case EVENTPROP_ALLOC_BWD_ALL_ERRORS:    return "Allocation failed: backward all_errors buffer";
        case EVENTPROP_ALLOC_BWD_LAMBDA_V:      return "Allocation failed: backward lambda_v array";
        case EVENTPROP_ALLOC_BWD_LAMBDA_I:      return "Allocation failed: backward lambda_i array";
        case EVENTPROP_ALLOC_BWD_JUMP_TIME:     return "Allocation failed: backward jump_time array";
        case EVENTPROP_ALLOC_BWD_JUMP_MAG:      return "Allocation failed: backward jump_mag array";
        case EVENTPROP_ALLOC_BWD_JUMP_COUNT:    return "Allocation failed: backward jump_count array";
        case EVENTPROP_ALLOC_BWD_JUMP_CAP:      return "Allocation failed: backward jump_cap array";
        case EVENTPROP_ALLOC_BWD_PER_NEURON:    return "Allocation failed: per-neuron backward buffer";
        case EVENTPROP_ALLOC_BATCH_OUTPUTS:     return "Allocation failed: batch outputs array";
        case EVENTPROP_ALLOC_BATCH_GRAD:        return "Allocation failed: accumulated gradient pointer";
        case EVENTPROP_ALLOC_DROPOUT_SAMPLE:    return "Allocation failed: dropout per-sample buffer";
        case EVENTPROP_ALLOC_DIAG_V_TRACE:      return "Allocation failed: voltage trace array";
        case EVENTPROP_ALLOC_DIAG_TS:           return "Allocation failed: time steps array";
        case EVENTPROP_ALLOC_DIAG_LAMBDA:       return "Allocation failed: lambda-I trace array";
        case EVENTPROP_BATCH_EMPTY:             return "Batch input was empty or NULL";
        default:                                return "Unknown error";
    }
}
