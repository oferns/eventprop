/**
 * @file eventprop.c
 * @brief EventProp: Event-based backpropagation for SNN training.
 *
 * Consolidated implementation with automatic memory allocation (alloc versions).
 * Includes legacy diagnostic functions for compatibility.
 */

#include "eventprop.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <errno.h>

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

#define COMPUTE_V_INLINE(t, i, spike_count, post_spike_times, neuron_offset, \
                        sum0, sum1, target_nrn, input, k_prefactor) \
({ \
    double mem = k_prefactor * (sum1[i * input->n_post + target_nrn] * exp(-t / input->tau_mem) \
                    - sum0[i * input->n_post + target_nrn] * exp(-t / input->tau_syn)); \
    for (int s = 0; s < spike_count; s++) { \
        double st = post_spike_times[neuron_offset + s]; \
        if (st <= t) {  /* Include all spikes where st <= t, don't break early */ \
            mem -= input->v_th * exp(-(t - st) / input->tau_mem); \
        } \
    } \
    mem; \
})

/* Factored version: exp(-(t-st)*tau-1) = exp(-t*tau-1) × exp(st*tau-1)
 * Precomputed post_spike_exp_st[s] = exp(post_spike_times[neuron_offset+s] * tau_mem_inv).
 * Eliminates N exp() calls per iteration → just 1 (shared with base terms). */
#define COMPUTE_V_FACTORED(t, i, spike_count, post_spike_times, neuron_offset, \
                         sum0, sum1, target_nrn, n_post, k_prefactor, tau_mem_inv, \
                         tau_syn_inv, v_th, post_spike_exp_st) \
({ \
    double e_t = exp(-(t) * tau_mem_inv); \
    double mem = k_prefactor * (sum1[i * n_post + target_nrn] * e_t \
                    - sum0[i * n_post + target_nrn] * exp(-(t) * tau_syn_inv)); \
    if (post_spike_exp_st) { \
        for (int s = 0; s < spike_count; s++) { \
            double st = post_spike_times[neuron_offset + s]; \
            if (st <= t) { \
                mem -= v_th * e_t * post_spike_exp_st[s]; \
            } \
        } \
    } else { \
        for (int s = 0; s < spike_count; s++) { \
            double st = post_spike_times[neuron_offset + s]; \
            if (st <= t) { \
                mem -= v_th * exp(-(t - st) * tau_mem_inv); \
            } \
        } \
    } \
    mem; \
})

/* ── Configuration constants ─────────────────────────────────────────── */
#define EXP_STACK_SIZE 1024
#define DIFF_STACK_SIZE 512

#define TOMS748_MAX_ITER 1024
#define TOMS748_TOL 1e-14

typedef double (*RootFinderFn)(void* ctx, double t);

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
    output->n_spikes = 0;
}
void eventprop_free_sums_output(SumsOutput* output)
{
    if (!output) return;
    free(output->sum0);
    free(output->sum1);
}
void eventprop_free_maxima_output(MaximaOutput* output)
{
    if (!output) return;
    free(output->times);
    free(output->values);
}
void eventprop_free_backward_output(BackwardOutput* output)
{
    if (!output) return;
    free(output->grad);
    free(output->spike_errors);
}
void eventprop_free_backward_batch_output(BackwardOutput* output, int batch_size)
{
    if (!output) return;
    for (int i = 0; i < batch_size; i++) {
        eventprop_free_backward_output(&output[i]);
    }
    free(output);
}
void eventprop_free_batch_spike_output(SpikeOutput* output, int batch_size)
{
    if (!output) return;
    for (int i = 0; i < batch_size; i++) {
        eventprop_free_spike_output(&output[i]);
    }
    free(output);
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
        double result[8];
#if defined(HAVE_AVX512F)
        _mm512_storeu_pd(result, acc);
#else
        _mm256_storeu_pd(result, acc);
#endif
        double sum = result[0] + result[1];
#if defined(HAVE_AVX512F)
        sum += result[2] + result[3] + result[4] + result[5] + result[6] + result[7];
#else
        sum += result[2] + result[3];
#endif
        for (int j = vec_count * vec; j < n; j++)
            sum += a[j] * b[j];
        return sum;
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

/* ── Helper functions (MACROS) for compute_spikes ─── */



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

static double root_fn(void* ctx_ptr, double t) {
    RootContext* c = (RootContext*)ctx_ptr;
    return COMPUTE_V_FACTORED(t, c->i, c->spike_count, c->post_spike_times, c->neuron_offset,
                     c->sum0, c->sum1, c->target_nrn, c->n_post,
                     c->k_prefactor, c->tau_mem_inv, c->tau_syn_inv, c->v_th,
                     c->post_spike_exp_st) - c->v_th;
}

static double toms748_solve(void* ctx, RootFinderFn fn, double a, double b) {

    double fa = fn(ctx, a), fb = fn(ctx, b);
    if (!isfinite(fa) || !isfinite(fb)) return a;

    /* Precondition: bracket must contain sign change */
    if (fa * fb > 0) {
        fprintf(stderr, "toms748_solve: invalid bracket at a=%.10f, b=%.10f (fa=%.2e, fb=%.2e)\n",
                a, b, fa, fb);
        errno = EDOM;  // Domain error - root not bracketed
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
        double prevB = b; b = s; fb = fn(ctx, b);
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

SpikeOutput eventprop_compute_spikes(const EventPropInput* input)
{
    SpikeOutput result;

    /* Use dynamic arrays internally, then copy to exact-size output */
    double k_prefactor = input->tau_syn / (input->tau_mem - input->tau_syn);
    double tmax_prefactor = 1.0 / (1.0 / input->tau_mem - 1.0 / input->tau_syn);
    double tmax_summand = log(input->tau_syn / input->tau_mem);
    double tau_mem_inv = 1.0 / input->tau_mem;
    double tau_syn_inv = 1.0 / input->tau_syn;

    /* Compute sums */
    SumsOutput sums = eventprop_compute_sums(input);
    const double* sum0 = sums.sum0;
    const double* sum1 = sums.sum1;

    /* Per-neuron dynamic buffers - growable to handle burst patterns */
    double** post_times_arr   = malloc(input->n_post * sizeof(double*));
    double** post_currents_arr = malloc(input->n_post * sizeof(double*));
    int* spike_caps  = calloc(input->n_post, sizeof(int));
    int* spike_counts = calloc(input->n_post, sizeof(int));

    /* Start with 3× estimated spikes to reduce realloc overhead */
    int initial_cap = input->n_spikes * 3 + 1;
    for (int n = 0; n < input->n_post; n++) {
        post_times_arr[n]   = calloc(initial_cap, sizeof(double));
        post_currents_arr[n] = calloc(initial_cap, sizeof(double));
        spike_caps[n]       = initial_cap;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int target_nrn = 0; target_nrn < input->n_post; target_nrn++) {
        if (target_nrn == 0) {
        }

        double* p_times   = post_times_arr[target_nrn];
        double* p_currents = post_currents_arr[target_nrn];
        int cap = spike_caps[target_nrn];
        int count = 0;
        int finished = 0, processed = 0;

        while (!finished) {
            int found = 0;
            for (int i = processed; i <= input->n_spikes; i++) {
                if (i == input->n_spikes) {
                    /* Last interval */
                    double s0 = sum0[input->n_spikes * input->n_post + target_nrn];
                    double s1 = sum1[input->n_spikes * input->n_post + target_nrn];
                    for (int s = 0; s < count; s++)
                        s1 += -input->v_th * exp(p_times[s] / input->tau_mem);

                    double tmax = tmax_prefactor * (tmax_summand + log(s1 / s0));
                    if (!isinf(tmax)) {
                        double v = COMPUTE_V_INLINE(tmax, i, count, p_times, 0, sum0, sum1,
                                             target_nrn, input, k_prefactor);
                        if (v > input->v_th + 1e-14) {
                            double t_lower = count > 0 ? p_times[count - 1] : 0.0;
                            RootContext rctx; memset(&rctx, 0, sizeof(rctx));
                            /* Precompute exp(st * tau_mem_inv) for each post-spike → factorization */
                            double* post_exp_st = NULL;
                            if (count > 0) {
                                post_exp_st = (double*)malloc(count * sizeof(double));
                                for (int s = 0; s < count; s++)
                                    post_exp_st[s] = exp(p_times[s] * tau_mem_inv);
                            }
                            rctx.sum0=sum0; rctx.sum1=sum1; rctx.i=i; rctx.spike_count=count;
                            rctx.post_spike_times=p_times; rctx.neuron_offset=0;
                            rctx.target_nrn=target_nrn; rctx.n_post=input->n_post;
                            rctx.k_prefactor=k_prefactor; rctx.tau_mem_inv=tau_mem_inv;
                            rctx.tau_syn_inv=tau_syn_inv; rctx.v_th=input->v_th;
                            rctx.post_spike_exp_st = post_exp_st;
                            double t_post = toms748_solve(&rctx, root_fn, t_lower, tmax);
                            free(post_exp_st);
                            if (t_post > t_lower && t_post < tmax) {
                                if (count >= cap) {
                                    cap *= 3;
                                    p_times    = realloc(p_times, cap * sizeof(double));
                                    p_currents = realloc(p_currents, cap * sizeof(double));
                                }
                                p_times[count]   = t_post;
                                p_currents[count] = compute_i(t_post, input->n_spikes, input->times, sum0,
                                                              target_nrn, input->n_post, input->tau_syn);
                                count++; found = 1;
                            }
                        }
                    }
                    if (!found) finished = 1;
                    else processed = input->n_spikes;
                } else {
                    double t_input = input->times[i];
                    double tmax = compute_tmax(i, input->n_spikes, input->times, count, p_times, 0,
                                               sum0, sum1, target_nrn, input->n_post,
                                               tmax_prefactor, tmax_summand, input->tau_mem, input->v_th);

                    if (!isinf(tmax)) {
                        double v = COMPUTE_V_INLINE(tmax, i, count, p_times, 0, sum0, sum1,
                                             target_nrn, input, k_prefactor);
                        if (v > input->v_th + 1e-14) {
                            double t_before = (i > 0) ? input->times[i-1] : 0.0;
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
                                for (int s = 0; s < count; s++)
                                    post_exp_st[s] = exp(p_times[s] * tau_mem_inv);
                            }
                            rctx.sum0=sum0; rctx.sum1=sum1; rctx.i=i; rctx.spike_count=count;
                            rctx.post_spike_times=p_times; rctx.neuron_offset=0;
                            rctx.target_nrn=target_nrn; rctx.n_post=input->n_post;
                            rctx.k_prefactor=k_prefactor; rctx.tau_mem_inv=tau_mem_inv;
                            rctx.tau_syn_inv=tau_syn_inv; rctx.v_th=input->v_th;
                            rctx.post_spike_exp_st = post_exp_st;
                            double t_post = toms748_solve(&rctx, root_fn, t_before, tmax);
                            free(post_exp_st);
                            if (t_post > t_before && t_post < tmax) {
                                if (count >= cap) {
                                    cap *= 3;
                                    p_times    = realloc(p_times, cap * sizeof(double));
                                    p_currents = realloc(p_currents, cap * sizeof(double));
                                }
                                p_times[count]   = t_post;
                                p_currents[count] = compute_i(t_post, input->n_spikes, input->times, sum0,
                                                              target_nrn, input->n_post, input->tau_syn);
                                count++; found = 1;
                            }
                            processed = i; break;
                        }
                    }

                    /* Check at input spike time */
                    double v_at_input = COMPUTE_V_INLINE(t_input, i, count, p_times, 0,
                                                          sum0, sum1, target_nrn, input,
                                                          k_prefactor);
                    if (v_at_input > input->v_th + 1e-14) {
                        double t_before = (i > 0) ? input->times[i-1] : 0.0;
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
                            for (int s = 0; s < count; s++)
                                post_exp_st[s] = exp(p_times[s] * tau_mem_inv);
                        }
                        rctx.sum0=sum0; rctx.sum1=sum1; rctx.i=i; rctx.spike_count=count;
                        rctx.post_spike_times=p_times; rctx.neuron_offset=0;
                        rctx.target_nrn=target_nrn; rctx.n_post=input->n_post;
                        rctx.k_prefactor=k_prefactor; rctx.tau_mem_inv=tau_mem_inv;
                        rctx.tau_syn_inv=tau_syn_inv; rctx.v_th=input->v_th;
                        rctx.post_spike_exp_st = post_exp_st;
                        double t_post = toms748_solve(&rctx, root_fn, t_before, t_input);
                        free(post_exp_st);
                        if (t_post > t_before && t_post < t_input) {
                            if (count >= cap) {
                                cap *= 3;
                                p_times    = realloc(p_times, cap * sizeof(double));
                                p_currents = realloc(p_currents, cap * sizeof(double));
                            }
                            p_times[count]   = t_post;
                            p_currents[count] = compute_i(t_post, input->n_spikes, input->times, sum0,
                                                          target_nrn, input->n_post, input->tau_syn);
                            count++; found = 1;
                        }
                        processed = i; break;
                    }
                }
            }
            if (!found) processed = (processed + 1 < input->n_spikes) ? processed + 1 : input->n_spikes;
        }

        /* Write back updated pointers */
        post_times_arr[target_nrn]   = p_times;
        post_currents_arr[target_nrn] = p_currents;
        spike_caps[target_nrn]       = cap;
        spike_counts[target_nrn]     = count;
    }


    /* Count total spikes and allocate exact-size output */
    result.n_spikes = 0;
    for (int n = 0; n < input->n_post; n++)
        result.n_spikes += spike_counts[n];


    if (result.n_spikes == 0) {
        result.times = NULL;
        result.sources = NULL;
        result.currents = NULL;

        for (int n = 0; n < input->n_post; n++) {
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
    int idx = 0;
    for (int n = 0; n < input->n_post; n++) {
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
    result.sources = malloc(result.n_spikes * sizeof(int));
    result.currents = malloc(result.n_spikes * sizeof(double));

    for (int i = 0; i < result.n_spikes; i++) {
        result.times[i]     = entries[i].time;
        result.sources[i]   = entries[i].source;
        result.currents[i]  = entries[i].current;
    }

    /* Cleanup */
    for (int n = 0; n < input->n_post; n++) {
        free(post_times_arr[n]);
        free(post_currents_arr[n]);
    }
    eventprop_free_sums_output(&sums);
    free(post_times_arr); free(post_currents_arr);
    free(spike_caps); free(spike_counts);
    free(entries);

    return result;
}

SumsOutput eventprop_compute_sums(const EventPropInput* input)
{
    SumsOutput result;
    int n_rows = input->n_spikes + 1;
    result.n_rows = n_rows;
    result.n_post = input->n_post;

    /* Allocate output - exact size known upfront */
    result.sum0 = calloc(n_rows * input->n_post, sizeof(double));
    result.sum1 = calloc(n_rows * input->n_post, sizeof(double));

    if (input->n_spikes == 0) return result;

    /* Pre-compute exponential lookup tables */
    double* exp_mem = calloc(input->n_spikes, sizeof(double));
    double* exp_syn = calloc(input->n_spikes, sizeof(double));

    for (int i = 0; i < input->n_spikes; i++) {
        exp_mem[i] = exp(input->times[i] / input->tau_mem);
        exp_syn[i] = exp(input->times[i] / input->tau_syn);
    }

    /* Accumulate sums using SIMD-accelerated FMA */
    for (int i = 0; i < input->n_spikes; i++) {
        int src = input->sources[i];
        fma_row_simd(result.sum0 + (i + 1) * input->n_post,
                           result.sum0 + i * input->n_post,
                           input->w + src * input->n_post, exp_syn[i], input->n_post);
        fma_row_simd(result.sum1 + (i + 1) * input->n_post,
                           result.sum1 + i * input->n_post,
                           input->w + src * input->n_post, exp_mem[i], input->n_post);
    }

    free(exp_mem);
    free(exp_syn);

    return result;
}

MaximaOutput eventprop_compute_maxima(const EventPropInput* input)
{
    MaximaOutput result;
    result.n_post = input->n_post;

    /* Allocate output - always exactly n_post elements */
    result.times   = calloc(input->n_post, sizeof(double));
    result.values  = calloc(input->n_post, sizeof(double));

    /* Initialize maxima to "not found" */
    for (int i = 0; i < input->n_post; i++) {
        result.times[i]   = NAN;
        result.values[i]  = -INFINITY;
    }

    /* Use our own alloc function */
    SumsOutput sums = eventprop_compute_sums(input);

    int n_rows = sums.n_rows;
    double* sum0 = sums.sum0;
    double* sum1 = sums.sum1;

    /* Find maxima for each post-synaptic neuron */
    double k_pref = input->tau_syn / (input->tau_mem - input->tau_syn);
    double tmax_pref = 1.0 / (1.0/input->tau_mem - 1.0/input->tau_syn);
    double tmax_sum = log(input->tau_syn / input->tau_mem);

    /* No spikes → nothing to compute, return NaN/-INF (matches Eigen) */
    if (input->n_spikes == 0) {
        eventprop_free_sums_output(&sums);
        return result;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int target_nrn = 0; target_nrn < input->n_post; target_nrn++) {
        for (int i = 0; i <= input->n_spikes; i++) {
            double t_input = (i == input->n_spikes) ? INFINITY : input->times[i];
            double s0 = sum0[i * input->n_post + target_nrn];
            double s1 = sum1[i * input->n_post + target_nrn];

            /* Analytical tmax from log(s1/s0) */
            if (s0 > 0 && s1 > 0) {
                double tmax = tmax_pref * (tmax_sum + log(s1 / s0));

                if (!isinf(tmax) && (i == 0 || input->times[i-1] <= tmax) &&
                    (i == input->n_spikes || tmax <= t_input)) {
                    double vmax = k_pref * (s1 * exp(-tmax/input->tau_mem) -
                                            s0 * exp(-tmax/input->tau_syn));
                    if (vmax > result.values[target_nrn]) {
                        result.values[target_nrn] = vmax;
                        result.times[target_nrn]  = tmax;
                    }
                }
            }

            /* Also check voltage at input spike time */
            if (t_input < INFINITY) {
                double v_at_input = k_pref * (s1 * exp(-t_input/input->tau_mem) -
                                              s0 * exp(-t_input/input->tau_syn));
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

BackwardOutput eventprop_backward_maxima(const BackwardMaximaInput* input)
{
    BackwardOutput result;

    result.n_pre = input->input->n_pre;
    result.n_post = input->input->n_post;

    /* Allocate output - exact sizes known upfront */
    result.grad = calloc(input->input->n_pre * input->input->n_post, sizeof(double));
    result.spike_errors = calloc(input->input->n_spikes, sizeof(double));

    if (input->input->n_spikes == 0 || input->n_maxima == 0)
        return result;

    double largest_time = input->input->times[input->input->n_spikes - 1];
    double k_bwd_prefactor = input->input->tau_mem / (input->input->tau_mem - input->input->tau_syn);

    /* Precompute inverse time constants */
    double tau_mem_inv = 1.0 / input->input->tau_mem;
    double tau_syn_inv = 1.0 / input->input->tau_syn;

    /* Sweep backward through input spikes */
    for (int spike_idx = input->input->n_spikes - 1; spike_idx >= 0; spike_idx--) {
        double t_bwd = largest_time - input->input->times[spike_idx];

        for (int max_idx = 0; max_idx < input->n_maxima; max_idx++) {
            if (isnan(input->maxima_times[max_idx]))
                continue;

            double t_vmax_bwd = largest_time - input->maxima_times[max_idx];
            if (t_vmax_bwd > t_bwd)
                continue;

            double dt_mem = -(t_bwd - t_vmax_bwd) * tau_mem_inv;
            double dt_syn = -(t_bwd - t_vmax_bwd) * tau_syn_inv;

            double lambda_v = -exp(dt_mem) * input->maxima_errors[max_idx] * tau_mem_inv;
            double k_bwd_val = k_bwd_prefactor * (exp(dt_mem) - exp(dt_syn));
            double lambda_i = -k_bwd_val * input->maxima_errors[max_idx] * tau_mem_inv;

            int source_nrn = input->input->sources[spike_idx];
            result.grad[source_nrn * input->input->n_post + max_idx] -= input->input->tau_syn * lambda_i;

            result.spike_errors[spike_idx] += lambda_v - lambda_i;
        }
    }

    return result;
}

BackwardOutput eventprop_backward_spikes(const BackwardSpikesInput* input)
{
    BackwardOutput result;

    /* Allocate output - exact sizes known upfront */
    result.n_pre = input->input->n_pre;
    result.n_post = input->input->n_post;
    result.grad = calloc(input->input->n_pre * input->input->n_post, sizeof(double));
    result.spike_errors = calloc(input->input->n_spikes, sizeof(double));

    if (input->n_post_spikes == 0)
        return result;

    int total = input->input->n_spikes + input->n_post_spikes;
    MergedSpike* merged = malloc(total * sizeof(MergedSpike));

    for (int i = 0; i < input->input->n_spikes; i++) { merged[i].time=input->input->times[i]; merged[i].idx=i; merged[i].is_input=1; }
    for (int i = 0; i < input->n_post_spikes; i++) { merged[input->input->n_spikes+i].time=input->fwd_times[i]; merged[input->input->n_spikes+i].idx=i; merged[input->input->n_spikes+i].is_input=0; }

    /* Sort descending by time: O(n log n) via qsort */
    qsort(merged, total, sizeof(MergedSpike), cmp_merged_desc);

    double* all_errors = calloc(total, sizeof(double));
    for (int i = 0; i < input->n_post_spikes; i++) all_errors[input->input->n_spikes + i] = input->post_errors[i];

    double* lambda_v = calloc(input->input->n_post, sizeof(double));
    double* lambda_i = malloc(input->input->n_post * sizeof(double));

    /* Per-neuron jump storage (matching Eigen's vector<vector<pair<double,double>>> structure) */
    double** jump_time = calloc(input->input->n_post, sizeof(double*));
    double** jump_mag  = calloc(input->input->n_post, sizeof(double*));
    int*   jump_count  = calloc(input->input->n_post, sizeof(int));
    int*   jump_cap    = calloc(input->input->n_post, sizeof(int));
    for (int i = 0; i < input->input->n_post; i++) {
        jump_cap[i] = 8;  /* initial capacity */
        jump_time[i] = malloc(jump_cap[i] * sizeof(double));
        jump_mag[i]  = malloc(jump_cap[i] * sizeof(double));
    }

    double largest_time = merged[0].time, previous_t = largest_time;
    double k_bwd = input->input->tau_mem / (input->input->tau_mem - input->input->tau_syn);

    for (int s = 0; s < total; s++) {
        int spike_idx = merged[s].idx, is_input = merged[s].is_input;
        double spike_time = merged[s].time;
        int spike_source = is_input ? input->input->sources[spike_idx] : input->fwd_sources[spike_idx];

        double t_bwd = largest_time - spike_time;
        double dt = previous_t - t_bwd;
        scale_inplace_simd(lambda_v, exp(dt / input->input->tau_mem), input->input->n_post);

        if (is_input) {
            /* Compute lambda_i for ALL post-synaptic neurons (matching Eigen structure) */
            memset(lambda_i, 0, input->input->n_post * sizeof(double));
            for (int nrn = 0; nrn < input->input->n_post; nrn++) {
                for (int j = 0; j < jump_count[nrn]; j++) {
                    double dt_jump = t_bwd - jump_time[nrn][j];
                    lambda_i[nrn] += jump_mag[nrn][j] * k_bwd * (exp(-dt_jump / input->input->tau_mem) - exp(-dt_jump / input->input->tau_syn));
                }
            }

            vadd_scaled_simd(result.grad + spike_source * input->input->n_post, lambda_i, -input->input->tau_syn, input->input->n_post);

            /* Compute dot product: w[row] · (lambda_v - lambda_i)
               Use stack allocation for small n_post, heap otherwise. */
#define BACKWARD_DIFF_STACK_SIZE 512
            double stack_diff[BACKWARD_DIFF_STACK_SIZE];
            int use_stack = (input->input->n_post <= BACKWARD_DIFF_STACK_SIZE);
            double* diff = use_stack ? stack_diff : malloc(input->input->n_post * sizeof(double));

            vsub_simd(diff, lambda_v, lambda_i, input->input->n_post);
            double outbound = dot_product_simd(input->input->w + spike_source * input->input->n_post, diff, input->input->n_post);
            all_errors[spike_idx] += outbound;

            if (!use_stack) free(diff);

            /* Write spike error to alloc buffer */
            result.spike_errors[spike_idx] = all_errors[spike_idx];
        } else {
            double current = input->fwd_currents[spike_idx];
            double inv_ic = 1.0 / (current - input->input->v_th);
            int nrn = spike_source;  /* post-spike: spike_source is the post-synaptic neuron */
            double post_err = all_errors[input->input->n_spikes + spike_idx];

            /* Record jump for this neuron (with capacity management) */
            if (jump_count[nrn] >= jump_cap[nrn]) {
                jump_cap[nrn] *= 2;
                jump_time[nrn] = realloc(jump_time[nrn], jump_cap[nrn] * sizeof(double));
                jump_mag[nrn]  = realloc(jump_mag[nrn],  jump_cap[nrn] * sizeof(double));
            }
            jump_time[nrn][jump_count[nrn]] = t_bwd;
            jump_mag[nrn][jump_count[nrn]]  = inv_ic * (input->input->v_th * lambda_v[nrn] + post_err);
            jump_count[nrn]++;

            lambda_v[nrn] = (current / (current - input->input->v_th)) * lambda_v[nrn] + inv_ic * post_err;
        }
        previous_t = t_bwd;
    }

    for (int i = 0; i < input->input->n_post; i++) {
        free(jump_time[i]); free(jump_mag[i]);
    }
    free(merged); free(all_errors); free(lambda_v); free(lambda_i);
    free(jump_time); free(jump_mag); free(jump_count); free(jump_cap);

    return result;
}

BackwardOutput* eventprop_backward_spikes_batch(const BatchEventPropInput* input, SpikeOutput* fwd_batch, const double* post_errors)
{
    int batch_size = input->batch_size;
    const EventPropInput* base = input->input;
    const int* n_spikes_per_sample = input->n_spikes_per_sample;

    /* Allocate array of BackwardOutput structs */
    BackwardOutput* result = (BackwardOutput*)calloc(batch_size, sizeof(BackwardOutput));

    /* Compute input sample offsets and forward error offsets */
    int* offsets = (int*)malloc(batch_size * sizeof(int));
    int* err_offsets = (int*)malloc(batch_size * sizeof(int));
    offsets[0] = 0;
    err_offsets[0] = 0;
    for (int i = 1; i < batch_size; i++) {
        offsets[i] = offsets[i-1] + n_spikes_per_sample[i-1];
        err_offsets[i] = err_offsets[i-1] + fwd_batch[i-1].n_spikes;
    }

    /* Process each sample in parallel */
#ifdef _OPENMP
   #pragma omp parallel for schedule(static) shared(base, offsets, err_offsets, n_spikes_per_sample,\
            batch_size, result, fwd_batch, post_errors) \
          if(batch_size >= 2 || batch_size * base->n_pre >= 50)
#endif
    for (int b = 0; b < batch_size; b++) {
        int offset = offsets[b];
        int n_spikes = n_spikes_per_sample[b];

        /* Extract forward spike data directly from SpikeOutput */
        int n_fwd_spikes = fwd_batch[b].n_spikes;

        EventPropInput fwbase = {
            .w=base->w, .n_pre=base->n_pre, .n_post=base->n_post,
            .times=base->times + offset, .sources=base->sources + offset, .n_spikes=n_spikes,
            .tau_mem=base->tau_mem, .tau_syn=base->tau_syn, .v_th=base->v_th
        };
        BackwardSpikesInput backward_input = {
            .input = &fwbase,
            .fwd_times=fwd_batch[b].times,
            .fwd_sources=fwd_batch[b].sources,
            .fwd_currents=fwd_batch[b].currents,
            .post_errors=post_errors + err_offsets[b],
            .n_post_spikes=n_fwd_spikes,
        };
        result[b] = eventprop_backward_spikes(&backward_input);
    }

    free(offsets);
    free(err_offsets);
    return result;
}

SpikeOutput* eventprop_compute_spikes_batch(const BatchEventPropInput* input)
{
    int batch_size = input->batch_size;
    const EventPropInput* base = input->input;
    const int* n_spikes_per_sample = input->n_spikes_per_sample;

    /* Allocate array of SpikeOutput structs */
    SpikeOutput* result = (SpikeOutput*)calloc(batch_size, sizeof(SpikeOutput));

    /* Compute sample offsets */
    int* offsets = (int*)malloc(batch_size * sizeof(int));
    offsets[0] = 0;
    for (int i = 1; i < batch_size; i++) {
        offsets[i] = offsets[i-1] + n_spikes_per_sample[i-1];
    }

    /* Process each sample in parallel */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) shared(base, offsets, n_spikes_per_sample, \
         batch_size, result) if(batch_size >= 2 || batch_size * base->n_pre >= 50)
#endif
    for (int b = 0; b < batch_size; b++) {
        int offset = offsets[b];
        int n_spikes = n_spikes_per_sample[b];

        /* Call single-sample function for this batch element */
        EventPropInput sample_input = {
            .w=base->w, .n_pre=base->n_pre, .n_post=base->n_post,
            .times=base->times + offset, .sources=base->sources + offset, .n_spikes=n_spikes,
            .tau_mem=base->tau_mem, .tau_syn=base->tau_syn,
            .v_th = base->v_th
        };
        result[b] = eventprop_compute_spikes(&sample_input);
    }
    free(offsets);
    return result;
}
