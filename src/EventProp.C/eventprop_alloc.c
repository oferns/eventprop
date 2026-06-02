/**
 * @file eventprop_alloc.c
 * @brief Automatic allocation wrappers for EventProp core functions.
 */

#include <math.h>
#include <stdlib.h>

/* SIMD headers */
#ifdef HAVE_AVX512F
    #include <immintrin.h>
#elif defined(HAVE_AVX2)
    #include <immintrin.h>
#endif
#ifdef __ARM_NEON
    #include <arm_neon.h>
#endif

#include "eventprop_alloc.h"

/* ── SIMD helpers (copied from eventprop_scalar.c) ─────────────── */

static int alloc_simd_vec_width(void)
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

static int alloc_simd_available(void)
{
    return alloc_simd_vec_width() > 0;
}

/* Fused multiply-add: dest[j] = src[j] + scalar * weights[j] */
static void alloc_fma_row_simd(double* dest, const double* src, const double* weights,
                               double scalar, int n)
{
    if (!alloc_simd_available() || n < 2) {
        for (int j = 0; j < n; j++)
            dest[j] = src[j] + scalar * weights[j];
        return;
    }

    int vec = alloc_simd_vec_width();
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

/* ── SpikeOutput ─────────────────────────────────────────────────────── */

SpikeOutput eventprop_compute_spikes_alloc(const double* w, int n_pre, int n_post,
                                           const double* times, const int32_t* sources, int n_spikes,
                                           double v_th, double tau_mem, double tau_syn)
{
    (void)w; (void)n_pre; (void)n_post;
    (void)times; (void)sources; (void)n_spikes;
    (void)v_th; (void)tau_mem; (void)tau_syn;

    SpikeOutput result = {0, NULL, NULL, NULL};
    /* TODO: implement */
    return result;
}

void eventprop_free_spike_output(SpikeOutput* output)
{
    if (!output) return;
    free(output->times);
    free(output->sources);
    free(output->currents);
    output->n_spikes = 0;
}

/* ── SumsOutput ─────────────────────────────────────────────────────── */

SumsOutput eventprop_compute_sums_alloc(const double* w, int n_pre, int n_post,
                                        const double* times, const int32_t* sources, int n_spikes,
                                        double tau_mem, double tau_syn)
{
    SumsOutput result;
    int n_rows = n_spikes + 1;
    result.n_rows = n_rows;
    result.n_post = n_post;

    /* Allocate output — exact size known upfront */
    result.sum0 = calloc(n_rows * n_post, sizeof(double));
    result.sum1 = calloc(n_rows * n_post, sizeof(double));

    if (n_spikes == 0) return result;

    /* Pre-compute exponential lookup tables */
    double* exp_mem = calloc(n_spikes, sizeof(double));
    double* exp_syn = calloc(n_spikes, sizeof(double));

    for (int i = 0; i < n_spikes; i++) {
        exp_mem[i] = exp(times[i] / tau_mem);
        exp_syn[i] = exp(times[i] / tau_syn);
    }

    /* Accumulate sums using SIMD-accelerated FMA */
    for (int i = 0; i < n_spikes; i++) {
        int src = sources[i];
        alloc_fma_row_simd(result.sum0 + (i + 1) * n_post,
                           result.sum0 + i * n_post,
                           w + src * n_post, exp_syn[i], n_post);
        alloc_fma_row_simd(result.sum1 + (i + 1) * n_post,
                           result.sum1 + i * n_post,
                           w + src * n_post, exp_mem[i], n_post);
    }

    free(exp_mem);
    free(exp_syn);

    return result;
}

void eventprop_free_sums_output(SumsOutput* output)
{
    if (!output) return;
    free(output->sum0);
    free(output->sum1);
}

/* ── MaximaOutput ───────────────────────────────────────────────────── */

MaximaOutput eventprop_compute_maxima_alloc(const double* w, int n_pre, int n_post,
                                           const double* times, const int32_t* sources, int n_spikes,
                                           double tau_mem, double tau_syn)
{
    MaximaOutput result;
    result.n_post = n_post;

    /* Allocate output — always exactly n_post elements */
    result.times   = calloc(n_post, sizeof(double));
    result.values  = calloc(n_post, sizeof(double));

    /* Initialize maxima to "not found" */
    for (int i = 0; i < n_post; i++) {
        result.times[i]   = NAN;
        result.values[i]  = -INFINITY;
    }

    /* Use our own alloc function */
    SumsOutput sums = eventprop_compute_sums_alloc(w, n_pre, n_post,
                                                    times, sources, n_spikes,
                                                    tau_mem, tau_syn);

    int n_rows = sums.n_rows;
    double* sum0 = sums.sum0;
    double* sum1 = sums.sum1;

    /* Find maxima for each post-synaptic neuron */
    double k_pref = tau_syn / (tau_mem - tau_syn);
    double tmax_pref = 1.0 / (1.0/tau_mem - 1.0/tau_syn);
    double tmax_sum = log(tau_syn / tau_mem);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(n_post >= 20)
#endif
    for (int target_nrn = 0; target_nrn < n_post; target_nrn++) {
        for (int i = 0; i <= n_spikes; i++) {
            double t_input = (i == n_spikes) ? INFINITY : times[i];
            double s0 = sum0[i * n_post + target_nrn];
            double s1 = sum1[i * n_post + target_nrn];

            /* Analytical tmax from log(s1/s0) */
            if (s0 > 0 && s1 > 0) {
                double tmax = tmax_pref * (tmax_sum + log(s1 / s0));

                if (!isinf(tmax) && (i == 0 || times[i-1] <= tmax) &&
                    (i == n_spikes || tmax <= t_input)) {
                    double vmax = k_pref * (s1 * exp(-tmax/tau_mem) -
                                            s0 * exp(-tmax/tau_syn));
                    if (vmax > result.values[target_nrn]) {
                        result.values[target_nrn] = vmax;
                        result.times[target_nrn]  = tmax;
                    }
                }
            }

            /* Also check voltage at input spike time */
            if (t_input < INFINITY) {
                double v_at_input = k_pref * (s1 * exp(-t_input/tau_mem) -
                                              s0 * exp(-t_input/tau_syn));
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

void eventprop_free_maxima_output(MaximaOutput* output)
{
    if (!output) return;
    free(output->times);
    free(output->values);
}

/* ── BackwardSpikesOutput ───────────────────────────────────────────── */

BackwardSpikesOutput eventprop_backward_spikes_alloc(const double* times, const int32_t* sources,
                                                     int n_spikes,
                                                     const double* fwd_times, const int32_t* fwd_sources,
                                                     const double* fwd_currents,
                                                     const double* post_errors, int n_post_spikes,
                                                     const double* w, int n_pre, int n_post,
                                                     double v_th, double tau_mem, double tau_syn)
{
    (void)times; (void)sources; (void)n_spikes;
    (void)fwd_times; (void)fwd_sources; (void)fwd_currents;
    (void)post_errors; (void)n_post_spikes;
    (void)w; (void)n_pre; (void)n_post;
    (void)v_th; (void)tau_mem; (void)tau_syn;

    BackwardSpikesOutput result = {0, 0, NULL, NULL};
    /* TODO: implement */
    return result;
}

void eventprop_free_backward_spikes_output(BackwardSpikesOutput* output)
{
    if (!output) return;
    free(output->grad);
    free(output->spike_errors);
}

/* ── BackwardMaximaOutput ───────────────────────────────────────────── */

BackwardMaximaOutput eventprop_backward_maxima_alloc(const double* input_times, const int32_t* input_sources,
                                                     int n_input_spikes,
                                                     const double* maxima_times, const double* maxima_values,
                                                     const double* maxima_errors, int n_maxima,
                                                     const double* w, int n_pre, int n_post,
                                                     double tau_mem, double tau_syn)
{
    (void)input_times; (void)input_sources; (void)n_input_spikes;
    (void)maxima_times; (void)maxima_values;
    (void)maxima_errors; (void)n_maxima;
    (void)w; (void)n_pre; (void)n_post;
    (void)tau_mem; (void)tau_syn;

    BackwardMaximaOutput result = {0, 0, NULL, NULL};
    /* TODO: implement */
    return result;
}

void eventprop_free_backward_maxima_output(BackwardMaximaOutput* output)
{
    if (!output) return;
    free(output->gradient);
    free(output->spike_errors);
}
