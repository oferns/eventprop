/**
 * @file eventprop_scalar.c
 * @brief Scalar C implementation of EventProp algorithm (no SIMD, no dependencies).
 */

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "eventprop.h"

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
        exp_mem[i] = exp(-times[i] / tau_mem);
        exp_syn[i] = exp(-times[i] / tau_syn);
    }

    for (int i = 0; i < n_spikes; i++) {
        int src = sources[i];
        double exp_syn_i = exp_syn[i], exp_mem_i = exp_mem[i];
        int row_out = i + 1, row_in = i;
        int src_offset = src * n_post;

        for (int j = 0; j < n_post; j++) {
            sum0[row_out * n_post + j] = sum0[row_in * n_post + j] + w[src_offset + j] * exp_syn_i;
            sum1[row_out * n_post + j] = sum1[row_in * n_post + j] + w[src_offset + j] * exp_mem_i;
        }
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

/* ── Toms748 (Brent) root-finding ─────────────────────────────────── */

#define TOMS748_MAX_ITER 100
#define TOMS748_TOL 1e-12

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

static double toms748_solve(void* ctx, root_fn_t fn, double a, double b) {
    double fa = fn(ctx, a), fb = fn(ctx, b);
    if (fa * fb > 0) return a;

    double c = a, fc = fa, d = 0, e = 0;
    for (int iter = 0; iter < TOMS748_MAX_ITER; iter++) {
        if (fb * fc > 0) { c = a; fc = fa; d = e = b - a; }
        if (fabs(fc) < fabs(fb)) { double t=t;a=b;b=c;c=t; t=fa;fa=fb;fb=fc;fc=t; }

        double tol = 2.0 * TOMS748_TOL * fabs(b) + TOMS748_TOL / 2.0;
        double m = (c - b) / 2.0;
        if (fabs(m) <= tol || fb == 0.0) return b;

        if (fabs(e) >= tol && fabs(fa) > fabs(fb)) {
            double s = fb / fa, p, q;
            if (a == c) { p = 2.0 * m * s; q = 1.0 - s; }
            else { double r = fb / fc; p = s*(2.0*m*q*(q-r)-(b-a)*(r-1.0)); q=(q-1.0)*(r-1.0)*(s-1.0); }
            if (p > 0) q = -q; p = fabs(p);
            if (2.0 * p < fmin(3.0*m*q - fabs(tol*q), fabs(e*q))) { d = p/q; e = d; }
            else { d = m; e = m; }
        } else { d = m; e = m; }

        a = b; fa = fb;
        if (fabs(d) > tol) b += d; else b += (m > 0 ? tol : -tol);
        fb = fn(ctx, b);
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
                    /* Last interval */
                    double s0 = sum0[target_nrn * (n_spikes+1) + n_spikes];
                    double s1 = sum1[target_nrn * (n_spikes+1) + n_spikes];
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

    typedef struct { double time; int source; double current; } SpikeEntry;
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

    for (int i = 0; i < total - 1; i++)
        for (int j = 0; j < total - i - 1; j++)
            if (entries[j].time > entries[j+1].time) { SpikeEntry t=entries[j]; entries[j]=entries[j+1]; entries[j+1]=t; }

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
    typedef struct { double time; int idx; int is_input; } MergedSpike;
    MergedSpike* merged = malloc(total * sizeof(MergedSpike));

    for (int i = 0; i < n_spikes; i++) { merged[i].time=times[i]; merged[i].idx=i; merged[i].is_input=1; }
    for (int i = 0; i < n_post_spikes; i++) { merged[n_spikes+i].time=fwd_times[i]; merged[n_spikes+i].idx=i; merged[n_spikes+i].is_input=0; }

    for (int i = 0; i < total-1; i++)
        for (int j = 0; j < total-i-1; j++)
            if (merged[j].time < merged[j+1].time) { MergedSpike t=merged[j]; merged[j]=merged[j+1]; merged[j+1]=t; }

    double* all_errors = calloc(total, sizeof(double));
    for (int i = 0; i < n_post_spikes; i++) all_errors[n_spikes + i] = post_errors[i];

    double* lambda_v = calloc(n_post, sizeof(double));
    int max_jumps = n_post_spikes;
    double* jump_time = malloc(max_jumps * sizeof(double));
    double* jump_mag = malloc(max_jumps * sizeof(double));
    int max_nrn = n_pre > n_post ? n_pre : n_post;
    int* jump_end = calloc(max_nrn, sizeof(int));

    double largest_time = merged[0].time, previous_t = largest_time;
    double k_bwd = tau_mem / (tau_mem - tau_syn);

    for (int s = 0; s < total; s++) {
        int spike_idx = merged[s].idx, is_input = merged[s].is_input;
        double spike_time = merged[s].time;
        int spike_source = is_input ? sources[spike_idx] : fwd_sources[spike_idx];

        double t_bwd = largest_time - spike_time;
        double dt = previous_t - t_bwd;
        for (int nrn = 0; nrn < n_post; nrn++) lambda_v[nrn] *= exp(-dt / tau_mem);

        if (is_input) {
            double lambda_i = 0.0;
            for (int j = 0, end = jump_end[spike_source]; j < end; j++)
                lambda_i += jump_mag[j] * k_bwd * (exp(-(t_bwd-jump_time[j])/tau_mem) - exp(-(t_bwd-jump_time[j])/tau_syn));

            for (int nrn = 0; nrn < n_post; nrn++)
                grad[spike_source * n_post + nrn] += -tau_syn * lambda_i;

            double outbound = 0.0;
            for (int nrn = 0; nrn < n_post; nrn++)
                outbound += w[spike_source * n_post + nrn] * (lambda_v[nrn] - lambda_i);
            all_errors[spike_idx] += outbound;
        } else {
            double current = fwd_currents[spike_idx];
            double inv_ic = 1.0 / (current - v_th);
            int j_idx = jump_end[spike_source];
            double post_err = all_errors[n_spikes + spike_idx];
            jump_mag[j_idx] = inv_ic * (v_th * lambda_v[spike_source] + post_err);
            jump_time[j_idx] = t_bwd;
            jump_end[spike_source]++;

            lambda_v[spike_source] = (current / (current - v_th)) * lambda_v[spike_source] + inv_ic * post_err;
        }
        previous_t = t_bwd;
    }

    free(merged); free(all_errors); free(lambda_v); free(jump_time); free(jump_mag); free(jump_end);
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
