#include "eventprop_batch.h"
#include <stdlib.h>

/**
 * Batch backward pass (spikes).
 */
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
        result[b] = *eventprop_backward_spikes(&backward_input);
    }

    free(offsets);
    free(err_offsets);
    return result;
}

/**
 * Batch backward pass (spikes) with gradient accumulation.
 */
BackwardOutput* eventprop_backward_spikes_batch_accumulate(const BatchEventPropInput* input, SpikeOutput* fwd_batch, const double* post_errors)
{
    int batch_size = input->batch_size;
    const EventPropInput* base = input->input;

    /* Allocate accumulated gradient buffer + flat spike errors buffer */
    int n_pre = base->n_pre;
    int n_post = base->n_post;
    double* grad_acc = (double*)calloc(n_pre * n_post, sizeof(double));

    /* Compute total input spikes for error buffer */
    int total_input_spikes = 0;
    for (int i = 0; i < batch_size; i++) {
        total_input_spikes += input->n_spikes_per_sample[i];
    }
    double* spike_errors_flat = (double*)calloc(total_input_spikes, sizeof(double));

    /* Compute offsets */
    int* offsets = (int*)malloc(batch_size * sizeof(int));
    int* err_offsets = (int*)malloc(batch_size * sizeof(int));
    offsets[0] = 0;
    err_offsets[0] = 0;
    for (int i = 1; i < batch_size; i++) {
        offsets[i] = offsets[i-1] + input->n_spikes_per_sample[i-1];
        err_offsets[i] = err_offsets[i-1] + fwd_batch[i-1].n_spikes;
    }

    /* Process each sample in parallel, accumulate gradients */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) shared(base, offsets, err_offsets, batch_size, grad_acc, fwd_batch, post_errors) if(batch_size >= 2 || batch_size * n_pre >= 50)
#endif
    for (int b = 0; b < batch_size; b++) {
        int offset = offsets[b];
        int n_spikes = input->n_spikes_per_sample[b];

        EventPropInput fwbase = {
            .w=base->w, .n_pre=n_pre, .n_post=n_post,
            .times=base->times + offset, .sources=base->sources + offset, .n_spikes=n_spikes,
            .tau_mem=base->tau_mem, .tau_syn=base->tau_syn, .v_th=base->v_th
        };
        BackwardSpikesInput backward_input = {
            .input = &fwbase,
            .fwd_times=fwd_batch[b].times,
            .fwd_sources=fwd_batch[b].sources,
            .fwd_currents=fwd_batch[b].currents,
            .post_errors=post_errors + err_offsets[b],
            .n_post_spikes=fwd_batch[b].n_spikes,
        };
        BackwardOutput* sample_result = eventprop_backward_spikes(&backward_input);

        /* Accumulate gradient */
#ifdef _OPENMP
#pragma omp critical
#endif
        for (int i = 0; i < n_pre * n_post; i++) {
            grad_acc[i] += sample_result->grad[i];
        }

        /* Copy spike errors into flat buffer */
        int spike_err_offset = offsets[b];
        for (int s = 0; s < n_spikes; s++) {
            spike_errors_flat[spike_err_offset + s] = sample_result->spike_errors[s];
        }

        eventprop_free_backward_output(sample_result);
    }

    free(offsets);
    free(err_offsets);

    /* Build result */
    BackwardOutput* result = (BackwardOutput*)malloc(sizeof(BackwardOutput));
    result->n_pre = n_pre;
    result->n_post = n_post;
    result->grad = grad_acc;
    result->spike_errors = spike_errors_flat;
    return result;
}

/**
 * Batch forward pass (spikes).
 */
SpikeOutput* eventprop_compute_spikes_batch(const BatchEventPropInput* input)
{
    int batch_size = input->batch_size;
    const EventPropInput* base = input->input;
    const int* n_spikes_per_sample = input->n_spikes_per_sample;

    /* Allocate array of SpikeOutput structs */
    SpikeOutput* result = (SpikeOutput*)calloc(batch_size, sizeof(SpikeOutput));

    /* Compute sample offsets into flat concatenated arrays */
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
        result[b] = *eventprop_compute_spikes(&sample_input);
    }
    free(offsets);
    return result;
}

/**
 * Batch forward pass — compute membrane potential maxima.
 */
MaximaOutput* eventprop_compute_maxima_batch(const BatchEventPropInput* input)
{
    int batch_size = input->batch_size;
    const EventPropInput* base = input->input;
    const int* n_spikes_per_sample = input->n_spikes_per_sample;

    /* Allocate array of MaximaOutput structs */
    MaximaOutput* result = (MaximaOutput*)calloc(batch_size, sizeof(MaximaOutput));

    /* Compute sample offsets into flat concatenated arrays */
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

        EventPropInput sample_input = {
            .w=base->w, .n_pre=base->n_pre, .n_post=base->n_post,
            .times=base->times + offset, .sources=base->sources + offset, .n_spikes=n_spikes,
            .tau_mem=base->tau_mem, .tau_syn=base->tau_syn,
            .v_th = base->v_th
        };
        result[b] = *eventprop_compute_maxima(&sample_input);
    }
    free(offsets);
    return result;
}

/**
 * Batch backward pass (maxima).
 */
BackwardOutput* eventprop_backward_maxima_batch(const BatchEventPropInput* input, MaximaOutput* fwd_batch, const double* maxima_errors)
{
    int batch_size = input->batch_size;
    const EventPropInput* base = input->input;
    const int* n_spikes_per_sample = input->n_spikes_per_sample;

    /* Allocate array of BackwardOutput structs */
    BackwardOutput* result = (BackwardOutput*)calloc(batch_size, sizeof(BackwardOutput));

    /* Compute input sample offsets and error offsets (n_post per sample) */
    int* offsets = (int*)malloc(batch_size * sizeof(int));
    int n_post = base->n_post;
    offsets[0] = 0;
    for (int i = 1; i < batch_size; i++) {
        offsets[i] = offsets[i-1] + n_spikes_per_sample[i-1];
    }

    /* Process each sample in parallel */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) shared(base, offsets, n_spikes_per_sample, \
            batch_size, result, fwd_batch, maxima_errors, n_post) \
          if(batch_size >= 2 || batch_size * base->n_pre >= 50)
#endif
    for (int b = 0; b < batch_size; b++) {
        int offset = offsets[b];
        int n_spikes = n_spikes_per_sample[b];

        EventPropInput fwbase = {
            .w=base->w, .n_pre=base->n_pre, .n_post=base->n_post,
            .times=base->times + offset, .sources=base->sources + offset, .n_spikes=n_spikes,
            .tau_mem=base->tau_mem, .tau_syn=base->tau_syn, .v_th=base->v_th
        };

        /* Error offset: each sample has n_post errors */
        int err_offset = b * n_post;

        BackwardMaximaInput backward_input = {
            .input = &fwbase,
            .maxima_times=fwd_batch[b].times,
            .maxima_values=fwd_batch[b].values,
            .maxima_errors=maxima_errors + err_offset,
            .n_maxima=n_post,
        };
        result[b] = *eventprop_backward_maxima(&backward_input);
    }

    free(offsets);
    return result;
}

/**
 * Batch backward pass (maxima) with gradient accumulation.
 */
BackwardOutput* eventprop_backward_maxima_batch_accumulate(const BatchEventPropInput* input, MaximaOutput* fwd_batch, const double* maxima_errors)
{
    int batch_size = input->batch_size;
    const EventPropInput* base = input->input;

    /* Allocate accumulated gradient buffer + flat spike errors buffer */
    int n_pre = base->n_pre;
    int n_post = base->n_post;
    double* grad_acc = (double*)calloc(n_pre * n_post, sizeof(double));

    /* Compute total input spikes for error buffer */
    int total_input_spikes = 0;
    for (int i = 0; i < batch_size; i++) {
        total_input_spikes += input->n_spikes_per_sample[i];
    }
    double* spike_errors_flat = (double*)calloc(total_input_spikes, sizeof(double));

    /* Compute offsets */
    int* offsets = (int*)malloc(batch_size * sizeof(int));
    offsets[0] = 0;
    for (int i = 1; i < batch_size; i++) {
        offsets[i] = offsets[i-1] + input->n_spikes_per_sample[i-1];
    }

    /* Process each sample in parallel, accumulate gradients */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) shared(base, offsets, batch_size, grad_acc, fwd_batch, maxima_errors, n_post) if(batch_size >= 2 || batch_size * n_pre >= 50)
#endif
    for (int b = 0; b < batch_size; b++) {
        int offset = offsets[b];
        int n_spikes = input->n_spikes_per_sample[b];

        EventPropInput fwbase = {
            .w=base->w, .n_pre=n_pre, .n_post=n_post,
            .times=base->times + offset, .sources=base->sources + offset, .n_spikes=n_spikes,
            .tau_mem=base->tau_mem, .tau_syn=base->tau_syn, .v_th=base->v_th
        };

        int err_offset = b * n_post;

        BackwardMaximaInput backward_input = {
            .input = &fwbase,
            .maxima_times=fwd_batch[b].times,
            .maxima_values=fwd_batch[b].values,
            .maxima_errors=maxima_errors + err_offset,
            .n_maxima=n_post,
        };
        BackwardOutput* sample_result = eventprop_backward_maxima(&backward_input);

#ifdef _OPENMP
#pragma omp critical
#endif
        for (int i = 0; i < n_pre * n_post; i++) {
            grad_acc[i] += sample_result->grad[i];
        }

        /* Copy spike errors into flat buffer */
        int spike_err_offset = offsets[b];
        for (int s = 0; s < n_spikes; s++) {
            spike_errors_flat[spike_err_offset + s] = sample_result->spike_errors[s];
        }

        eventprop_free_backward_output(sample_result);
    }

    free(offsets);

    /* Build result */
    BackwardOutput* result = (BackwardOutput*)malloc(sizeof(BackwardOutput));
    result->n_pre = n_pre;
    result->n_post = n_post;
    result->grad = grad_acc;
    result->spike_errors = spike_errors_flat;
    return result;
}

/**
 * ── Pointer-based batch functions (no flattening) ────────────────
 *
 * These mirror the flat-array versions but accept per-sample pointers,
 * matching how pybind11 iterates std::vector<Spikes>.
 */

/**
 * Pointer-based batch forward pass (spikes).
 */
SpikeOutput* eventprop_compute_spikes_batch_ptrs(const BatchEventPropInputPtrs* input)
{
    int batch_size = input->batch_size;

    /* Allocate result array */
    SpikeOutput* result = (SpikeOutput*)calloc(batch_size, sizeof(SpikeOutput));

    /* Process each sample in parallel */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(batch_size >= 2)
#endif
    for (int b = 0; b < batch_size; b++) {
        EventPropInput sample_input = {
            .w=input->w,
            .n_pre=input->n_pre,
            .n_post=input->n_post,
            .times=input->times[b],
            .sources=input->sources[b],
            .n_spikes=input->n_spikes[b],
            .tau_mem=input->tau_mem,
            .tau_syn=input->tau_syn,
            .v_th=input->v_th
        };
        result[b] = *eventprop_compute_spikes(&sample_input);
    }
    return result;
}

/**
 * Pointer-based batch forward pass (maxima).
 */
MaximaOutput* eventprop_compute_maxima_batch_ptrs(const BatchEventPropInputPtrs* input)
{
    int batch_size = input->batch_size;

    /* Allocate result array */
    MaximaOutput* result = (MaximaOutput*)calloc(batch_size, sizeof(MaximaOutput));

    /* Process each sample in parallel */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(batch_size >= 2)
#endif
    for (int b = 0; b < batch_size; b++) {
        EventPropInput sample_input = {
            .w=input->w,
            .n_pre=input->n_pre,
            .n_post=input->n_post,
            .times=input->times[b],
            .sources=input->sources[b],
            .n_spikes=input->n_spikes[b],
            .tau_mem=input->tau_mem,
            .tau_syn=input->tau_syn,
            .v_th=input->v_th
        };
        result[b] = *eventprop_compute_maxima(&sample_input);
    }
    return result;
}

/**
 * Pointer-based batch backward pass (spikes) with gradient accumulation.
 */
BackwardOutput* eventprop_backward_spikes_batch_accumulate_ptrs(
    const BatchEventPropInputPtrs* input,
    SpikeOutput* fwd_batch,
    const double** post_errors,
    const int* n_post_errors
)
{
    int batch_size = input->batch_size;
    int n_pre = input->n_pre;
    int n_post = input->n_post;

    /* Allocate accumulated gradient buffer */
    double* grad_acc = (double*)calloc(n_pre * n_post, sizeof(double));

    /* Compute total input spikes for error buffer */
    int total_input_spikes = 0;
    for (int i = 0; i < batch_size; i++) {
        total_input_spikes += input->n_spikes[i];
    }
    double* spike_errors_flat = (double*)calloc(total_input_spikes, sizeof(double));

    /* Compute spike error offsets */
    int* err_offsets = (int*)malloc(batch_size * sizeof(int));
    err_offsets[0] = 0;
    for (int i = 1; i < batch_size; i++) {
        err_offsets[i] = err_offsets[i-1] + input->n_spikes[i-1];
    }

    /* Process each sample in parallel, accumulate gradients */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) shared(grad_acc, spike_errors_flat, err_offsets) if(batch_size >= 2)
#endif
    for (int b = 0; b < batch_size; b++) {
        EventPropInput sample_input = {
            .w=input->w,
            .n_pre=n_pre,
            .n_post=n_post,
            .times=input->times[b],
            .sources=input->sources[b],
            .n_spikes=input->n_spikes[b],
            .tau_mem=input->tau_mem,
            .tau_syn=input->tau_syn,
            .v_th=input->v_th
        };
        BackwardSpikesInput backward_input = {
            .input = &sample_input,
            .fwd_times=fwd_batch[b].times,
            .fwd_sources=fwd_batch[b].sources,
            .fwd_currents=fwd_batch[b].currents,
            .post_errors=post_errors[b],
            .n_post_spikes=n_post_errors[b],
        };
        BackwardOutput* sample_result = eventprop_backward_spikes(&backward_input);

#ifdef _OPENMP
#pragma omp critical
#endif
        for (int i = 0; i < n_pre * n_post; i++) {
            grad_acc[i] += sample_result->grad[i];
        }

        /* Copy spike errors into flat buffer */
        int offset = err_offsets[b];
        int n_spikes = input->n_spikes[b];
        for (int s = 0; s < n_spikes; s++) {
            spike_errors_flat[offset + s] = sample_result->spike_errors[s];
        }

        eventprop_free_backward_output(sample_result);
    }

    free(err_offsets);

    /* Build result */
    BackwardOutput* result = (BackwardOutput*)malloc(sizeof(BackwardOutput));
    result->n_pre = n_pre;
    result->n_post = n_post;
    result->grad = grad_acc;
    result->spike_errors = spike_errors_flat;
    return result;
}

/**
 * Pointer-based batch backward pass (maxima) with gradient accumulation.
 */
BackwardOutput* eventprop_backward_maxima_batch_accumulate_ptrs(
    const BatchEventPropInputPtrs* input,
    MaximaOutput* fwd_batch,
    const double** maxima_errors,
    int n_post
)
{
    int batch_size = input->batch_size;
    int n_pre = input->n_pre;

    /* Allocate accumulated gradient buffer */
    double* grad_acc = (double*)calloc(n_pre * n_post, sizeof(double));

    /* Compute total input spikes for error buffer */
    int total_input_spikes = 0;
    for (int i = 0; i < batch_size; i++) {
        total_input_spikes += input->n_spikes[i];
    }
    double* spike_errors_flat = (double*)calloc(total_input_spikes, sizeof(double));

    /* Compute spike error offsets */
    int* err_offsets = (int*)malloc(batch_size * sizeof(int));
    err_offsets[0] = 0;
    for (int i = 1; i < batch_size; i++) {
        err_offsets[i] = err_offsets[i-1] + input->n_spikes[i-1];
    }

    /* Process each sample in parallel, accumulate gradients */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) shared(grad_acc, spike_errors_flat, err_offsets) if(batch_size >= 2)
#endif
    for (int b = 0; b < batch_size; b++) {
        EventPropInput sample_input = {
            .w=input->w,
            .n_pre=n_pre,
            .n_post=n_post,
            .times=input->times[b],
            .sources=input->sources[b],
            .n_spikes=input->n_spikes[b],
            .tau_mem=input->tau_mem,
            .tau_syn=input->tau_syn,
            .v_th=input->v_th
        };
        BackwardMaximaInput backward_input = {
            .input = &sample_input,
            .maxima_times=fwd_batch[b].times,
            .maxima_values=fwd_batch[b].values,
            .maxima_errors=maxima_errors[b],
            .n_maxima=n_post,
        };
        BackwardOutput* sample_result = eventprop_backward_maxima(&backward_input);

#ifdef _OPENMP
#pragma omp critical
#endif
        for (int i = 0; i < n_pre * n_post; i++) {
            grad_acc[i] += sample_result->grad[i];
        }

        /* Copy spike errors into flat buffer */
        int offset = err_offsets[b];
        int n_spikes = input->n_spikes[b];
        for (int s = 0; s < n_spikes; s++) {
            spike_errors_flat[offset + s] = sample_result->spike_errors[s];
        }

        eventprop_free_backward_output(sample_result);
    }

    free(err_offsets);

    /* Build result */
    BackwardOutput* result = (BackwardOutput*)malloc(sizeof(BackwardOutput));
    result->n_pre = n_pre;
    result->n_post = n_post;
    result->grad = grad_acc;
    result->spike_errors = spike_errors_flat;
    return result;
}

/**
 * Free dropout output array.
 */
void eventprop_free_dropout_output(DropoutOutput* output, int batch_size)
{
    if (output == NULL) return;
    for (int i = 0; i < batch_size; i++) {
        free(output[i].times);
        free(output[i].sources);
    }
    free(output);
}

/**
 * Simple LCG PRNG (glibc parameters) for reproducible dropout.
 * Uses uint64_t to ensure consistent wrapping across platforms.
 */
#include <stdint.h>

static inline double lcg_rand(uint64_t* state)
{
    *state = (*state * 6364136223846793005ULL + 1442695040888963407ULL);
    return ((double)(*state >> 11)) / 9007199254740992.0;  /* [0, 1), ~53 bits of precision */
}

/**
 * Batch dropout — randomly drop spikes from each sample.
 */
DropoutOutput* eventprop_dropout_spikes_batch(
    const BatchEventPropInput* input,
    double p_drop,
    int seed
)
{
    int batch_size = input->batch_size;
    const EventPropInput* base = input->input;
    const int* n_spikes_per_sample = input->n_spikes_per_sample;

    /* Allocate result array */
    DropoutOutput* result = (DropoutOutput*)calloc(batch_size, sizeof(DropoutOutput));

    /* Compute sample offsets */
    int* offsets = (int*)malloc(batch_size * sizeof(int));
    offsets[0] = 0;
    for (int i = 1; i < batch_size; i++) {
        offsets[i] = offsets[i-1] + n_spikes_per_sample[i-1];
    }

    /* Process each sample — sequential (malloc per-sample) */
    for (int b = 0; b < batch_size; b++) {
        int offset = offsets[b];
        int n_spikes = n_spikes_per_sample[b];

        /* Per-sample seed for reproducibility */
        uint64_t state = (uint64_t)(seed + b) + 1;

        if (n_spikes == 0 || p_drop <= 0.0) {
            /* No dropout — copy all spikes */
            result[b].n_spikes = n_spikes;
            if (n_spikes > 0) {
                result[b].times = (double*)malloc(n_spikes * sizeof(double));
                result[b].sources = (int*)malloc(n_spikes * sizeof(int));
                for (int i = 0; i < n_spikes; i++) {
                    result[b].times[i] = base->times[offset + i];
                    result[b].sources[i] = base->sources[offset + i];
                }
            }
            continue;
        }

        /* Single-pass: pre-allocate max size, then compact */
        result[b].times = (double*)malloc(n_spikes * sizeof(double));
        result[b].sources = (int*)malloc(n_spikes * sizeof(int));

        int idx = 0;
        for (int i = 0; i < n_spikes; i++) {
            if (lcg_rand(&state) > p_drop) {
                result[b].times[idx] = base->times[offset + i];
                result[b].sources[idx] = base->sources[offset + i];
                idx++;
            }
        }
        result[b].n_spikes = idx;
    }

    free(offsets);
    return result;
}
