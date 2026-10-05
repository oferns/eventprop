/**
 * @file batch_functions.cpp
 * @brief Tests for EventProp batch functions.
 *
 * Focus: has_errors flag propagation, per-sample correctness, and edge cases.
 */

#include <gtest/gtest.h>
#include "eventprop_batch.h"
#include <cmath>
#include <vector>
#include <cstring>

static constexpr double kTauMem = 0.02;
static constexpr double kTauSyn = 0.005;
static constexpr double kVTh    = 1.0;

// ──────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────

inline EventPropInput make_input(const std::vector<double>& w, int n_pre, int n_post,
                                  const std::vector<double>& times,
                                  const std::vector<int>& sources) {
    return EventPropInput{
        w.data(), n_pre, n_post,
        times.data(), sources.data(), (int)times.size(),
        kTauMem, kTauSyn, kVTh
    };
}

// Build a batch of identical valid inputs.
typedef struct {
    EventPropInput* inputs;   // malloc'd, length = size
    double* w;                // shared weight matrix (malloc'd)
    double* times;            // per-sample spike times (1 per sample)
    int* sources;             // per-sample source indices (1 per sample)
} BatchData;

static BatchData make_valid_batch(int size, int n_pre = 2, int n_post = 3) {
    EventPropInput* inputs = (EventPropInput*)malloc(size * sizeof(EventPropInput));
    double* w = (double*)calloc(n_pre * n_post, sizeof(double));
    for (int i = 0; i < n_pre * n_post; ++i) w[i] = 1.0;

    double* times = (double*)malloc(size * sizeof(double));
    int* sources = (int*)calloc(size, sizeof(int));
    for (int i = 0; i < size; ++i) times[i] = 0.01;

    for (int i = 0; i < size; ++i) {
        inputs[i] = EventPropInput{
            w, n_pre, n_post,
            &times[i], &sources[i], 1,
            kTauMem, kTauSyn, kVTh
        };
    }

    return {inputs, w, times, sources};
}

static void free_batch_data(BatchData data) {
    free(data.inputs);
    free(data.w);
    free(data.times);
    free(data.sources);
}

// ──────────────────────────────────────────────────────────────
// compute_spikes_batch
// ──────────────────────────────────────────────────────────────

TEST(BatchSpikes, ValidBatchHasNoErrors) {
    BatchData data = make_valid_batch(4);
    SpikesBatch batch = eventprop_compute_spikes_batch(data.inputs, 4);

    EXPECT_EQ(batch.has_errors, 0);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(batch.spikes[i].error_no, EVENTPROP_OK);
    }

    eventprop_free_spikes_batch(batch);
    free_batch_data(data);
}

TEST(BatchSpikes, ZeroBatchSizeReturnsError) {
    SpikesBatch batch = eventprop_compute_spikes_batch(nullptr, 0);
    EXPECT_EQ(batch.batch_size, 0);
    EXPECT_EQ(batch.has_errors, 1); // zero batch size is now an error
    eventprop_free_spikes_batch(batch);
}

TEST(BatchSpikes, OneBadSampleSetsHasErrors) {
    // Build a batch where sample 1 has invalid n_pre (negative)
    std::vector<double> w0(6.0), w2(6.0); // valid weight arrays
    std::vector<double> t0{0.01}, t2{0.01};
    std::vector<int> s0{0}, s2{0};

    EventPropInput inputs[3];
    inputs[0] = make_input(w0, 2, 3, t0, s0);
    inputs[1] = make_input(w0, -1, 3, t0, s0); // bad n_pre
    inputs[2] = make_input(w2, 2, 3, t2, s2);

    SpikesBatch batch = eventprop_compute_spikes_batch(inputs, 3);

    EXPECT_EQ(batch.has_errors, 1);
    EXPECT_EQ(batch.spikes[0].error_no, EVENTPROP_OK);
    EXPECT_NE(batch.spikes[1].error_no, EVENTPROP_OK);
    EXPECT_EQ(batch.spikes[2].error_no, EVENTPROP_OK);

    eventprop_free_spikes_batch(batch);
}

TEST(BatchSpikes, AllBadSamplesSetsHasErrors) {
    std::vector<double> w0(6.0), w1(6.0);
    std::vector<double> t0{0.01}, t1{0.01};
    std::vector<int> s0{0}, s1{0};

    EventPropInput inputs[2];
    inputs[0] = make_input(w0, -1, 3, t0, s0); // bad n_pre
    inputs[1] = make_input(w1, 2, -1, t1, s1); // bad n_post

    SpikesBatch batch = eventprop_compute_spikes_batch(inputs, 2);

    EXPECT_EQ(batch.has_errors, 1);
    for (int i = 0; i < 2; ++i) {
        EXPECT_NE(batch.spikes[i].error_no, EVENTPROP_OK);
    }

    eventprop_free_spikes_batch(batch);
}

// ──────────────────────────────────────────────────────────────
// compute_spikes_batch (struct-based API)
// ──────────────────────────────────────────────────────────────

TEST(BatchSpikesStruct, ValidBatchHasNoErrors) {
    std::vector<double> w = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // 2x3
    std::vector<double> times0 = {0.01, 0.02}, times1 = {0.01, 0.02};
    std::vector<int> sources0 = {0, 1}, sources1 = {0, 1};

    EventPropInput inputs[2];
    inputs[0] = make_input(w, 2, 3, times0, sources0);
    inputs[1] = make_input(w, 2, 3, times1, sources1);

    SpikesBatch batch = eventprop_compute_spikes_batch(inputs, 2);

    EXPECT_EQ(batch.has_errors, 0);
    for (int i = 0; i < 2; ++i) {
        EXPECT_EQ(batch.spikes[i].error_no, EVENTPROP_OK);
    }

    eventprop_free_spikes_batch(batch);
}

TEST(BatchSpikesStruct, ZeroBatchSizeReturnsError) {
    SpikesBatch batch = eventprop_compute_spikes_batch(nullptr, 0);
    EXPECT_EQ(batch.batch_size, 0);
    EXPECT_EQ(batch.has_errors, 1); // zero batch size is now an error
    eventprop_free_spikes_batch(batch);
}

TEST(BatchSpikesStruct, OneBadSampleSetsHasErrors) {
    std::vector<double> w = {1.0, 2.0, 3.0}; // 1x3
    std::vector<double> times0 = {0.01}, times1 = {0.02};
    std::vector<int> sources0 = {0}, sources1 = {5};       // second source out of bounds (>= n_pre=1)

    EventPropInput inputs[2];
    inputs[0] = make_input(w, 1, 3, times0, sources0);
    inputs[1] = make_input(w, 1, 3, times1, sources1);

    SpikesBatch batch = eventprop_compute_spikes_batch(inputs, 2);

    EXPECT_EQ(batch.has_errors, 1);
    EXPECT_EQ(batch.spikes[0].error_no, EVENTPROP_OK);
    EXPECT_NE(batch.spikes[1].error_no, EVENTPROP_OK);

    eventprop_free_spikes_batch(batch);
}

// ──────────────────────────────────────────────────────────────
// backward_spikes_batch
// ──────────────────────────────────────────────────────────────

TEST(BatchBackwardSpikes, ValidBatchHasNoErrors) {
    BatchData data = make_valid_batch(3);
    SpikesBatch spikes = eventprop_compute_spikes_batch(data.inputs, 3);
    EXPECT_EQ(spikes.has_errors, 0);

    BackwardsBatch batch = eventprop_backward_spikes_batch(data.inputs, spikes);

    EXPECT_EQ(batch.has_errors, 0);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(batch.outputs[i].error_no, EVENTPROP_OK);
    }

    eventprop_free_spikes_batch(spikes);
    eventprop_free_backwards_batch(batch);
    free_batch_data(data);
}

TEST(BatchBackwardSpikes, OneBadSampleSetsHasErrors) {
    std::vector<double> w0(6.0), w1(6.0);
    std::vector<double> t0{0.01}, t1{0.01};
    std::vector<int> s0{0}, s1{0};

    EventPropInput inputs[2];
    inputs[0] = make_input(w0, 2, 3, t0, s0);
    inputs[1] = make_input(w1, -1, 3, t1, s1); // bad n_pre

    SpikesBatch spikes = eventprop_compute_spikes_batch(inputs, 2);
    EXPECT_EQ(spikes.has_errors, 1);

    BackwardsBatch batch = eventprop_backward_spikes_batch(inputs, spikes);

    EXPECT_EQ(batch.has_errors, 1);
    EXPECT_EQ(batch.outputs[0].error_no, EVENTPROP_OK);
    EXPECT_NE(batch.outputs[1].error_no, EVENTPROP_OK);

    eventprop_free_spikes_batch(spikes);
    eventprop_free_backwards_batch(batch);
}

// ──────────────────────────────────────────────────────────────
// compute_maxima_batch
// ──────────────────────────────────────────────────────────────

TEST(BatchMaxima, ValidBatchHasNoErrors) {
    BatchData data = make_valid_batch(3);
    SpikesBatch spikes = eventprop_compute_spikes_batch(data.inputs, 3);

    MaximaBatch batch = eventprop_compute_maxima_batch(data.inputs, spikes);

    EXPECT_EQ(batch.has_errors, 0);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(batch.outputs[i].error_no, EVENTPROP_OK);
    }

    eventprop_free_spikes_batch(spikes);
    eventprop_free_maxima_batch(batch);
    free_batch_data(data);
}

TEST(BatchMaxima, OneBadSampleSetsHasErrors) {
    std::vector<double> w0(6.0), w1(6.0);
    std::vector<double> t0{0.01}, t1{0.01};
    std::vector<int> s0{0}, s1{0};

    EventPropInput inputs[2];
    inputs[0] = make_input(w0, 2, 3, t0, s0);
    inputs[1] = make_input(w1, 2, -1, t1, s1); // bad n_post

    SpikesBatch spikes = eventprop_compute_spikes_batch(inputs, 2);

    MaximaBatch batch = eventprop_compute_maxima_batch(inputs, spikes);

    EXPECT_EQ(batch.has_errors, 1);
    EXPECT_EQ(batch.outputs[0].error_no, EVENTPROP_OK);
    EXPECT_NE(batch.outputs[1].error_no, EVENTPROP_OK);

    eventprop_free_spikes_batch(spikes);
    eventprop_free_maxima_batch(batch);
}

// ──────────────────────────────────────────────────────────────
// compute_maxima_batch (struct-based API)
// ──────────────────────────────────────────────────────────────

TEST(BatchMaximaStruct, ValidBatchHasNoErrors) {
    std::vector<double> w = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // 2x3
    std::vector<double> times0 = {0.01, 0.02}, times1 = {0.01, 0.02};
    std::vector<int> sources0 = {0, 1}, sources1 = {0, 1};

    EventPropInput inputs[2];
    inputs[0] = make_input(w, 2, 3, times0, sources0);
    inputs[1] = make_input(w, 2, 3, times1, sources1);

    SpikesBatch spikes = eventprop_compute_spikes_batch(inputs, 2);
    MaximaBatch batch = eventprop_compute_maxima_batch(inputs, spikes);

    EXPECT_EQ(batch.has_errors, 0);
    for (int i = 0; i < 2; ++i) {
        EXPECT_EQ(batch.outputs[i].error_no, EVENTPROP_OK);
    }

    eventprop_free_spikes_batch(spikes);
    eventprop_free_maxima_batch(batch);
}

TEST(BatchMaximaStruct, OneBadSampleSetsHasErrors) {
    std::vector<double> w = {1.0, 2.0, 3.0}; // 1x3
    std::vector<double> times0 = {0.01}, times1 = {0.02};
    std::vector<int> sources0 = {0}, sources1 = {5};       // second source out of bounds (>= n_pre=1)

    EventPropInput inputs[2];
    inputs[0] = make_input(w, 1, 3, times0, sources0);
    inputs[1] = make_input(w, 1, 3, times1, sources1);

    SpikesBatch spikes = eventprop_compute_spikes_batch(inputs, 2);
    MaximaBatch batch = eventprop_compute_maxima_batch(inputs, spikes);

    EXPECT_EQ(batch.has_errors, 1);
    EXPECT_EQ(batch.outputs[0].error_no, EVENTPROP_OK);
    EXPECT_NE(batch.outputs[1].error_no, EVENTPROP_OK);

    eventprop_free_spikes_batch(spikes);
    eventprop_free_maxima_batch(batch);
}

// ──────────────────────────────────────────────────────────────
// backward_maxima_batch
// ──────────────────────────────────────────────────────────────

TEST(BatchBackwardMaxima, ValidBatchHasNoErrors) {
    std::vector<double> w = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // 2x3
    std::vector<double> times0 = {0.01, 0.02}, times1 = {0.01, 0.02};
    std::vector<int> sources0 = {0, 1}, sources1 = {0, 1};

    EventPropInput inputs[2];
    inputs[0] = make_input(w, 2, 3, times0, sources0);
    inputs[1] = make_input(w, 2, 3, times1, sources1);

    SpikesBatch spikes = eventprop_compute_spikes_batch(inputs, 2);
    MaximaBatch maxima = eventprop_compute_maxima_batch(inputs, spikes);

    // Set errors before backward pass
    for (int i = 0; i < 2; ++i) {
        maxima.outputs[i].errors[0] = 1.0;
    }

    BackwardsBatch batch = eventprop_backward_maxima_batch(inputs, maxima);

    EXPECT_EQ(batch.has_errors, 0);
    for (int i = 0; i < 2; ++i) {
        EXPECT_EQ(batch.outputs[i].error_no, EVENTPROP_OK);
    }

    eventprop_free_spikes_batch(spikes);
    eventprop_free_maxima_batch(maxima);
    eventprop_free_backwards_batch(batch);
}

TEST(BatchBackwardMaxima, OneBadSampleSetsHasErrors) {
    std::vector<double> w = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // 2x3
    std::vector<double> times0 = {0.01, 0.02}, times1 = {0.01, 0.02};
    std::vector<int> sources0 = {0, 1}, sources1 = {0, 1};

    EventPropInput inputs[2];
    inputs[0] = make_input(w, 2, 3, times0, sources0);
    inputs[1] = make_input(w, 2, 3, times1, sources1);

    SpikesBatch spikes = eventprop_compute_spikes_batch(inputs, 2);
    MaximaBatch maxima = eventprop_compute_maxima_batch(inputs, spikes);

    // Set errors before backward pass
    for (int i = 0; i < 2; ++i) {
        maxima.outputs[i].errors[0] = 1.0;
    }

    // Corrupt one input after computing maxima but before backward
    inputs[1].n_pre = -1;

    BackwardsBatch batch = eventprop_backward_maxima_batch(inputs, maxima);

    EXPECT_EQ(batch.has_errors, 1);
    EXPECT_EQ(batch.outputs[0].error_no, EVENTPROP_OK);       // valid sample still works
    EXPECT_NE(batch.outputs[1].error_no, EVENTPROP_OK);      // bad sample fails

    eventprop_free_spikes_batch(spikes);
    eventprop_free_maxima_batch(maxima);
    eventprop_free_backwards_batch(batch);
}

// ──────────────────────────────────────────────────────────────
// accumulate_gradients
// ──────────────────────────────────────────────────────────────

TEST(BatchAccumulate, SumsGradientsCorrectly) {
    // Build a BackwardsBatch manually with known gradients
    int n_pre = 2, n_post = 3;
    int batch_size = 3;

    BackwardOutput* outputs = (BackwardOutput*)calloc(batch_size, sizeof(BackwardOutput));
    for (int b = 0; b < batch_size; ++b) {
        outputs[b].n_pre = n_pre;
        outputs[b].n_post = n_post;
        outputs[b].grad = (double*)malloc(n_pre * n_post * sizeof(double));
        // Each sample has grad[i] = b+1 (so sum should be batch_size*(b+1))
        for (int i = 0; i < n_pre * n_post; ++i) {
            outputs[b].grad[i] = (double)(b + 1);
        }
    }

    BackwardsBatch batch = { .error_no = EVENTPROP_OK, .batch_size = batch_size, .has_errors = 0, .outputs = outputs };

    GradientOutput grad = eventprop_accumulate_gradients(batch, n_pre, n_post);

    // Sum of {1, 2, 3} = 6 for each element
    EXPECT_EQ(grad.error_no, EVENTPROP_OK);
    EXPECT_EQ(grad.rows, n_pre);
    EXPECT_EQ(grad.cols, n_post);
    for (int i = 0; i < n_pre * n_post; ++i) {
        EXPECT_DOUBLE_EQ(grad.data[i], 6.0);
    }

    eventprop_free_gradient_output(grad);
    // Clean up manually since accumulate doesn't free the batch
    for (int b = 0; b < batch_size; ++b) {
        free(outputs[b].grad);
    }
    free(outputs);
}

TEST(BatchAccumulate, FailedSampleYieldsPartialSumFlag) {
    int n_pre = 2, n_post = 3;
    int batch_size = 2;

    BackwardOutput* outputs = (BackwardOutput*)calloc(batch_size, sizeof(BackwardOutput));
    for (int b = 0; b < batch_size; ++b) {
        outputs[b].n_pre = n_pre;
        outputs[b].n_post = n_post;
        outputs[b].grad = (double*)malloc(n_pre * n_post * sizeof(double));
        for (int i = 0; i < n_pre * n_post; ++i)
            outputs[b].grad[i] = 1.0;
    }
    // Sample 1 fails — its contribution must be excluded and flagged
    outputs[1].error_no = EVENTPROP_INVALID_DIMENSION;

    BackwardsBatch batch = { .error_no = EVENTPROP_OK, .batch_size = batch_size, .has_errors = 1, .outputs = outputs };

    GradientOutput grad = eventprop_accumulate_gradients(batch, n_pre, n_post);

    EXPECT_EQ(grad.error_no, EVENTPROP_BATCH_PARTIAL_ACCUMULATION);
    for (int i = 0; i < n_pre * n_post; ++i)
        EXPECT_DOUBLE_EQ(grad.data[i], 1.0);  // only sample 0 counted

    eventprop_free_gradient_output(grad);
    for (int b = 0; b < batch_size; ++b) free(outputs[b].grad);
    free(outputs);
}

// ──────────────────────────────────────────────────────────────
// backward_spikes_batch_accumulate
// ──────────────────────────────────────────────────────────────

TEST(BatchAccumulateBackwardSpikes, ValidBatchHasNoErrors) {
    BatchData data = make_valid_batch(3);
    SpikesBatch spikes = eventprop_compute_spikes_batch(data.inputs, 3);

    AccumulatedBackwardResult result = eventprop_backward_spikes_batch_accumulate(data.inputs, spikes);

    EXPECT_EQ(result.has_errors, 0);
    EXPECT_EQ(result.summed_grads->error_no, EVENTPROP_OK);
    EXPECT_NE(result.summed_grads->data, nullptr);
    EXPECT_EQ(result.batch_size, 3);

    // All per-sample outputs should be OK (grad freed after accumulation)
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(result.outputs[i].error_no, EVENTPROP_OK);
    }

    eventprop_free_spikes_batch(spikes);
    eventprop_free_accumulated_backward_result(result);
    free_batch_data(data);
}

TEST(BatchAccumulateBackwardSpikes, OneBadSampleSetsHasErrors) {
    std::vector<double> w0(6.0), w1(6.0);
    std::vector<double> t0{0.01}, t1{0.01};
    std::vector<int> s0{0}, s1{0};

    EventPropInput inputs[2];
    inputs[0] = make_input(w0, 2, 3, t0, s0);
    inputs[1] = make_input(w1, -1, 3, t1, s1); // bad n_pre

    SpikesBatch spikes = eventprop_compute_spikes_batch(inputs, 2);

    AccumulatedBackwardResult result = eventprop_backward_spikes_batch_accumulate(inputs, spikes);

    EXPECT_EQ(result.has_errors, 1);
    EXPECT_NE(result.summed_grads->data, nullptr); // still produces a gradient (partial)
    EXPECT_EQ(result.summed_grads->error_no, EVENTPROP_BATCH_PARTIAL_ACCUMULATION);

    eventprop_free_spikes_batch(spikes);
    eventprop_free_accumulated_backward_result(result);
}

// ──────────────────────────────────────────────────────────────
// backward_maxima_batch_accumulate
// ──────────────────────────────────────────────────────────────

TEST(BatchAccumulateBackwardMaxima, ValidBatchHasNoErrors) {
    std::vector<double> w = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // 2x3
    std::vector<double> times0 = {0.01, 0.02}, times1 = {0.01, 0.02};
    std::vector<int> sources0 = {0, 1}, sources1 = {0, 1};

    EventPropInput inputs[2];
    inputs[0] = make_input(w, 2, 3, times0, sources0);
    inputs[1] = make_input(w, 2, 3, times1, sources1);

    SpikesBatch spikes = eventprop_compute_spikes_batch(inputs, 2);
    MaximaBatch maxima = eventprop_compute_maxima_batch(inputs, spikes);

    // Set errors before backward pass
    for (int i = 0; i < 2; ++i) {
        maxima.outputs[i].errors[0] = 1.0;
    }

    AccumulatedBackwardResult result = eventprop_backward_maxima_batch_accumulate(inputs, maxima);

    EXPECT_EQ(result.has_errors, 0);
    EXPECT_EQ(result.summed_grads->error_no, EVENTPROP_OK);
    EXPECT_NE(result.summed_grads->data, nullptr);
    EXPECT_EQ(result.batch_size, 2);

    eventprop_free_spikes_batch(spikes);
    eventprop_free_maxima_batch(maxima);
    eventprop_free_accumulated_backward_result(result);
}

TEST(BatchAccumulateBackwardMaxima, OneBadSampleSetsHasErrors) {
    std::vector<double> w = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // 2x3
    std::vector<double> times0 = {0.01, 0.02}, times1 = {0.01, 0.02};
    std::vector<int> sources0 = {0, 1}, sources1 = {0, 1};

    EventPropInput inputs[2];
    inputs[0] = make_input(w, 2, 3, times0, sources0);
    inputs[1] = make_input(w, 2, 3, times1, sources1);

    SpikesBatch spikes = eventprop_compute_spikes_batch(inputs, 2);
    MaximaBatch maxima = eventprop_compute_maxima_batch(inputs, spikes);

    // Set errors before backward pass
    for (int i = 0; i < 2; ++i) {
        maxima.outputs[i].errors[0] = 1.0;
    }

    // Corrupt one input after computing maxima but before backward
    inputs[1].n_pre = -1;

    AccumulatedBackwardResult result = eventprop_backward_maxima_batch_accumulate(inputs, maxima);

    EXPECT_EQ(result.has_errors, 1);
    EXPECT_NE(result.summed_grads->data, nullptr); // still produces a gradient (partial)
    EXPECT_EQ(result.summed_grads->error_no, EVENTPROP_BATCH_PARTIAL_ACCUMULATION);

    eventprop_free_spikes_batch(spikes);
    eventprop_free_maxima_batch(maxima);
    eventprop_free_accumulated_backward_result(result);
}

TEST(BatchAccumulateBackwardMaxima, ZeroBatchSizeSetsHasErrors) {
    MaximaBatch maxima = (MaximaBatch){};

    AccumulatedBackwardResult result = eventprop_backward_maxima_batch_accumulate(nullptr, maxima);
    EXPECT_EQ(result.error_no, EVENTPROP_BATCHSIZE_ZERO_OR_NEG);
    EXPECT_EQ(result.has_errors, 1);

    eventprop_free_accumulated_backward_result(result);
}

// ──────────────────────────────────────────────────────────────
// dropout_spikes_batch
// ──────────────────────────────────────────────────────────────

TEST(BatchDropout, NoDropoutKeepsAllSpikes) {
    std::vector<double> times = {0.01, 0.02, 0.03, 0.04};
    std::vector<int> sources = {0, 1, 0, 1};
    std::vector<int> n_per_sample = {2, 2};

    DropoutBatch batch = eventprop_dropout_spikes_batch(
        times.data(), sources.data(), n_per_sample.data(),
        2, 0.0, 42); // p_drop=0 → keep everything

    EXPECT_EQ(batch.has_errors, 0);
    for (int i = 0; i < 2; ++i) {
        EXPECT_EQ(batch.outputs[i].n_spikes, 2);
    }

    eventprop_free_dropout_batch(batch);
}

TEST(BatchDropout, FullDropoutRemovesAllSpikes) {
    std::vector<double> times = {0.01, 0.02, 0.03, 0.04};
    std::vector<int> sources = {0, 1, 0, 1};
    std::vector<int> n_per_sample = {2, 2};

    DropoutBatch batch = eventprop_dropout_spikes_batch(
        times.data(), sources.data(), n_per_sample.data(),
        2, 1.0, 42); // p_drop=1 → drop everything

    EXPECT_EQ(batch.has_errors, 0);
    for (int i = 0; i < 2; ++i) {
        EXPECT_EQ(batch.outputs[i].n_spikes, 0);
    }

    eventprop_free_dropout_batch(batch);
}

TEST(BatchDropout, ZeroBatchSizeReturnsError) {
    DropoutBatch batch = eventprop_dropout_spikes_batch(
        nullptr, nullptr, nullptr, 0, 0.5, 42);
    EXPECT_EQ(batch.error_no, EVENTPROP_BATCHSIZE_ZERO_OR_NEG);
    EXPECT_EQ(batch.batch_size, 0);
    EXPECT_EQ(batch.has_errors, 0);
    eventprop_free_dropout_batch(batch);
}

TEST(BatchDropout, NullInputsReturnError) {
    std::vector<double> times = {0.01, 0.02};
    std::vector<int> sources = {0, 1};
    std::vector<int> n_per_sample = {2};

    // Each required pointer NULLed in turn — all must report EVENTPROP_BATCH_EMPTY
    DropoutBatch batch = eventprop_dropout_spikes_batch(
        nullptr, sources.data(), n_per_sample.data(), 1, 0.5, 42);
    EXPECT_EQ(batch.error_no, EVENTPROP_BATCH_EMPTY);
    eventprop_free_dropout_batch(batch);

    batch = eventprop_dropout_spikes_batch(
        times.data(), nullptr, n_per_sample.data(), 1, 0.5, 42);
    EXPECT_EQ(batch.error_no, EVENTPROP_BATCH_EMPTY);
    eventprop_free_dropout_batch(batch);

    batch = eventprop_dropout_spikes_batch(
        times.data(), sources.data(), nullptr, 1, 0.5, 42);
    EXPECT_EQ(batch.error_no, EVENTPROP_BATCH_EMPTY);
    eventprop_free_dropout_batch(batch);
}

TEST(BatchDropout, PartialDropoutReducesCount) {
    // With p_drop=0.5 and a fixed seed, we expect some but not all spikes to survive
    std::vector<double> times(100, 0.01);
    std::vector<int> sources(100, 0);
    std::vector<int> n_per_sample = {100};

    DropoutBatch batch = eventprop_dropout_spikes_batch(
        times.data(), sources.data(), n_per_sample.data(),
        1, 0.5, 42);

    EXPECT_EQ(batch.has_errors, 0);
    // Should drop roughly half (deterministic with seed)
    EXPECT_GT(batch.outputs[0].n_spikes, 0);
    EXPECT_LT(batch.outputs[0].n_spikes, 100);

    eventprop_free_dropout_batch(batch);
}

TEST(BatchDropout, InvalidDropoutProbReturnsError) {
    std::vector<double> times = {0.01, 0.02};
    std::vector<int> sources = {0, 1};
    std::vector<int> n_per_sample = {2};

    // p_drop > 1 would silently drop everything — must be rejected
    DropoutBatch batch = eventprop_dropout_spikes_batch(
        times.data(), sources.data(), n_per_sample.data(), 1, 1.5, 42);
    EXPECT_EQ(batch.error_no, EVENTPROP_INVALID_DROPOUT_PROB);
    EXPECT_EQ(batch.batch_size, 0);
    eventprop_free_dropout_batch(batch);

    // Negative probability is equally invalid
    batch = eventprop_dropout_spikes_batch(
        times.data(), sources.data(), n_per_sample.data(), 1, -0.1, 42);
    EXPECT_EQ(batch.error_no, EVENTPROP_INVALID_DROPOUT_PROB);
    eventprop_free_dropout_batch(batch);

    // NaN probability must be rejected too
    batch = eventprop_dropout_spikes_batch(
        times.data(), sources.data(), n_per_sample.data(), 1, std::nan(""), 42);
    EXPECT_EQ(batch.error_no, EVENTPROP_INVALID_DROPOUT_PROB);
    eventprop_free_dropout_batch(batch);
}

TEST(BatchDropout, NegativeSampleCountReturnsError) {
    std::vector<double> times = {0.01, 0.02};
    std::vector<int> sources = {0, 1};
    std::vector<int> n_per_sample = {2, -1};

    DropoutBatch batch = eventprop_dropout_spikes_batch(
        times.data(), sources.data(), n_per_sample.data(), 2, 0.5, 42);
    EXPECT_EQ(batch.error_no, EVENTPROP_NEGATIVE_SAMPLE_COUNT);
    EXPECT_EQ(batch.batch_size, 0);
    eventprop_free_dropout_batch(batch);
}

TEST(BatchAccumulate, DimensionMismatchSkippedAsPartial) {
    // Sample 1 has smaller dims than requested — must be excluded, not read OOB
    BackwardsBatch batch = (BackwardsBatch){};
    batch.batch_size = 2;
    batch.error_no = EVENTPROP_OK;
    batch.outputs = (BackwardOutput*)calloc(2, sizeof(BackwardOutput));

    batch.outputs[0].error_no = EVENTPROP_OK;
    batch.outputs[0].n_pre = 2;
    batch.outputs[0].n_post = 3;
    batch.outputs[0].n_spikes = 0;
    batch.outputs[0].grad = (double*)malloc(6 * sizeof(double));
    for (int i = 0; i < 6; ++i) batch.outputs[0].grad[i] = 1.0;
    batch.outputs[0].spike_errors = nullptr;

    batch.outputs[1].error_no = EVENTPROP_OK;
    batch.outputs[1].n_pre = 1;
    batch.outputs[1].n_post = 3;
    batch.outputs[1].n_spikes = 0;
    batch.outputs[1].grad = (double*)malloc(3 * sizeof(double));
    for (int i = 0; i < 3; ++i) batch.outputs[1].grad[i] = 100.0;
    batch.outputs[1].spike_errors = nullptr;

    GradientOutput sum = eventprop_accumulate_gradients(batch, 2, 3);
    EXPECT_EQ(sum.error_no, EVENTPROP_BATCH_PARTIAL_ACCUMULATION);
    for (int i = 0; i < 6; ++i) EXPECT_DOUBLE_EQ(sum.data[i], 1.0);

    eventprop_free_gradient_output(sum);
    eventprop_free_backwards_batch(batch);
}

TEST(BatchAccumulate, ErrorStructWithNullOutputsFreesSafely) {
    // Error paths return batch_size > 0 with outputs == NULL — the free
    // functions must not dereference the NULL array.
    AccumulatedBackwardResult r = (AccumulatedBackwardResult){};
    r.batch_size = 3;
    r.error_no = EVENTPROP_ALLOC_BATCH_OUTPUTS;
    r.summed_grads = nullptr;
    r.outputs = nullptr;
    eventprop_free_accumulated_backward_result(r);

    SpikesBatch sb = (SpikesBatch){};
    sb.batch_size = 3;
    sb.error_no = EVENTPROP_ALLOC_BATCH_OUTPUTS;
    sb.spikes = nullptr;
    eventprop_free_spikes_batch(sb);

    BackwardsBatch wb = (BackwardsBatch){};
    wb.batch_size = 3;
    wb.outputs = nullptr;
    eventprop_free_backwards_batch(wb);

    MaximaBatch mb = (MaximaBatch){};
    mb.batch_size = 3;
    mb.outputs = nullptr;
    eventprop_free_maxima_batch(mb);

    DropoutBatch db = (DropoutBatch){};
    db.batch_size = 3;
    db.outputs = nullptr;
    eventprop_free_dropout_batch(db);
}

// ──────────────────────────────────────────────────────────────
// Free functions for batch structs (null safety)
// ──────────────────────────────────────────────────────────────

TEST(BatchFree, NullPointerSafety) {
    // Free functions take values by value — zero-init structs are safe to free
    eventprop_free_spikes_batch((SpikesBatch){});
    eventprop_free_maxima_batch((MaximaBatch){});
    eventprop_free_backwards_batch((BackwardsBatch){});
    eventprop_free_gradient_output((GradientOutput){});
    eventprop_free_dropout_batch((DropoutBatch){});
    eventprop_free_accumulated_backward_result((AccumulatedBackwardResult){});
}


