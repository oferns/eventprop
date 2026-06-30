/**
 * @file tests.cpp
 * @brief EventProp C++ (Eigen) implementation test suite — mirrors src/EventProp.C/tests.c.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include "eventprop_eigen_stripped.h"

static constexpr double EPSILON = 1e-6;
static int tests_run = 0, tests_passed = 0, tests_failed = 0;

#define TEST(name) \
    do { printf("  Test: %-60s ", name); tests_run++; } while (0)

#define ASSERT(cond, msg) \
    if (!(cond)) { printf("FAIL [%s]\n", msg); tests_failed++; return; }

#define ASSERT_NEAR(a, b, tol) \
    if (fabs((a) - (b)) > (tol)) { printf("FAIL [%g vs %g]\n", (double)(a), (double)(b)); tests_failed++; return; }

#define TEST_END \
    do { printf("PASS\n"); tests_passed++; } while (0)

static Spikes gen_data(int n_pre, int n_post, int n_spikes)
{
    Eigen::ArrayXd times(n_spikes);
    Eigen::ArrayXi sources(n_spikes);
    double t_end = fmin(0.1, (double)n_spikes / 50.0);
    for (int i = 0; i < n_spikes; i++) {
        times[i] = 0.01 + (i + 1.0) / (n_spikes + 1.0) * (t_end - 0.01);
        sources[i] = i % n_pre;
    }
    return Spikes(std::move(times), std::move(sources));
}

static RowMatrixXd gen_weights(int n_pre, int n_post)
{
    RowMatrixXd w(n_pre, n_post);
    for (int i = 0; i < n_pre * n_post; i++)
        w.data()[i] = 5.0 + sin((double)i * 0.1);
    return w;
}

static void test_compute_sums(void)
{
    TEST("ComputeSums Tiny");
    auto [sum0, sum1] = compute_sums(gen_weights(10, 5), gen_data(10, 5, 20), 0.02, 0.005);
    for (int j = 0; j < 5; j++) { ASSERT_NEAR(sum0(0, j), 0.0, EPSILON); ASSERT_NEAR(sum1(0, j), 0.0, EPSILON); }
    TEST_END;
}

static void test_compute_spikes(void)
{
    TEST("ComputeSpikes Tiny");
    Spikes fwd = compute_spikes(gen_weights(10, 5), gen_data(10, 5, 20), 1.0, 0.02, 0.005);
    ASSERT(fwd.n_spikes >= 0, "negative spike count");
    for (int i = 1; i < fwd.n_spikes; i++) ASSERT(fwd.times[i] >= fwd.times[i - 1], "times not sorted");
    TEST_END;
}

static void test_backward_spikes(void)
{
    TEST("BackwardSpikes Tiny");
    auto w = gen_weights(10, 5);
    Spikes input = gen_data(10, 5, 20);
    Spikes fwd = compute_spikes(w, input, 1.0, 0.02, 0.005);
    for (int i = 0; i < fwd.n_spikes; i++) fwd.set_error(i, 1.0);
    auto [grad, se] = backward(input, fwd, w, 1.0, 0.02, 0.005);
    for (int i = 0; i < grad.size(); i++) { ASSERT(!std::isnan(grad.data()[i]), "NaN"); ASSERT(!std::isinf(grad.data()[i]), "Inf"); }
    TEST_END;
}

static void test_compute_maxima(void)
{
    TEST("ComputeMaxima Tiny");
    Maxima maxima = compute_maxima(gen_weights(10, 5), gen_data(10, 5, 20), 0.02, 0.005);
    for (int i = 0; i < 5; i++) ASSERT(!std::isnan(maxima.times[i]) || std::isnan(maxima.values[i]), "mismatch");
    TEST_END;
}

static void test_backward_maxima(void)
{
    TEST("BackwardMaxima Tiny");
    auto w = gen_weights(10, 5);
    Spikes input = gen_data(10, 5, 20);
    Maxima maxima = compute_maxima(w, input, 0.02, 0.005);
    for (int i = 0; i < 5; i++) maxima.set_error(i, 1.0);
    auto [grad, se] = backward(input, maxima, w, 0.02, 0.005);
    double gs = 0;
    for (int i = 0; i < grad.size(); i++) { ASSERT(!std::isnan(grad.data()[i]), "NaN"); gs += fabs(grad.data()[i]); }
    ASSERT(gs > 1e-12, "all gradients zero");
    TEST_END;
}

static void test_backward_maxima_zero_weights(void)
{
    TEST("BackwardMaxima ZeroWeights -> ZeroGradients");
    RowMatrixXd w(10, 5);
    w.setZero();
    Spikes input = gen_data(10, 5, 20);
    Maxima maxima = compute_maxima(w, input, 0.02, 0.005);
    for (int i = 0; i < 5; i++) maxima.set_error(i, 1.0);
    auto [grad, se] = backward(input, maxima, w, 0.02, 0.005);
    for (int i = 0; i < grad.size(); i++) ASSERT_NEAR(grad.data()[i], 0.0, EPSILON);
    TEST_END;
}

static void test_backward_maxima_linear_scale(void)
{
    TEST("BackwardMaxima GradientScalesLinearly");
    auto w = gen_weights(10, 5);
    Spikes input = gen_data(10, 5, 20);
    Maxima maxima = compute_maxima(w, input, 0.02, 0.005);
    for (int i = 0; i < 5; i++) maxima.set_error(i, 1.0);
    auto [grad1, se1] = backward(input, maxima, w, 0.02, 0.005);
    for (int i = 0; i < 5; i++) maxima.set_error(i, 2.0);
    auto [grad2, se2] = backward(input, maxima, w, 0.02, 0.005);
    for (int i = 0; i < grad1.size(); i++) ASSERT_NEAR(grad2.data()[i], 2.0 * grad1.data()[i], EPSILON);
    TEST_END;
}

static void test_compute_first_spike(void)
{
    TEST("ComputeFirstSpike normal + no-spike neuron");
    auto w = gen_weights(10, 5);
    Spikes fwd = compute_spikes(w, gen_data(10, 5, 20), 1.0, 0.02, 0.005);
    std::vector<double> first_times(5, NAN);
    std::vector<int> first_idxs(5, -1);
    for (int i = 0; i < fwd.n_spikes; i++) {
        int nrn = fwd.sources[i];
        if (first_idxs[nrn] == -1) { first_times[nrn] = fwd.times[i]; first_idxs[nrn] = i; }
    }
    int count = 0;
    for (int j = 0; j < 5; j++) {
        if (!std::isnan(first_times[j])) {
            ASSERT(first_idxs[j] >= 0 && first_times[j] > 0.0, "bad first spike"); count++; }
    }
    ASSERT(count > 0, "no neurons fired");
    TEST_END;
}

static void test_zero_spikes(void)
{
    TEST("ZeroSpikes -> no output spikes");
    Eigen::ArrayXd times(0);
    Eigen::ArrayXi sources(0);
    Spikes input(std::move(times), std::move(sources));
    auto w = gen_weights(10, 5);
    Spikes fwd = compute_spikes(w, input, 1.0, 0.02, 0.005);
    ASSERT(fwd.n_spikes == 0, "expected zero output spikes");
    TEST_END;
}

static void test_negative_weights(void)
{
    TEST("NegativeWeights -> valid output");
    RowMatrixXd w(10, 5);
    for (int i = 0; i < 50; i++) w.data()[i] = -3.0 + sin((double)i * 0.2);
    Spikes fwd = compute_spikes(w, gen_data(10, 5, 20), 1.0, 0.02, 0.005);
    ASSERT(fwd.n_spikes >= 0, "negative spike count");
    for (int i = 1; i < fwd.n_spikes; i++) ASSERT(fwd.times[i] >= fwd.times[i - 1], "times not sorted");
    TEST_END;
}

static void test_backward_spikes_linear_scale(void)
{
    TEST("BackwardSpikes GradientScalesLinearly");
    auto w = gen_weights(10, 5);
    Spikes input = gen_data(10, 5, 20);
    Spikes fwd = compute_spikes(w, input, 1.0, 0.02, 0.005);
    for (int i = 0; i < fwd.n_spikes; i++) fwd.set_error(i, 1.0);
    auto [grad1, se1] = backward(input, fwd, w, 1.0, 0.02, 0.005);
    for (int i = 0; i < fwd.n_spikes; i++) fwd.set_error(i, 3.0);
    auto [grad2, se2] = backward(input, fwd, w, 1.0, 0.02, 0.005);
    for (int i = 0; i < grad1.size(); i++) ASSERT_NEAR(grad2.data()[i], 3.0 * grad1.data()[i], EPSILON);
    TEST_END;
}

static void test_single_neuron(void)
{
    TEST("SingleNeuron 5x1");
    auto w = gen_weights(5, 1);
    Spikes fwd = compute_spikes(w, gen_data(5, 1, 10), 1.0, 0.02, 0.005);
    ASSERT(fwd.n_spikes >= 0, "negative spike count");
    TEST_END;
}

static void test_wide_fan_in(void)
{
    TEST("WideFanIn 100x2");
    auto w = gen_weights(100, 2);
    Spikes fwd = compute_spikes(w, gen_data(100, 2, 50), 1.0, 0.02, 0.005);
    ASSERT(fwd.n_spikes >= 0, "negative spike count");
    TEST_END;
}

static void test_tall_fan_out(void)
{
    TEST("TallFanOut 3x50");
    auto w = gen_weights(3, 50);
    Spikes fwd = compute_spikes(w, gen_data(3, 50, 15), 1.0, 0.02, 0.005);
    ASSERT(fwd.n_spikes >= 0, "negative spike count");
    TEST_END;
}

static void test_different_tau(void)
{
    TEST("DifferentTauMem values");
    auto w = gen_weights(10, 5);
    Spikes fwd1 = compute_spikes(w, gen_data(10, 5, 20), 1.0, 0.02, 0.005);
    Spikes fwd2 = compute_spikes(w, gen_data(10, 5, 20), 1.0, 0.05, 0.005);
    ASSERT(fwd1.n_spikes != fwd2.n_spikes || fabs(fwd1.times[0] - fwd2.times[0]) > EPSILON, "tau should affect output");
    TEST_END;
}

static void test_large_burst(void)
{
    TEST("LargeBurst 200x80 500spikes");
    auto w = gen_weights(200, 80);
    Eigen::ArrayXd times(500);
    Eigen::ArrayXi sources(500);
    for (int i = 0; i < 500; i++) {
        times[i] = 0.01 + (i + 1.0) / 501.0 * 0.09;
        sources[i] = i % 200;
    }
    Spikes input(std::move(times), std::move(sources));
    Spikes fwd = compute_spikes(w, input, 1.0, 0.02, 0.005);
    ASSERT(fwd.n_spikes >= 0, "negative spike count");
    for (int i = 1; i < fwd.n_spikes; i++) ASSERT(fwd.times[i] >= fwd.times[i - 1], "times not sorted");
    TEST_END;
}

static void test_compute_sums_large(void)
{
    TEST("ComputeSums Large 200x80");
    auto w = gen_weights(200, 80);
    Eigen::ArrayXd times(500);
    Eigen::ArrayXi sources(500);
    for (int i = 0; i < 500; i++) { times[i] = 0.01 + (i + 1.0) / 501.0 * 0.09; sources[i] = i % 200; }
    Spikes input(std::move(times), std::move(sources));
    auto [sum0, sum1] = compute_sums(w, input, 0.02, 0.005);
    ASSERT(sum0.cols() == 80, "bad sums cols");
    for (int i = 0; i < sum0.size(); i++) { ASSERT(!std::isnan(sum0.data()[i]), "NaN in sums"); }
    TEST_END;
}

static void test_backward_spikes_large(void)
{
    TEST("BackwardSpikes Large 200x80");
    auto w = gen_weights(200, 80);
    Eigen::ArrayXd times(500);
    Eigen::ArrayXi sources(500);
    for (int i = 0; i < 500; i++) { times[i] = 0.01 + (i + 1.0) / 501.0 * 0.09; sources[i] = i % 200; }
    Spikes input(std::move(times), std::move(sources));
    Spikes fwd = compute_spikes(w, input, 1.0, 0.02, 0.005);
    for (int i = 0; i < fwd.n_spikes; i++) fwd.set_error(i, 1.0);
    auto [grad, se] = backward(input, fwd, w, 1.0, 0.02, 0.005);
    ASSERT(grad.rows() == 200 && grad.cols() == 80, "bad gradient shape");
    for (int i = 0; i < grad.size(); i++) { ASSERT(!std::isnan(grad.data()[i]), "NaN"); }
    TEST_END;
}

static void test_compute_maxima_large(void)
{
    TEST("ComputeMaxima Large 200x80");
    auto w = gen_weights(200, 80);
    Eigen::ArrayXd times(500);
    Eigen::ArrayXi sources(500);
    for (int i = 0; i < 500; i++) { times[i] = 0.01 + (i + 1.0) / 501.0 * 0.09; sources[i] = i % 200; }
    Spikes input(std::move(times), std::move(sources));
    Maxima maxima = compute_maxima(w, input, 0.02, 0.005);
    ASSERT(maxima.times.size() == 80 && maxima.values.size() == 80, "bad maxima shape");
    TEST_END;
}

static void test_backward_maxima_large(void)
{
    TEST("BackwardMaxima Large 200x80");
    auto w = gen_weights(200, 80);
    Eigen::ArrayXd times(500);
    Eigen::ArrayXi sources(500);
    for (int i = 0; i < 500; i++) { times[i] = 0.01 + (i + 1.0) / 501.0 * 0.09; sources[i] = i % 200; }
    Spikes input(std::move(times), std::move(sources));
    Maxima maxima = compute_maxima(w, input, 0.02, 0.005);
    for (int i = 0; i < 80; i++) maxima.set_error(i, 1.0);
    auto [grad, se] = backward(input, maxima, w, 0.02, 0.005);
    ASSERT(grad.rows() == 200 && grad.cols() == 80, "bad gradient shape");
    for (int i = 0; i < grad.size(); i++) { ASSERT(!std::isnan(grad.data()[i]), "NaN"); }
    TEST_END;
}

static void test_simultaneous_spikes(void)
{
    TEST("SimultaneousSpikes same time different sources");
    Eigen::ArrayXd times(10);
    Eigen::ArrayXi sources(10);
    for (int i = 0; i < 10; i++) { times[i] = 0.05; sources[i] = i % 8; }
    Spikes input(std::move(times), std::move(sources));
    auto w = gen_weights(8, 4);
    Spikes fwd = compute_spikes(w, input, 1.0, 0.02, 0.005);
    ASSERT(fwd.n_spikes >= 0, "negative spike count");
    TEST_END;
}

static void test_t_zero_cluster(void)
{
    TEST("TZeroCluster spikes near t=0");
    Eigen::ArrayXd times(20);
    Eigen::ArrayXi sources(20);
    for (int i = 0; i < 20; i++) { times[i] = 1e-6 * (i + 1); sources[i] = i % 8; }
    Spikes input(std::move(times), std::move(sources));
    auto w = gen_weights(8, 4);
    Spikes fwd = compute_spikes(w, input, 1.0, 0.02, 0.005);
    ASSERT(fwd.n_spikes >= 0, "negative spike count");
    TEST_END;
}

static void test_long_window(void)
{
    TEST("LongWindow 1s duration");
    Eigen::ArrayXd times(200);
    Eigen::ArrayXi sources(200);
    for (int i = 0; i < 200; i++) { times[i] = 0.005 + (i + 1.0) / 201.0 * 0.99; sources[i] = i % 16; }
    Spikes input(std::move(times), std::move(sources));
    auto w = gen_weights(16, 8);
    Spikes fwd = compute_spikes(w, input, 1.0, 0.02, 0.005);
    ASSERT(fwd.n_spikes >= 0, "negative spike count");
    TEST_END;
}

static void test_compute_spikes_low_threshold(void)
{
    TEST("ComputeSpikes LowThreshold (v_th=0.1) -> many spikes");
    int n_pre = 50, n_post = 20, n_spikes = 100;
    Eigen::ArrayXd times(n_spikes);
    Eigen::ArrayXi sources(n_spikes);
    for (int i = 0; i < n_spikes; i++) {
        times[i] = 0.01 + (i + 1.0) / (n_spikes + 1.0) * 0.09;
        sources[i] = i % n_pre;
    }
    Spikes input(std::move(times), std::move(sources));
    auto w = gen_weights(n_pre, n_post);

    /* Low threshold — should produce many output spikes */
    double v_th = 0.1;
    Spikes fwd = compute_spikes(w, input, v_th, 0.02, 0.005);

    printf("     Low threshold v_th=%.1f: %d output spikes\n", v_th, fwd.n_spikes);
    ASSERT(fwd.n_spikes > 0, "should produce spikes with low threshold");

    /* Verify times are sorted */
    for (int i = 1; i < fwd.n_spikes; i++)
        ASSERT(fwd.times[i] >= fwd.times[i-1], "times not sorted");

    TEST_END;
}

static void test_compute_spikes_very_low_threshold(void)
{
    TEST("ComputeSpikes VeryLowThreshold (v_th=0.01) -> Toms748 exception");
    int n_pre = 50, n_post = 20, n_spikes = 100;
    Eigen::ArrayXd times(n_spikes);
    Eigen::ArrayXi sources(n_spikes);
    for (int i = 0; i < n_spikes; i++) {
        times[i] = 0.01 + (i + 1.0) / (n_spikes + 1.0) * 0.09;
        sources[i] = i % n_pre;
    }
    Spikes input(std::move(times), std::move(sources));
    auto w = gen_weights(n_pre, n_post);

    /* Very low threshold — produces massive output spike count.
     * With OpenMP parallelization this completes successfully (per-thread
     * workload is small enough). Without OpenMP it used to throw Toms748. */
    double v_th = 0.01;
    Spikes fwd = compute_spikes(w, input, v_th, 0.02, 0.005);
    printf("     Very low threshold v_th=%.2f: %d output spikes\n", v_th, fwd.n_spikes);

    /* Test passes if we get a large but valid spike count */
    ASSERT(fwd.n_spikes > 10000, "expected massive output spike count with very low threshold");

    TEST_END;
}

int main(void)
{
    printf("EventProp C++ (Eigen) Tests\n");
    test_compute_sums();
    test_compute_spikes();
    test_backward_spikes();
    test_compute_maxima();
    test_backward_maxima();
    test_backward_maxima_zero_weights();
    test_backward_maxima_linear_scale();
    test_compute_first_spike();
    test_zero_spikes();
    test_negative_weights();
    test_backward_spikes_linear_scale();
    test_single_neuron();
    test_wide_fan_in();
    test_tall_fan_out();
    test_different_tau();
    test_large_burst();
    test_compute_sums_large();
    test_backward_spikes_large();
    test_compute_maxima_large();
    test_backward_maxima_large();
    test_simultaneous_spikes();
    test_t_zero_cluster();
    test_long_window();

    /* Low threshold tests — demonstrates Eigen's root-finder limitation */
    test_compute_spikes_low_threshold();
    test_compute_spikes_very_low_threshold();
    printf("\nResults: %d/%d passed, %d failed\n", tests_passed, tests_run, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
