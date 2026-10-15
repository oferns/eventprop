import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
from itertools import product
import unittest

from eventprop_c.loss_layer import (
    TTFSCrossEntropyLoss,
    TTFSCrossEntropyLossParameters,
    VMaxCrossEntropyLoss,
    VMaxCrossEntropyLossParameters,
)
from eventprop_c.lif_layer import LIFLayer, LIFLayerParameters
from eventprop_c.spikes import Spikes, SpikesVector
from .test_lif_layer import get_normalization_factor, get_poisson_spikes


class LossLIFLIFChainTest(unittest.TestCase):
    def test_gradient_vs_numerical_random_ttfs(self):
        np.random.seed(0)
        n_in = 10
        n_upper = 5
        n_lower = 4
        n_batch = 3
        isi = 5e-3
        t_max = 0.2
        upper_pars = LIFLayerParameters(
            n=n_upper, n_in=n_in, tau_mem=20e-3, tau_syn=5e-3
        )
        lower_pars = LIFLayerParameters(
            n=n_lower, n_in=n_upper, tau_mem=20e-3, tau_syn=5e-3
        )
        loss_pars = TTFSCrossEntropyLossParameters(lif_parameters=lower_pars)
        norm_factor = get_normalization_factor(upper_pars.tau_mem, upper_pars.tau_syn)
        w_upper = np.random.normal(0.2, 0.1, size=(n_in, n_upper)) * norm_factor
        w_lower = np.random.normal(0.2, 0.1, size=(n_upper, n_lower)) * norm_factor
        w_eps = 1e-4
        input_spikes = SpikesVector(
            [get_poisson_spikes(isi, t_max, n_in) for _ in range(n_batch)]
        )
        labels = np.arange(n_batch)
        grad_numerical_upper = np.zeros_like(w_upper)
        for batch_idx in range(n_batch):
            for idx in product(range(n_in), range(n_upper)):
                w_plus = np.copy(w_upper)
                w_plus[idx] += w_eps
                upper_layer = LIFLayer(upper_pars, w_plus)
                loss_layer = TTFSCrossEntropyLoss(loss_pars, w_in=w_lower)
                loss_layer(upper_layer(input_spikes))
                assert loss_layer.post_batch[batch_idx].n_spikes > 0
                loss_plus = loss_layer.get_losses(labels)[batch_idx]

                w_minus = np.copy(w_upper)
                w_minus[idx] -= w_eps
                upper_layer = LIFLayer(upper_pars, w_minus)
                loss_layer = TTFSCrossEntropyLoss(loss_pars, w_in=w_lower)
                loss_layer(upper_layer(input_spikes))
                assert loss_layer.post_batch[batch_idx].n_spikes > 0
                loss_minus = loss_layer.get_losses(labels)[batch_idx]

                grad_numerical_upper[idx] += (loss_plus - loss_minus) / (2 * w_eps)

        grad_numerical_lower = np.zeros_like(w_lower)
        for batch_idx in range(n_batch):
            for idx in product(range(n_upper), range(n_lower)):
                w_plus = np.copy(w_lower)
                w_plus[idx] += w_eps
                upper_layer = LIFLayer(upper_pars, w_upper)
                loss_layer = TTFSCrossEntropyLoss(loss_pars, w_in=w_plus)
                loss_layer(upper_layer(input_spikes))
                loss_plus = loss_layer.get_losses(labels)[batch_idx]

                w_minus = np.copy(w_lower)
                w_minus[idx] -= w_eps
                upper_layer = LIFLayer(upper_pars, w_upper)
                loss_layer = TTFSCrossEntropyLoss(loss_pars, w_in=w_minus)
                loss_layer(upper_layer(input_spikes))
                loss_minus = loss_layer.get_losses(labels)[batch_idx]

                grad_numerical_lower[idx] += (loss_plus - loss_minus) / (2 * w_eps)

        upper_layer = LIFLayer(upper_pars, w_upper)
        loss_layer = TTFSCrossEntropyLoss(loss_pars, w_in=w_lower)
        loss_layer(upper_layer(input_spikes))
        loss_layer.backward(labels)
        grad_numerical_lower /= n_batch
        grad_numerical_upper /= n_batch
        assert_almost_equal(grad_numerical_lower, loss_layer.gradient)
        assert_almost_equal(grad_numerical_upper, upper_layer.gradient)

    def test_gradient_vs_numerical_random_vmax(self):
        np.random.seed(0)
        n_in = 10
        n_upper = 5
        n_lower = 4
        n_batch = 3
        isi = 5e-3
        t_max = 0.2
        loss_pars = VMaxCrossEntropyLossParameters(n=n_lower, n_in=n_lower)
        upper_pars = LIFLayerParameters(
            n=n_upper, n_in=n_in, tau_mem=20e-3, tau_syn=5e-3
        )
        lower_pars = LIFLayerParameters(
            n=n_lower, n_in=n_upper, tau_mem=20e-3, tau_syn=5e-3
        )
        norm_factor = get_normalization_factor(upper_pars.tau_mem, upper_pars.tau_syn)
        w_upper = np.random.normal(0.2, 0.1, size=(n_in, n_upper)) * norm_factor
        w_lower = np.random.normal(0.2, 0.1, size=(n_upper, n_lower)) * norm_factor
        w_vmax = np.random.normal(1, 1, size=(n_lower, n_lower))
        w_eps = 1e-6
        input_spikes = SpikesVector(
            [get_poisson_spikes(isi, t_max, n_in) for _ in range(n_batch)]
        )
        labels = np.arange(n_batch)
        grad_numerical_upper = np.zeros_like(w_upper)
        for batch_idx in range(n_batch):
            for idx in product(range(n_in), range(n_upper)):
                w_plus = np.copy(w_upper)
                w_plus[idx] += w_eps
                upper_layer = LIFLayer(upper_pars, w_plus)
                lower_layer = LIFLayer(lower_pars, w_lower)
                loss_layer = VMaxCrossEntropyLoss(loss_pars)
                loss_layer.w_in = w_vmax
                loss_layer(lower_layer(upper_layer(input_spikes)))
                assert lower_layer.post_batch[batch_idx].n_spikes > 0
                loss_plus = loss_layer.get_losses(labels)[batch_idx]

                w_minus = np.copy(w_upper)
                w_minus[idx] -= w_eps
                upper_layer = LIFLayer(upper_pars, w_minus)
                lower_layer = LIFLayer(lower_pars, w_lower)
                loss_layer = VMaxCrossEntropyLoss(loss_pars)
                loss_layer.w_in = w_vmax
                loss_layer(lower_layer(upper_layer(input_spikes)))
                assert lower_layer.post_batch[batch_idx].n_spikes > 0
                loss_minus = loss_layer.get_losses(labels)[batch_idx]

                grad_numerical_upper[idx] += (loss_plus - loss_minus) / (2 * w_eps)

        grad_numerical_lower = np.zeros_like(w_lower)
        for batch_idx in range(n_batch):
            for idx in product(range(n_upper), range(n_lower)):
                w_plus = np.copy(w_lower)
                w_plus[idx] += w_eps
                upper_layer = LIFLayer(upper_pars, w_upper)
                lower_layer = LIFLayer(lower_pars, w_plus)
                loss_layer = VMaxCrossEntropyLoss(loss_pars)
                loss_layer.w_in = w_vmax
                loss_layer(lower_layer(upper_layer(input_spikes)))
                loss_plus = loss_layer.get_losses(labels)[batch_idx]

                w_minus = np.copy(w_lower)
                w_minus[idx] -= w_eps
                upper_layer = LIFLayer(upper_pars, w_upper)
                lower_layer = LIFLayer(lower_pars, w_minus)
                loss_layer = VMaxCrossEntropyLoss(loss_pars)
                loss_layer.w_in = w_vmax
                loss_layer(lower_layer(upper_layer(input_spikes)))
                loss_minus = loss_layer.get_losses(labels)[batch_idx]

                grad_numerical_lower[idx] += (loss_plus - loss_minus) / (2 * w_eps)

        upper_layer = LIFLayer(upper_pars, w_upper)
        lower_layer = LIFLayer(lower_pars, w_lower)
        loss_layer = VMaxCrossEntropyLoss(loss_pars)
        loss_layer.w_in = w_vmax
        loss_layer(lower_layer(upper_layer(input_spikes)))
        loss_layer.backward(labels)
        grad_numerical_lower /= n_batch
        grad_numerical_upper /= n_batch
        # Use rtol+atol for 3-layer chain numerical vs analytical comparison
        assert_allclose(grad_numerical_lower, lower_layer.gradient, rtol=1e-5, atol=1e-7)
        assert_allclose(grad_numerical_upper, upper_layer.gradient, rtol=1e-5, atol=1e-7)


class LIFLIFChainTest(unittest.TestCase):
    def test_gradient_vs_numerical_random(self):
        np.random.seed(0)
        n_in = 10
        n_upper = 5
        n_lower = 3
        n_batch = 5
        isi = 10e-3
        t_max = 0.1
        upper_pars = LIFLayerParameters(
            n=n_upper, n_in=n_in, tau_mem=20e-3, tau_syn=5e-3
        )
        lower_pars = LIFLayerParameters(
            n=n_lower, n_in=n_upper, tau_mem=20e-3, tau_syn=5e-3
        )
        norm_factor = get_normalization_factor(upper_pars.tau_mem, upper_pars.tau_syn)
        w_upper = np.random.normal(0.1, 0.01, size=(n_in, n_upper)) * norm_factor
        w_lower = np.random.normal(0.1, 0.01, size=(n_upper, n_lower)) * norm_factor
        w_eps = 1e-6
        input_spikes = SpikesVector(
            [get_poisson_spikes(isi, t_max, n_in) for _ in range(n_batch)]
        )
        grad_numerical_upper = np.zeros_like(w_upper)
        for batch_idx in range(n_batch):
            for idx in product(range(n_in), range(n_upper)):
                w_plus = np.copy(w_upper)
                w_plus[idx] += w_eps
                upper_layer = LIFLayer(upper_pars, w_plus)
                lower_layer = LIFLayer(lower_pars, w_lower)
                lower_layer(upper_layer(input_spikes))
                assert lower_layer.post_batch[batch_idx].n_spikes > 0
                t_plus = np.sum(lower_layer.post_batch[batch_idx].times)

                w_minus = np.copy(w_upper)
                w_minus[idx] -= w_eps
                upper_layer = LIFLayer(upper_pars, w_minus)
                lower_layer = LIFLayer(lower_pars, w_lower)
                lower_layer(upper_layer(input_spikes))
                assert lower_layer.post_batch[batch_idx].n_spikes > 0
                t_minus = np.sum(lower_layer.post_batch[batch_idx].times)

                grad_numerical_upper[idx] += (t_plus - t_minus) / (2 * w_eps)

        grad_numerical_lower = np.zeros_like(w_lower)
        for batch_idx in range(n_batch):
            for idx in product(range(n_upper), range(n_lower)):
                w_plus = np.copy(w_lower)
                w_plus[idx] += w_eps
                upper_layer = LIFLayer(upper_pars, w_upper)
                lower_layer = LIFLayer(lower_pars, w_plus)
                lower_layer(upper_layer(input_spikes))
                assert lower_layer.post_batch[batch_idx].n_spikes > 0
                t_plus = np.sum(lower_layer.post_batch[batch_idx].times)

                w_minus = np.copy(w_lower)
                w_minus[idx] -= w_eps
                upper_layer = LIFLayer(upper_pars, w_upper)
                lower_layer = LIFLayer(lower_pars, w_minus)
                lower_layer(upper_layer(input_spikes))
                assert lower_layer.post_batch[batch_idx].n_spikes > 0
                t_minus = np.sum(lower_layer.post_batch[batch_idx].times)

                grad_numerical_lower[idx] += (t_plus - t_minus) / (2 * w_eps)

        upper_layer = LIFLayer(upper_pars, w_upper)
        lower_layer = LIFLayer(lower_pars, w_lower)
        lower_layer(upper_layer(input_spikes))
        for spikes in lower_layer.post_batch:
            for spike_idx in range(spikes.n_spikes):
                spikes.set_error(spike_idx, 1)
        lower_layer.backward()
        assert_almost_equal(grad_numerical_lower, lower_layer.gradient)
        assert_almost_equal(grad_numerical_upper, upper_layer.gradient)


if __name__ == "__main__":
    unittest.main()


class SpikeErrorPropagationLifetimeTest(unittest.TestCase):
    """The ancestor layer must read spike errors from live memory.

    LIFLayer.backward() hands the *previous* layer zero-copy numpy views into
    the native AccumulatedBackwardResult returned by
    backward_spikes_batch_accumulate. That object frees the native buffer in
    __dealloc__, so if the local `result` is the only reference it dies at the
    end of backward() and the views dangle: the ancestor then reads freed heap
    memory. Observed as an identical 1.2e+79 in every sample of the batch, and
    a hidden-layer gradient ~1000x too small with most entries exactly zero.
    """

    def _chain(self, seed):
        np.random.seed(seed)
        n_in, n_hidden, n_out, n_batch = 10, 12, 6, 6
        isi, t_max = 5e-3, 0.2
        hidden_pars = LIFLayerParameters(
            n=n_hidden, n_in=n_in, tau_mem=20e-3, tau_syn=5e-3
        )
        loss_pars = TTFSCrossEntropyLossParameters(
            lif_parameters=LIFLayerParameters(
                n=n_out, n_in=n_hidden, tau_mem=20e-3, tau_syn=5e-3
            )
        )
        norm = get_normalization_factor(20e-3, 5e-3)
        w_hidden = np.random.normal(0.2, 0.1, size=(n_in, n_hidden)) * norm
        w_loss = np.random.normal(0.2, 0.1, size=(n_hidden, n_out)) * norm
        input_spikes = SpikesVector(
            [get_poisson_spikes(isi, t_max, n_in) for _ in range(n_batch)]
        )
        hidden = LIFLayer(hidden_pars, w_hidden)
        loss = TTFSCrossEntropyLoss(loss_pars)
        loss.w_in = w_loss
        loss(hidden(input_spikes))
        return hidden, loss, np.arange(n_batch)

    def test_propagated_errors_survive_the_backward_call(self):
        """Reading the propagated errors after backward() must not touch freed memory.

        The freed spike_errors blocks are ~11 KB, so allocating a batch of
        same-size-class arrays right after backward() recycles them and makes
        the dangling read deterministic. Measured with the result unheld:
        max|error| = 5.1e+77 in every sample. With the result held on the layer:
        max|error| ~ 8-11, stable across seeds.
        """
        hidden, loss, labels = self._chain(seed=1)
        loss.backward(labels)

        # Recycle the size class of the native spike_errors buffers.
        poison = [np.full(1400, -7.0) for _ in range(4000)]
        try:
            incoming = [np.asarray(hidden.post_batch[i].errors)
                        for i in range(len(hidden.post_batch))]
            self.assertTrue(any(np.any(e != 0) for e in incoming),
                            "chain produced no propagated spike errors")
            for i, e in enumerate(incoming):
                self.assertTrue(np.all(np.isfinite(e)),
                                f"sample {i}: non-finite spike error (freed memory?)")
                self.assertLess(np.abs(e).max(), 1e3,
                                f"sample {i}: spike error magnitude {np.abs(e).max():.3e} "
                                "is out of range — the native buffer was freed under it")
        finally:
            del poison

    def test_hidden_gradient_matches_an_independently_held_result(self):
        """The chain gradient must not depend on when the native result dies.

        Runs the same chain twice: once normally, once with the native
        AccumulatedBackwardResult pinned in a module-level list so its buffers
        cannot be recycled. A dangling read shows up as a gradient that is
        ~1000x too small with most entries exactly zero.
        """
        hidden_a, loss_a, labels = self._chain(seed=2)
        loss_a.backward(labels)
        grad_normal = np.array(hidden_a.gradient)

        held = []
        import eventprop_c.lif_layer as lif_mod
        original = lif_mod.backward_spikes_batch_accumulate

        def pinning(*args, **kwargs):
            result = original(*args, **kwargs)
            held.append(result)          # keep the native buffers alive
            return result

        lif_mod.backward_spikes_batch_accumulate = pinning
        try:
            hidden_b, loss_b, _ = self._chain(seed=2)
            loss_b.backward(labels)
            grad_pinned = np.array(hidden_b.gradient)
        finally:
            lif_mod.backward_spikes_batch_accumulate = original
            del held

        self.assertGreater(np.abs(grad_pinned).max(), 0.0,
                           "chain gradient is identically zero — test is vacuous")
        assert_allclose(grad_normal, grad_pinned, rtol=1e-7, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
