"""Loss layers for Cython library — delegates to LIFLayer/LILayer forward/backward."""

from typing import NamedTuple
import numpy as np

from .li_layer import LILayer, LILayerParameters
from .lif_layer import LIFLayer, LIFLayerParameters


# fmt: off
class TTFSCrossEntropyLossParameters(NamedTuple):
    lif_parameters : LIFLayerParameters = LIFLayerParameters()
    alpha          : float = 1e-2
    tau0           : float = 2e-3  # s
    tau1           : float = 10e-3 # s
# fmt: on

VMaxCrossEntropyLossParameters = LILayerParameters


class TTFSCrossEntropyLoss(LIFLayer):
    def __init__(
        self,
        loss_parameters: TTFSCrossEntropyLossParameters = TTFSCrossEntropyLossParameters(),
        **kwargs
    ):
        super().__init__(parameters=loss_parameters.lif_parameters, **kwargs)
        self.loss_parameters = loss_parameters
        self.first_spike_times = None
        self.first_spike_idxs = None

    def forward(self, input_batch):
        super().forward(input_batch)
        self._batch_idxs = np.arange(len(self.post_batch))
        self.n_batch = len(self.post_batch)
        # Find first spike times
        self._find_first_spikes()
        self._ran_forward = True

    def _find_first_spikes(self):
        n_post = self.parameters.n
        self.first_spike_times = np.full((self.n_batch, n_post), np.nan)
        self.first_spike_idxs = np.full((self.n_batch, n_post), -1, dtype=int)
        for batch_idx in range(self.n_batch):
            spikes = self.post_batch[batch_idx]
            fts = spikes.first_spike_times
            fidxs = spikes.first_spike_idxs
            # first_spike_times/idxs may be smaller than n_post if not all neurons fired
            for nrn in range(len(fts)):
                self.first_spike_times[batch_idx, nrn] = fts[nrn]
                self.first_spike_idxs[batch_idx, nrn] = fidxs[nrn]

    def get_losses(self, labels: np.ndarray):
        """
        Compute cross-entropy losses over first spike times
        """
        if not self._ran_forward:
            raise RuntimeError("Run forward first!")
        t_labels = self.first_spike_times[self._batch_idxs, labels]
        sum0 = np.nansum(
            np.exp(-self.first_spike_times / self.loss_parameters.tau0), axis=1
        )
        loss = -np.log(np.exp(-t_labels / self.loss_parameters.tau0) / sum0)
        loss += self.loss_parameters.alpha * (
            np.exp(t_labels / self.loss_parameters.tau1) - 1
        )
        return loss

    def get_accuracy(self, labels: np.ndarray):
        t_labels = self.first_spike_times[self._batch_idxs, labels]
        results = list()
        nan_masks = ~np.isnan(self.first_spike_times)
        for batch_idx in self._batch_idxs:
            # check if label spike exists
            if np.isnan(t_labels[batch_idx]):
                results.append(0)
                continue
            # check if all other spikes come later
            mask = nan_masks[batch_idx]
            if np.all(self.first_spike_times[batch_idx][mask] >= t_labels[batch_idx]):
                results.append(1)
            else:
                results.append(0)
        return np.mean(results)

    def backward(self, labels: np.ndarray):
        if not self._ran_forward:
            raise RuntimeError("Run forward first!")
        tau0, tau1, alpha = (
            self.loss_parameters.tau0,
            self.loss_parameters.tau1,
            self.loss_parameters.alpha,
        )
        sum0 = np.nansum(np.exp(-self.first_spike_times / tau0), axis=1)
        # set sum to inf if no spikes at all
        sum0[sum0 == 0] = np.inf
        # compute error for label neuron first
        t_labels = self.first_spike_times[self._batch_idxs, labels]
        exp_t_label = np.exp(-t_labels / tau0)
        label_error = 1 / tau0 - exp_t_label / (tau0 * sum0)
        label_error += alpha / tau1 * np.exp(t_labels / tau1)
        label_error /= self.n_batch
        # errors for all non-label neurons (label gets its own term above)
        errors = -1 / (tau0 * sum0[:, None]) * np.exp(-self.first_spike_times / tau0)
        errors /= self.n_batch
        # One vector write per sample into the caller-owned errors buffer,
        # replacing the per-neuron set_error loop (a Python call + property
        # lookup per neuron per sample). The spike backward reads these buffers
        # via _error_arrays, so writing the whole row at once is equivalent.
        for batch_idx in range(self.n_batch):
            if np.isnan(label_error[batch_idx]):
                continue
            spikes = self.post_batch[batch_idx]
            idxs = self.first_spike_idxs[batch_idx]
            row_err = errors[batch_idx]
            lbl = labels[batch_idx]
            for nrn_idx in range(self.parameters.n):
                sidx = idxs[nrn_idx]
                if sidx < 0:
                    continue
                if nrn_idx == lbl:
                    spikes.errors[sidx] = label_error[batch_idx]
                elif not np.isnan(row_err[nrn_idx]):
                    spikes.errors[sidx] = row_err[nrn_idx]
        self._ran_backward = True
        super().backward()


class VMaxCrossEntropyLoss(LILayer):
    def forward(self, input_batch):
        super().forward(input_batch)
        self.n_batch = len(self.input_batch)
        # Stack the per-sample maxima values into one (n_batch, n) array.
        # Use a numerically stable softmax: subtract the per-sample max before
        # exp. Raw maxima can be large (trained MNIST weights), and the naive
        # exp(v)/sum(exp(v)) overflows to inf/inf -> NaN. The reference pybind11
        # path overflows too; the stable form matches it where it is finite and
        # stays finite where it is not.
        self.values_batch = np.stack(
            [self.maxima_batch[i].values for i in range(self.n_batch)]
        )
        self._vmax_shift = self.values_batch.max(axis=1, keepdims=True)
        self.exp_values = np.exp(self.values_batch - self._vmax_shift)
        self.sum0 = self.exp_values.sum(axis=1)
        self._ran_forward = True

    def get_losses(self, labels: np.ndarray):
        """Compute cross-entropy loss over voltage maxima."""
        if not self._ran_forward:
            raise RuntimeError("Run forward first!")
        loss = [
            -np.log(self.exp_values[batch_idx][labels[batch_idx]] / self.sum0[batch_idx])
            for batch_idx in range(self.n_batch)
        ]
        return loss

    def get_predictions(self) -> np.ndarray:
        if not self._ran_forward:
            raise RuntimeError("Run forward first!")
        return np.array([np.argmax(v) for v in self.values_batch])

    def get_accuracy(self, labels: np.ndarray):
        return np.mean(
            [
                np.sum(self.values_batch[batch_idx] >= self.values_batch[batch_idx][labels[batch_idx]])
                == 1
                for batch_idx in range(self.n_batch)
            ]
        )

    def backward(self, labels: np.ndarray):
        if not self._ran_forward:
            raise RuntimeError("Run forward first!")
        n_batch = len(self.maxima_batch)
        inv_nb = 1.0 / n_batch
        for batch_idx in range(self.n_batch):
            # softmax/n_batch, with the label neuron getting the -1/n_batch
            # cross-entropy term. One vector write into the C-owned errors
            # buffer replaces the per-neuron set_error loop (a Python call +
            # property lookup per neuron per sample).
            error = self.exp_values[batch_idx] / self.sum0[batch_idx]
            error *= inv_nb
            errors = self.maxima_batch[batch_idx].errors
            np.copyto(errors, error)
            lbl = labels[batch_idx]
            errors[lbl] = error[lbl] - inv_nb
        super().backward()
