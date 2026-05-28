"""EventProp types: Spikes, Maxima, SpikesVector, MaximaVector."""

from __future__ import annotations
import math
import numpy as np
from typing import List, Tuple


# ── Spikes ──────────────────────────────────────────────────────────────────

class Spikes:
    """Container for post-synaptic spike times, sources, and associated data."""

    __slots__ = ("times", "sources", "errors", "currents",
                 "first_spike_times", "first_spike_idxs", "_n_spikes")

    def __init__(self, times: np.ndarray = None, sources: np.ndarray = None,
                 currents: np.ndarray = None,
                 first_spike_times=None, first_spike_idxs=None):
        self.times = np.asarray(times, dtype=np.float64) if times is not None else np.array([], dtype=np.float64)
        self.sources = np.asarray(sources, dtype=np.int32) if sources is not None else np.array([], dtype=np.int32)
        self.errors = np.zeros_like(self.times, dtype=np.float64) if self.times.size > 0 else np.array([], dtype=np.float64)
        self.currents = np.asarray(currents, dtype=np.float64) if currents is not None else self.errors.copy()
        self._n_spikes = len(self.times)

        if first_spike_times is not None and first_spike_idxs is not None:
            self.first_spike_times = list(first_spike_times)
            self.first_spike_idxs = list(first_spike_idxs)
        else:
            n_post = max((self.sources.max() + 1).item(), 1) if self.sources.size > 0 else 0
            fst = [None] * n_post
            fsi = [-1] * n_post
            for i in range(self._n_spikes):
                nrn = int(self.sources[i])
                if fst[nrn] is None or self.times[i] < fst[nrn]:
                    fst[nrn] = float(self.times[i])
                    fsi[nrn] = i
            self.first_spike_times = [t if t is not None else float('nan') for t in fst]
            self.first_spike_idxs = fsi

    @property
    def n_spikes(self) -> int:
        return self._n_spikes

    def set_error(self, spike_idx: int, error: float) -> None:
        self.errors[spike_idx] = error

    def set_time(self, spike_idx: int, time: float) -> None:
        self.times[spike_idx] = time

    def __len__(self) -> int:
        return self._n_spikes


# ── Maxima ──────────────────────────────────────────────────────────────────

class Maxima:
    """Container for membrane potential maxima per neuron."""

    __slots__ = ("times", "values", "errors")

    def __init__(self, times=None, values=None, errors=None):
        self.times = np.asarray(times, dtype=np.float64) if times is not None else np.array([], dtype=np.float64)
        self.values = np.asarray(values, dtype=np.float64) if values is not None else np.zeros_like(self.times, dtype=np.float64)
        self.errors = np.asarray(errors, dtype=np.float64) if errors is not None else np.zeros_like(self.times, dtype=np.float64)

    def set_error(self, nrn_idx: int, error: float) -> None:
        self.errors[nrn_idx] = error

    def set_value(self, nrn_idx: int, value: float) -> None:
        self.values[nrn_idx] = value


# ── SpikesVector / MaximaVector ─────────────────────────────────────────────

class SpikesVector(list):
    """Batch of Spikes — behaves like a list for indexing, slicing, len."""
    pass


class MaximaVector(list):
    """Batch of Maxima — behaves like a list for indexing, slicing, len."""
    pass
