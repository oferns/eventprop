"""Spikes and Maxima containers with zero-copy C output support.

SpikesVector wraps either:
  - Input mode: list of Spikes objects (Python-owned, copied)
  - Output mode: C SpikeOutput* array via BatchSpikeOutput (zero-copy, C-owned)
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Sequence


class Spikes:
    """Spiking neuron data.

    Constructed from numpy arrays for input spikes, or as zero-copy views
    into C SpikeOutput memory via SpikesVector output mode.
    """

    __slots__ = ('_times', '_sources', '_errors')

    def __init__(self, times: Optional[np.ndarray] = None,
                 sources: Optional[np.ndarray] = None):
        self._times = np.asarray(times, dtype=np.float64).copy() if times is not None else np.array([], dtype=np.float64)
        self._sources = np.asarray(sources, dtype=np.int32).copy() if sources is not None else np.array([], dtype=np.int32)
        self._errors = None

    @property
    def times(self) -> np.ndarray:
        return self._times

    @property
    def sources(self) -> np.ndarray:
        return self._sources

    @property
    def n_spikes(self) -> int:
        return len(self._times)

    @property
    def errors(self) -> np.ndarray:
        if self._errors is None:
            self._errors = np.zeros(self.n_spikes, dtype=np.float64)
        return self._errors

    def set_error(self, spike_idx: int, error: float):
        if self._errors is None:
            self._errors = np.zeros(self.n_spikes, dtype=np.float64)
        self._errors[spike_idx] = error

    def set_time(self, spike_idx: int, time: float):
        self._times[spike_idx] = time

    @property
    def first_spike_times(self) -> np.ndarray:
        """Return times of first spike per neuron (cached)."""
        max_nrn = max(int(np.max(self._sources)) + 1, 1) if self.n_spikes > 0 else 1
        result = np.full(max_nrn, np.nan)
        if self.n_spikes == 0:
            return result
        seen = set()
        for time, source in zip(self._times, self._sources):
            src = int(source)
            if src not in seen:
                result[src] = time
                seen.add(src)
        return result


class SpikesVector:
    """Batch of spikes — input (Python-owned) or output (C-owned, zero-copy).

    Input mode:
        vector = SpikesVector([Spikes(t1, s1), Spikes(t2, s2), ...])

    Output mode (zero-copy, owns C memory):
        vector = SpikesVector._from_batch_output(batch_c_ptr, batch_size)
    """

    __slots__ = ('_spikes', '_batch_output')

    def __init__(self, spikes_list: Optional[Sequence[Spikes]] = None):
        self._spikes = list(spikes_list) if spikes_list is not None else []
        self._batch_output = None

    @classmethod
    def _from_batch_output(cls, batch_output: 'BatchSpikeOutput') -> 'SpikesVector':
        """Create from C-owned BatchSpikeOutput (zero-copy, owns memory)."""
        obj = object.__new__(cls)
        obj._spikes = None
        obj._batch_output = batch_output  # keeps C memory alive
        return obj

    @classmethod
    def _from_dropout_output(cls, batch_output: 'DropoutBatchOutput') -> 'SpikesVector':
        """Create from C-owned DropoutBatchOutput (zero-copy, owns memory)."""
        obj = object.__new__(cls)
        obj._spikes = None
        obj._batch_output = batch_output  # keeps C memory alive
        return obj

    def __len__(self) -> int:
        if self._batch_output is not None:
            return len(self._batch_output)
        return len(self._spikes)

    def __getitem__(self, idx: int):
        if self._batch_output is not None:
            # Return zero-copy view of sample idx from C memory
            return self._batch_output[idx]
        return self._spikes[idx]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def append(self, spikes: Spikes):
        if self._batch_output is not None:
            raise RuntimeError("Cannot append to C-owned SpikesVector")
        self._spikes.append(spikes)

    @property
    def errors(self, sample_idx: int = None):
        """Get/set error arrays. For output mode, stores errors on the batch."""
        # Errors are managed per-sample via Spikes.errors or directly
        raise AttributeError("Use vector[i].errors for per-sample access")


class Maxima:
    """Membrane potential maxima."""

    __slots__ = ('_times', '_values', '_errors')

    def __init__(self, times: Optional[np.ndarray] = None,
                 values: Optional[np.ndarray] = None):
        self._times = np.asarray(times, dtype=np.float64).copy() if times is not None else np.array([], dtype=np.float64)
        self._values = np.asarray(values, dtype=np.float64).copy() if values is not None else np.array([], dtype=np.float64)
        self._errors = None

    @property
    def times(self) -> np.ndarray:
        return self._times

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def errors(self) -> np.ndarray:
        if self._errors is None:
            self._errors = np.zeros(len(self._times), dtype=np.float64)
        return self._errors

    def set_error(self, nrn_idx: int, error: float):
        if self._errors is None:
            self._errors = np.zeros(len(self._times), dtype=np.float64)
        self._errors[nrn_idx] = error

    def set_value(self, nrn_idx: int, value: float):
        self._values[nrn_idx] = value


class MaximaVector:
    """List of Maxima objects."""

    def __init__(self, maxima_list: Optional[Sequence[Maxima]] = None):
        self._maxima = list(maxima_list) if maxima_list is not None else []

    def __len__(self) -> int:
        return len(self._maxima)

    def __getitem__(self, idx: int) -> Maxima:
        return self._maxima[idx]

    def __iter__(self):
        yield from self._maxima

    def append(self, maxima: Maxima):
        self._maxima.append(maxima)
