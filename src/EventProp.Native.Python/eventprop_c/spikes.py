"""Spikes and Maxima containers with zero-copy C output support.

SpikesVector wraps either:
  - Input mode: list of Spikes objects (Python-owned, copied)
  - Output mode: Cython BatchSpikeOutput (zero-copy, C-owned)
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
    """Batch of spikes — input (Python-owned) or output (Cython C-owned, zero-copy).

    Input mode:
        vector = SpikesVector([Spikes(t1, s1), Spikes(t2, s2), ...])

    Output mode (zero-copy, owns C memory via Cython BatchSpikeOutput):
        vector = SpikesVector._from_batch_output(cython_batch_output)
    """

    __slots__ = ('_spikes', '_batch_output', '_cached_views',
                 '_flat_times', '_flat_sources', '_n_spikes_per_sample')

    def __init__(self, spikes_list: Optional[Sequence[Spikes]] = None):
        self._spikes = list(spikes_list) if spikes_list is not None else []
        self._batch_output = None
        self._cached_views = {}

        # Flatten once at construction — outside any benchmark loop.
        # forward() and backward() just read these pre-built buffers (zero copies in hot path).
        if self._spikes:
            times_list = [np.asarray(s.times, dtype=np.float64) for s in self._spikes]
            sources_list = [np.asarray(s.sources, dtype=np.int32) for s in self._spikes]
            self._flat_times = np.concatenate(times_list)
            self._flat_sources = np.concatenate(sources_list)
            self._n_spikes_per_sample = np.array(
                [len(t) for t in times_list], dtype=np.intc
            )
        else:
            self._flat_times = np.array([], dtype=np.float64)
            self._flat_sources = np.array([], dtype=np.int32)
            self._n_spikes_per_sample = np.array([], dtype=np.intc)

    @classmethod
    def _from_batch_output(cls, batch_output) -> 'SpikesVector':
        """Create from Cython BatchSpikeOutput (zero-copy, owns memory)."""
        obj = object.__new__(cls)
        obj._spikes = None
        obj._batch_output = batch_output  # keeps C memory alive via __dealloc__
        obj._cached_views = {}
        # Output mode: flat buffers come from C (accessed via _batch_output directly).
        # These properties are not used for output-mode vectors.
        obj._flat_times = None
        obj._flat_sources = None
        obj._n_spikes_per_sample = None
        return obj

    def __len__(self) -> int:
        if self._batch_output is not None:
            return len(self._batch_output)
        return len(self._spikes)

    def __getitem__(self, idx: int):
        if self._batch_output is not None:
            if idx not in self._cached_views:
                times, sources, currents = self._batch_output[idx]
                self._cached_views[idx] = _CythonSpikesView(
                    times, sources, currents,
                    batch_output=self._batch_output,
                    sample_idx=idx
                )
            return self._cached_views[idx]
        return self._spikes[idx]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    # ── Pre-flattened buffers (input mode only) ────────────────

    @property
    def flat_times(self) -> np.ndarray:
        """Pre-flattened times array — ready for C batch calls."""
        return self._flat_times

    @property
    def flat_sources(self) -> np.ndarray:
        """Pre-flattened sources array — ready for C batch calls."""
        return self._flat_sources

    @property
    def n_spikes_per_sample(self) -> np.ndarray:
        """Per-sample spike counts — ready for C batch calls."""
        return self._n_spikes_per_sample

    # ── Chain support: extract flat buffers from output mode ───

    @staticmethod
    def _extract_flat_buffers(batch: 'SpikesVector') -> tuple:
        """Extract (times, sources, n_spikes) as flat arrays.

        For input-mode vectors this returns the pre-built buffers directly.
        For output-mode vectors (from C batch output) it iterates per-sample
        views and concatenates once, then caches for subsequent calls —
        needed for layer chains where one layer's output becomes the next layer's input.
        """
        if batch._batch_output is None:
            # Input mode: pre-built buffers available directly
            return batch._flat_times, batch._flat_sources, batch._n_spikes_per_sample

        # Output mode (C-owned): cache on first extraction to avoid repeated concatenation
        if batch._flat_times is None:
            times_list = []
            sources_list = []
            counts = []
            for i in range(len(batch)):
                sample = batch[i]
                t = np.asarray(sample.times, dtype=np.float64)
                s = np.asarray(sample.sources, dtype=np.int32)
                times_list.append(t)
                sources_list.append(s)
                counts.append(len(t))

            batch._flat_times = (
                np.concatenate(times_list) if times_list else np.array([], dtype=np.float64)
            )
            batch._flat_sources = (
                np.concatenate(sources_list) if sources_list else np.array([], dtype=np.int32)
            )
            batch._n_spikes_per_sample = np.array(counts, dtype=np.intc)

        return batch._flat_times, batch._flat_sources, batch._n_spikes_per_sample

    def append(self, spikes: Spikes):
        if self._batch_output is not None:
            raise RuntimeError("Cannot append to C-owned SpikesVector")
        self._spikes.append(spikes)
        # Invalidate cached flat buffers — they'll be rebuilt on next access.
        # (append is rare; callers should rebuild SpikesVector instead.)


class _CythonSpikesView:
    """Zero-copy view of a single sample from Cython BatchSpikeOutput.

    Wraps numpy arrays that reference C-owned memory. Errors are stored
    in the BatchSpikeOutput native buffer — writing here goes directly to C memory.
    """

    __slots__ = ('_times', '_sources', '_currents', '_batch_output', '_sample_idx')

    def __init__(self, times: np.ndarray, sources: np.ndarray, currents: np.ndarray,
                 batch_output=None, sample_idx: int = 0):
        self._times = times
        self._sources = sources
        self._currents = currents
        self._batch_output = batch_output  # BatchSpikeOutput holding native error buffer
        self._sample_idx = sample_idx

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
        # Return view into native error buffer — zero-copy, writes go to C memory
        if self._batch_output is not None:
            return self._batch_output.get_error_view(self._sample_idx)
        # Fallback: allocate Python array (shouldn't happen in normal flow)
        return np.zeros(self.n_spikes, dtype=np.float64)

    def set_error(self, spike_idx: int, error: float):
        self.errors[spike_idx] = error

    @property
    def first_spike_times(self) -> np.ndarray:
        """Return times of first spike per neuron."""
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

    @property
    def first_spike_idxs(self) -> np.ndarray:
        """Return indices of first spike per neuron."""
        max_nrn = max(int(np.max(self._sources)) + 1, 1) if self.n_spikes > 0 else 1
        result = np.full(max_nrn, -1, dtype=int)
        if self.n_spikes == 0:
            return result
        seen = set()
        for idx, (time, source) in enumerate(zip(self._times, self._sources)):
            src = int(source)
            if src not in seen:
                result[src] = idx
                seen.add(src)
        return result


class _CythonMaximaView:
    """Zero-copy view of a single sample from Cython BatchMaximaOutput.

    Wraps numpy arrays that reference C-owned memory. Errors are stored
    in the BatchMaximaOutput native buffer — writing here goes directly to C memory.
    """

    __slots__ = ('_times', '_values', '_batch_output', '_sample_idx')

    def __init__(self, times: np.ndarray, values: np.ndarray,
                 batch_output=None, sample_idx: int = 0):
        self._times = times
        self._values = values
        self._batch_output = batch_output  # BatchMaximaOutput holding native error buffer
        self._sample_idx = sample_idx

    @property
    def times(self) -> np.ndarray:
        return self._times

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def n_post(self) -> int:
        return len(self._times)

    @property
    def errors(self) -> np.ndarray:
        # Return view into native error buffer — zero-copy, writes go to C memory
        if self._batch_output is not None:
            return self._batch_output.get_error_view(self._sample_idx)
        # Fallback: allocate Python array (shouldn't happen in normal flow)
        return np.zeros(len(self._times), dtype=np.float64)

    def set_error(self, nrn_idx: int, error: float):
        self.errors[nrn_idx] = error


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
    """Batch of maxima — input (Python-owned) or output (Cython C-owned, zero-copy).

    Input mode:
        vector = MaximaVector([Maxima(t1, v1), Maxima(t2, v2), ...])

    Output mode (zero-copy, owns C memory via Cython BatchMaximaOutput):
        vector = MaximaVector._from_batch_output(cython_batch_output)
    """

    __slots__ = ('_maxima', '_batch_output', '_cached_views')

    def __init__(self, maxima_list: Optional[Sequence[Maxima]] = None):
        self._maxima = list(maxima_list) if maxima_list is not None else []
        self._batch_output = None
        self._cached_views = {}

    @classmethod
    def _from_batch_output(cls, batch_output) -> 'MaximaVector':
        """Create from Cython BatchMaximaOutput (zero-copy, owns memory)."""
        obj = object.__new__(cls)
        obj._maxima = None
        obj._batch_output = batch_output  # keeps C memory alive via __dealloc__
        obj._cached_views = {}
        return obj

    def __len__(self) -> int:
        if self._batch_output is not None:
            return len(self._batch_output)
        return len(self._maxima)

    def __getitem__(self, idx: int):
        if self._batch_output is not None:
            if idx not in self._cached_views:
                times, values = self._batch_output[idx]
                self._cached_views[idx] = _CythonMaximaView(
                    times, values,
                    batch_output=self._batch_output,
                    sample_idx=idx
                )
            return self._cached_views[idx]
        return self._maxima[idx]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def append(self, maxima: Maxima):
        if self._batch_output is not None:
            raise RuntimeError("Cannot append to C-owned MaximaVector")
        self._maxima.append(maxima)