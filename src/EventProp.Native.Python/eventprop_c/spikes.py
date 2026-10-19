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

        # Skip flat buffer creation — eliminates Python-level list comprehension
        # and np.concatenate overhead. Flat buffers are created on-demand or
        # provided directly by the Python layer via _from_flat_arrays().
        self._flat_times = None
        self._flat_sources = None
        self._n_spikes_per_sample = None

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

    @classmethod
    def _from_dropout_output(cls, batch_output) -> 'SpikesVector':
        """Create from Cython DropoutBatchOutput (zero-copy, owns memory).

        Dropout output has no error buffers — just times and sources.
        """
        obj = object.__new__(cls)
        obj._spikes = None
        obj._batch_output = batch_output  # keeps C memory alive via __dealloc__
        obj._cached_views = {}
        obj._flat_times = None
        obj._flat_sources = None
        obj._n_spikes_per_sample = None
        return obj

    @classmethod
    def _from_flat_arrays(cls, flat_times: np.ndarray, flat_sources: np.ndarray,
                          n_spikes_per_sample: np.ndarray) -> 'SpikesVector':
        """Create from flat numpy arrays — zero Python overhead.
        
        This is the most efficient way to create a SpikesVector for the C batch API.
        No list comprehension or np.concatenate overhead.
        """
        obj = object.__new__(cls)
        obj._spikes = None
        obj._batch_output = None
        obj._cached_views = {}
        obj._flat_times = flat_times
        obj._flat_sources = flat_sources
        obj._n_spikes_per_sample = n_spikes_per_sample
        return obj

    def __len__(self) -> int:
        if self._batch_output is not None:
            return self._batch_output.__len__()
        return len(self._spikes)

    def __getitem__(self, idx: int):
        if self._batch_output is not None:
            if idx not in self._cached_views:
                # Both batch outputs hand back (times, sources): the current and
                # the derivative buffer are no longer part of the spike output.
                times, sources = self._batch_output.__getitem__(idx)
                self._cached_views[idx] = _CythonSpikesView(
                    times, sources,
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

    @staticmethod
    def _error_arrays(batch: 'SpikesVector') -> list:
        """One caller-owned derivative array per sample, in sample order.

        The C forward pass no longer carries these, so backward() collects them
        through the views — the arrays are allocated on first use and cached —
        and hands the list to the batch backward call, which reads it straight
        into C.
        """
        return [batch[i].errors for i in range(len(batch))]

    # ── Chain support: extract flat buffers from output mode ───

    @staticmethod
    def _extract_flat_buffers(batch: 'SpikesVector') -> tuple:
        """Extract (times, sources, n_spikes) as flat arrays.

        For input-mode vectors this returns the pre-built buffers directly,
        or creates them on-demand from the _spikes list.
        For output-mode vectors (from C batch output) it iterates per-sample
        views and concatenates once, then caches for subsequent calls —
        needed for layer chains where one layer's output becomes the next layer's input.
        """
        if batch._batch_output is None:
            # Input mode: check for pre-built buffers first
            if batch._flat_times is not None:
                return batch._flat_times, batch._flat_sources, batch._n_spikes_per_sample
            
            # Create flat buffers from _spikes list on-demand
            if batch._spikes:
                # Optimize for single-sample batch: avoid list comprehension and np.concatenate overhead
                if len(batch._spikes) == 1:
                    s = batch._spikes[0]
                    batch._flat_times = np.asarray(s.times, dtype=np.float64)
                    batch._flat_sources = np.asarray(s.sources, dtype=np.int32)
                    batch._n_spikes_per_sample = np.array([len(batch._flat_times)], dtype=np.intc)
                else:
                    times_list = [np.asarray(s.times, dtype=np.float64) for s in batch._spikes]
                    sources_list = [np.asarray(s.sources, dtype=np.int32) for s in batch._spikes]
                    batch._flat_times = np.concatenate(times_list)
                    batch._flat_sources = np.concatenate(sources_list)
                    batch._n_spikes_per_sample = np.array(
                        [len(t) for t in times_list], dtype=np.intc
                    )
            else:
                batch._flat_times = np.array([], dtype=np.float64)
                batch._flat_sources = np.array([], dtype=np.int32)
                batch._n_spikes_per_sample = np.array([], dtype=np.intc)
            
            return batch._flat_times, batch._flat_sources, batch._n_spikes_per_sample

        # Output mode (C-owned): cache on first extraction to avoid repeated concatenation
        if batch._flat_times is None:
            # Optimize for single-sample batch: avoid list comprehension and np.concatenate overhead
            if len(batch) == 1:
                sample = batch[0]
                batch._flat_times = np.asarray(sample.times, dtype=np.float64)
                batch._flat_sources = np.asarray(sample.sources, dtype=np.int32)
                batch._n_spikes_per_sample = np.array([len(batch._flat_times)], dtype=np.intc)
            else:
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

    Times and sources reference C-owned memory. The derivative buffer is not in
    C any more — backward() takes caller-owned arrays — so it is allocated here
    and kept alive by the view.
    """

    __slots__ = ('_times', '_sources', '_batch_output', '_sample_idx', '_errors_buf')

    def __init__(self, times: np.ndarray, sources: np.ndarray,
                 batch_output=None, sample_idx: int = 0):
        self._times = times
        self._sources = sources
        self._batch_output = batch_output  # BatchSpikeOutput holding C spike memory
        self._sample_idx = sample_idx
        self._errors_buf = None            # caller-owned, allocated on first use

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
        # Caller-owned buffer: allocated once per view, then handed to C by
        # backward_spikes_batch_accumulate.
        if self._errors_buf is None:
            self._errors_buf = np.zeros(self.n_spikes, dtype=np.float64)
        return self._errors_buf

    @property
    def _errors(self) -> np.ndarray:
        """Backward compatibility — layer backward() reads this directly."""
        return self.errors

    @_errors.setter
    def _errors(self, value: np.ndarray):
        """Backward compatibility — layer backward() sets this directly."""
        object.__setattr__(self, '_errors_buf', np.asarray(value, dtype=np.float64))

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

    __slots__ = ('_times', '_values', '_batch_output', '_sample_idx', '_errors_buf')

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
        # The C MaximaOutput owns the error buffer the backward pass reads.
        # Return a zero-copy view into it (via the batch output) so set_error /
        # vector writes land where C looks. The old code returned a detached
        # Python buffer, so loss-layer errors never reached the C backward.
        if self._batch_output is not None:
            return self._batch_output.get_maxima_errors(self._sample_idx)
        if not hasattr(self, '_errors_buf') or self._errors_buf is None:
            object.__setattr__(self, '_errors_buf', np.zeros(len(self._times), dtype=np.float64))
        return self._errors_buf

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
            return self._batch_output.__len__()
        return len(self._maxima)

    def __getitem__(self, idx: int):
        if self._batch_output is not None:
            if idx not in self._cached_views:
                times, values = self._batch_output.__getitem__(idx)
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