"""Python data classes wrapping C library output structs.

Provides zero-copy access to C-allocated memory while matching the pybind11 interface.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Sequence


class Spikes:
    """Spiking neuron data with zero-copy access to C library memory.

    Can be constructed from numpy arrays (for input) or wrapped around
    C SpikeOutput structs (from compute_spikes output).

    Example:
        # Construct from numpy arrays (input)
        spikes = Spikes(times, sources)

        # Wrap C output (zero-copy)
        c_output = lib.compute_spikes(...)
        spikes = Spikes.from_c_output(c_output)

        # Access data (returns views, no copy)
        for t in spikes.times:
            print(t)

        # Modify in-place
        spikes.set_error(0, 1.0)
    """

    def __init__(self, times: Optional[np.ndarray] = None,
                 sources: Optional[np.ndarray] = None):
        """Construct Spikes from numpy arrays.

        Arrays are converted to contiguous C-order format if needed.
        """
        self._times = np.asarray(times, dtype=np.float64).copy() if times is not None else np.array([], dtype=np.float64)
        self._sources = np.asarray(sources, dtype=np.int32).copy() if sources is not None else np.array([], dtype=np.int32)
        self._errors = None
        
        # First spike per neuron (computed lazily)
        self._first_spike_times = None
        self._first_spike_idxs = None
        self._max_nrn_idx = None
    
    @property
    def first_spike_times(self) -> np.ndarray:
        """Return times of first spike per neuron.
        
        Computes and caches the time of each source neuron's first spike.
        Returns NaN for neurons with no spikes.
        """
        if self._first_spike_times is not None:
            return self._first_spike_times
        
        # Find max neuron index to determine size (at least 1)
        self._max_nrn_idx = max(int(np.max(self._sources)) + 1, 1) if self.n_spikes > 0 else 1
        
        # Initialize with NaN (no spikes)
        self._first_spike_times = np.full(self._max_nrn_idx, np.nan)
        self._first_spike_idxs = np.zeros(self._max_nrn_idx, dtype=np.int32)
        
        if self.n_spikes == 0:
            return self._first_spike_times
        
        # Track first spike per neuron
        seen = set()
        for idx, (time, source) in enumerate(zip(self._times, self._sources)):
            if int(source) not in seen:
                self._first_spike_times[int(source)] = time
                self._first_spike_idxs[int(source)] = idx
                seen.add(int(source))
        
        return self._first_spike_times
    
    @property
    def first_spike_idxs(self) -> np.ndarray:
        """Return indices of first spike per neuron.
        
        Computes and caches the index in times/sources arrays of each
        source neuron's first spike. Returns 0 for neurons with no spikes.
        """
        # Access first_spike_times to trigger computation
        _ = self.first_spike_times
        return self._first_spike_idxs

    @classmethod
    def from_c_output(cls, c_struct) -> 'Spikes':
        """Create Spikes object wrapping C SpikeOutput memory (zero-copy).

        Uses numpy's ctypes utilities to view C-allocated arrays directly.
        """
        obj = object.__new__(cls)
        n_spikes = c_struct.n_spikes

        if n_spikes > 0:
            obj._times = np.ctypeslib.as_array(c_struct.times, shape=(n_spikes,))
            obj._sources = np.ctypeslib.as_array(c_struct.sources, shape=(n_spikes,))
        else:
            obj._times = np.array([], dtype=np.float64)
            obj._sources = np.array([], dtype=np.int32)

        obj._errors = None
        obj._first_spike_times = None
        obj._first_spike_idxs = None
        obj._max_nrn_idx = None
        return obj

    @property
    def times(self) -> np.ndarray:
        """Return view of spike times (no copy)."""
        return self._times

    @property
    def sources(self) -> np.ndarray:
        """Return view of source neuron indices (no copy)."""
        return self._sources

    @property
    def n_spikes(self) -> int:
        """Number of spikes."""
        return len(self._times)

    @property
    def errors(self) -> np.ndarray:
        """Return error array (lazy initialization on first access)."""
        if self._errors is None:
            self._errors = np.zeros(self.n_spikes, dtype=np.float64)
        return self._errors

    def set_error(self, spike_idx: int, error: float):
        """Set error value for a specific spike (in-place)."""
        if self._errors is None:
            self._errors = np.zeros(self.n_spikes, dtype=np.float64)
        self._errors[spike_idx] = error

    def set_time(self, spike_idx: int, time: float):
        """Set time for a specific spike (in-place)."""
        self._times[spike_idx] = time


class SpikesVector:
    """List of Spikes objects with vector-like interface.

    Matches the pybind11 std::vector<Spikes> interface.
    """

    def __init__(self, spikes_list: Optional[Sequence[Spikes]] = None):
        self._spikes = list(spikes_list) if spikes_list is not None else []

    def __len__(self) -> int:
        return len(self._spikes)

    def __getitem__(self, idx: int) -> Spikes:
        return self._spikes[idx]

    def __iter__(self):
        yield from self._spikes

    def append(self, spikes: Spikes):
        """Append a Spikes object to the vector."""
        self._spikes.append(spikes)

    def __array__(self) -> np.ndarray:
        """Convert to numpy array of objects."""
        return np.array(self._spikes, dtype=object)


class Maxima:
    """Membrane potential maxima with zero-copy access to C library memory.

    Can be constructed from numpy arrays or wrapped around
    C MaximaOutput structs (from compute_maxima output).
    """

    def __init__(self, times: Optional[np.ndarray] = None,
                 values: Optional[np.ndarray] = None):
        """Construct Maxima from numpy arrays."""
        self._times = np.asarray(times, dtype=np.float64).copy() if times is not None else np.array([], dtype=np.float64)
        self._values = np.asarray(values, dtype=np.float64).copy() if values is not None else np.array([], dtype=np.float64)
        self._errors = None

    @classmethod
    def from_c_output(cls, c_struct) -> 'Maxima':
        """Create Maxima object wrapping C MaximaOutput memory (zero-copy)."""
        obj = object.__new__(cls)
        n_post = c_struct.n_post

        if n_post > 0:
            obj._times = np.ctypeslib.as_array(c_struct.times, shape=(n_post,))
            obj._values = np.ctypeslib.as_array(c_struct.values, shape=(n_post,))
        else:
            obj._times = np.array([], dtype=np.float64)
            obj._values = np.array([], dtype=np.float64)

        obj._errors = None
        return obj

    @property
    def times(self) -> np.ndarray:
        """Return view of maxima times (no copy)."""
        return self._times

    @property
    def values(self) -> np.ndarray:
        """Return view of maxima values (no copy)."""
        return self._values

    @property
    def errors(self) -> np.ndarray:
        """Return error array (lazy initialization on first access)."""
        if self._errors is None:
            self._errors = np.zeros(len(self._times), dtype=np.float64)
        return self._errors

    def set_error(self, nrn_idx: int, error: float):
        """Set error value for a specific neuron (in-place)."""
        if self._errors is None:
            self._errors = np.zeros(len(self._times), dtype=np.float64)
        self._errors[nrn_idx] = error

    def set_value(self, nrn_idx: int, value: float):
        """Set value for a specific neuron (in-place)."""
        self._values[nrn_idx] = value


class MaximaVector:
    """List of Maxima objects with vector-like interface.

    Matches the pybind11 std::vector<Maxima> interface.
    """

    def __init__(self, maxima_list: Optional[Sequence[Maxima]] = None):
        self._maxima = list(maxima_list) if maxima_list is not None else []

    def __len__(self) -> int:
        return len(self._maxima)

    def __getitem__(self, idx: int) -> Maxima:
        return self._maxima[idx]

    def __iter__(self):
        yield from self._maxima

    def append(self, maxima: Maxima):
        """Append a Maxima object to the vector."""
        self._maxima.append(maxima)

    def __array__(self) -> np.ndarray:
        """Convert to numpy array of objects."""
        return np.array(self._maxima, dtype=object)
