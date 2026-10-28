"""A thin registry over the two native backends.

Two independent implementations of the same maths ship in this repository:

* **C library** — `src/EventProp.C`, reached through the Cython layer in
  `src/EventProp.Native.Python`. This is the artefact the repository exists to produce.
* **Eigen reference** — the pybind11 core inside the read-only reference library
  `src/EventProp.Python`. This is the specification the C library must reproduce.

Both expose the same *shape* of API (`Spikes`, `SpikesVector`, `LIFLayer`,
`LIFLayerParameters`), but not the same *surface*: the reference has voltage and
adjoint-trace helpers that the Cython layer does not wrap, and the two bindings take
structurally different spike arguments on purpose (pybind11 per-sample 2D arrays,
Cython flat arrays). The wrappers below hide that difference for the views without
pretending the two are interchangeable — capabilities are *probed*, never asserted, so a
view cannot claim a feature a backend has since stopped providing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import runtime


@dataclass(frozen=True)
class Backend:
    """One native implementation, plus the adapters this app needs to drive it."""

    key: str
    label: str
    short: str
    color: str
    module_prefix: str
    spikes_module: str
    lif_module: str
    _modules: dict = field(default_factory=dict, repr=False, compare=False)

    def load(self) -> None:
        if self._modules:
            return
        runtime.add_backend_paths()
        from importlib import import_module

        self._modules["spikes"] = import_module(self.spikes_module)
        self._modules["lif"] = import_module(self.lif_module)

    @property
    def Spikes(self):
        self.load()
        return self._modules["spikes"].Spikes

    @property
    def SpikesVector(self):
        self.load()
        return self._modules["spikes"].SpikesVector

    @property
    def LIFLayer(self):
        self.load()
        return self._modules["lif"].LIFLayer

    @property
    def LIFLayerParameters(self):
        self.load()
        return self._modules["lif"].LIFLayerParameters

    def make_vector(self, times: np.ndarray, sources: np.ndarray):
        """Build a `SpikesVector` from per-sample, time-sorted `(n, k)` arrays."""
        return self.SpikesVector(
            [self.Spikes(t, s) for t, s in zip(times, sources)]
        )

    def params(self, n: int, n_in: int, **kw):
        return self.LIFLayerParameters(n=n, n_in=n_in, **kw)

    def layer(self, n: int, n_in: int, w_in: np.ndarray, **kw):
        return self.LIFLayer(self.params(n=n, n_in=n_in, **kw), w_in=np.ascontiguousarray(w_in))

    def supports_traces(self) -> bool:
        """Can this backend produce membrane-voltage / adjoint traces from Python?

        Probed, not asserted: the C library *does* export
        `eventprop_compute_voltage_trace` (see `src/EventProp.C/eventprop_diag.h`), and
        `_native.pyx` declares it — but the Cython layer currently wraps no Python-level
        entry point for it, so it is unreachable from here. Reporting it as available
        would be a claim about a header, not about the binding in use.
        """
        return hasattr(self.LIFLayer, "get_voltage_trace_for_neuron")


BACKENDS: dict[str, Backend] = {
    "c": Backend(
        key="c",
        label="C library (Cython binding)",
        short="C",
        color="#22c55e",
        module_prefix="eventprop_c",
        spikes_module="eventprop_c.spikes",
        lif_module="eventprop_c.lif_layer",
    ),
    "eigen": Backend(
        key="eigen",
        label="Eigen reference (pybind11 binding)",
        short="Eigen",
        color="#3b82f6",
        module_prefix="eventprop",
        spikes_module="eventprop.eventprop_cpp",
        lif_module="eventprop.lif_layer",
    ),
}


def get(key: str) -> Backend:
    return BACKENDS[key]


def available() -> list[str]:
    """Backend keys that import successfully in this interpreter."""
    ok = []
    for key, be in BACKENDS.items():
        try:
            be.load()
        except Exception:
            continue
        ok.append(key)
    return ok
