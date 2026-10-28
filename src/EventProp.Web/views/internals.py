"""View 4 — under the hood.

The previous version of this app led with backend comparison and live retraining, which
made it a developer tool wearing a demo's clothes. The comparison is genuinely useful, so
it stays — but demoted to a page of its own, made honest about what it is and is not.

Nothing here is a benchmark. The measured benchmarks live in `bench-reports/` with
provenance (`vcs_rev`, `cpu`, `tool_version`, dirty flags) and repeat statistics, and
`merge.py` refuses to combine them across hosts. The timings below are a single median
taken in a browser process, which makes them an existence proof, not a result.
"""

from __future__ import annotations

import sys

import streamlit as st

import state
from eventweb import figures as F
from eventweb import runtime


def render() -> None:
    st.header("Under the hood")

    st.subheader("Runtime")
    tag = runtime.abi_tag()
    found = runtime.find_extensions()
    r1, r2 = st.columns(2, gap="large")
    with r1:
        st.markdown(
            f"""
| | |
|---|---|
| Interpreter | `{sys.version.split()[0]}` |
| Executable | `{sys.executable}` |
| Extension ABI tag required | `{tag}` |
| Repository root | `{runtime.ROOT}` |
"""
        )
    with r2:
        rows = "\n".join(
            f"| {name} | {', '.join(f'`{f}`' for f in files) or '—'} |"
            for name, files in found.items()
        )
        st.markdown(f"| backend | built extension |\n|---|---|\n{rows}")
    st.caption("Extension filenames carry their ABI tag, so an extension built for another "
               "interpreter is invisible rather than merely stale. `eventweb/runtime.py` "
               "checks this at startup and refuses to start with a readable message; "
               "`build.sh` asks the same question before it decides whether to rebuild.")

    st.divider()
    st.subheader("What each backend can do from Python")
    be_c, be_e = state.backend("c"), state.backend("eigen")
    cap_rows = "\n".join(
        f"| {label} | {yes(be_c)} | {yes(be_e)} |"
        for label, probe in (
            ("forward pass (spike times)", lambda b: True),
            ("backward pass (gradients)", lambda b: True),
            ("membrane voltage trace", lambda b: b.supports_traces()),
            ("adjoint λ(t) trace", lambda b: b.supports_traces()),
            # The maxima pass is the other read-out the library offers, and the only one
            # the digits page can use: it is reached through `li_layer`, a module neither
            # page used before, so it is probed on the same basis as the traces.
            ("voltage maxima (VMax read-out)", lambda b: b.supports_maxima()),
        )
    )
    st.markdown(f"| capability | C library (Cython) | Eigen reference (pybind11) |\n"
                f"|---|:---:|:---:|\n{cap_rows}")
    st.markdown(
        "Both traces come from both bindings, and `selfcheck.py` compares them: voltage "
        "agrees to machine precision, the adjoint to ~1e-13 on a shipped network. The rows "
        "are still a *probe* rather than a constant because a declared `extern` is not a "
        "reachable Python function — the C library exported both traces long before the "
        "Cython layer wrapped either, and a table that read the header would have said "
        "\"yes\" while the app had nothing to show.\n\n"
        "The adjoint row has a history worth knowing. The C voltage trace has had a "
        "C-vs-Eigen parity test from the start; the λ trace had a test with that name which "
        "never called the Eigen side. The C implementation predated the correct adjoint jump "
        "rule in the backward pass and computed a different, loss-independent quantity — "
        "plausible-looking, and green. It now follows the reference "
        "`1/(i - v_th) · (v_th · λ_v + error)`, and the parity test is known to bite: it "
        "fails if the error term is removed."
    )

    st.divider()
    st.subheader("Do the two implementations agree?")
    a1, a2, a3 = st.columns(3)
    n = a1.selectbox("samples", [100, 300, 1000], index=1)
    agree = state.backend_agreement(0, 10, "test", n)
    a2.metric("Identical predictions", f"{agree['agree']:.2%}",
              f"{agree['disagreements']} disagreement(s) of {agree['n']}")
    a3.metric("Same weights, same spikes", "yes" if agree["disagreements"] == 0 else "no",
              "first-spike read-out, seed 0", delta_color="off")
    st.caption("Two independent implementations of the same control theory — one Eigen, one "
               "C — fed byte-identical inputs. `selfcheck.py` runs this on all 1000 test "
               "samples and additionally checks the C library against the accuracy the "
               "reference run recorded for those weights.")

    st.divider()
    st.subheader("Wall clock, roughly")
    t = state.timings("test")
    rows = [(state.backend(k).label, t[k]["forward_ms"], t[k]["backward_ms"]) for k in ("c", "eigen")]
    b1, b2 = st.columns([2, 1], gap="large")
    with b1:
        st.plotly_chart(F.timing_bars(rows, "Median per-batch time"), width="stretch")
    with b2:
        st.markdown(
            f"| | |\n|---|---|\n"
            f"| batch | {t['c']['batch']} samples |\n"
            f"| batches timed | {t['c']['n_batches']} |\n"
            f"| network | 5 → 200 → 3 |\n"
            f"| input | real test-set spikes |"
        )
    st.warning(
        "These are single medians taken inside a Streamlit handler. They are not "
        "comparable with `bench-reports/report_*.md`, which measure pre-built kernels "
        "with provenance and repeat runs. Quote the committed reports, not this chart — "
        "and per `BENCHMARK_SCHEMA.md`, compare per-pair geomeans across hosts, never the "
        "µs columns.",
        icon="⚠️",
    )

    st.divider()
    st.subheader("One defect this page found")
    st.error(
        "**`compute_lambda_i` reads out of bounds for a neuron that never spiked.**\n\n"
        "`post_spikes.times[post_spikes.n_spikes - 1]` is evaluated with no check, so with "
        "`n_spikes == 0` it reads `times[-1]`. In this checkout that is a reproducible "
        "SIGSEGV through `get_lambda_i_trace_for_neuron`, not a wrong number.\n\n"
        "It is present in the reference core (`src/EventProp.Python/src/eventprop_cpp.cpp`)"
        " and inherited by our copy "
        "(`src/EventProp.Cpp.Wrapper/eventprop_eigen_core_stripped.cpp`). The C library is "
        "the one that got this right: `eventprop_diag.c` special-cases zero spikes and "
        "returns an all-zero trace with `EVENTPROP_OK`, which is also the mathematically "
        "correct answer — no postsynaptic spikes means no adjoint source.\n\n"
        "This app never requests λ for a silent neuron (see `eventweb/analysis.py`), so the "
        "page does not crash. The defect is not fixed by that guard.",
        icon="🐞",
    )

    st.divider()
    st.subheader("What this app reads, and who owns it")
    st.markdown(
        f"""
| path | role |
|---|---|
| `{state.DATA_DIR.relative_to(runtime.ROOT)}` | frozen train/valid/test split — a nested submodule, published dataset, regenerate at your peril |
| `{state.CKPT_DIR.relative_to(runtime.ROOT)}/yinyang_{{0..9}}.pkl` | 10 published training runs: accuracy, loss, per-epoch first-spike times, 11 weight snapshots each |
| `{(runtime.ROOT / 'src/EventProp.C').relative_to(runtime.ROOT)}` | the C library, via the Cython binding |
| `{(runtime.ROOT / 'src/EventProp.Python').relative_to(runtime.ROOT)}` | read-only reference library, via pybind11 |

The reference library and its nested Yin-Yang submodule are read-only: local patches to
them are owned by `scripts/submodule-state.sh`, not by this app.
"""
    )


def yes(ok: bool) -> str:
    return "✅" if ok else "—"
