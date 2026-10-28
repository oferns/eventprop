"""Headless self-check for the Visualizer. Run this before blaming a view.

    .venv/bin/python selfcheck.py

It verifies the things the UI cannot see and would otherwise display confidently:

1. the interpreter can load BOTH pre-built native backends;
2. the frozen Yin-Yang split still matches its own geometry (catches a swapped/corrupted
   `.npy`, or a rewrite of `which_class` that drifts from the submodule);
3. running the shipped weights through the C library reproduces the accuracy the
   reference library recorded for those same weights — the real test of the encoder, the
   TTFS read-out and the weight plumbing end to end;
4. the weight snapshots line up with `SNAPSHOT_EPOCHS`, i.e. with the trainer's
   pre-epoch evaluation / post-epoch save asymmetry;
5. both backends agree sample-by-sample.

Exit code is non-zero on any failure, with the failing check named.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eventweb import runtime  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


TRACE_DT = 1e-3


def _set_post_errors(spikes, values: list[float]) -> None:
    """Write the post-spike error signal through whichever setter this binding exposes.

    The two bindings expose this differently on purpose; the error arrays are what the
    backward pass reads, so writing them here means the adjoint is driven by the same
    signal a training step would use.
    """
    for j, val in enumerate(values):
        if hasattr(spikes, "set_error"):
            spikes.set_error(j, float(val))
        else:
            spikes.errors[j] = float(val)


def _trace_probe(net, split) -> tuple[np.ndarray, int] | None:
    """First test point whose output layer fires, with a traced neuron that spiked.

    A silent output neuron has an all-zero adjoint on every backend, so tracing one and
    calling the agreement a parity result would be comparing zeros.
    """
    for i in range(min(20, split.n)):
        coords_i = split.coords[i:i + 1]
        vec = net.spikes_for(coords_i)
        post = net.output.post_batch[0]
        sources = np.asarray(post.sources).ravel()
        if len(sources) == 0:
            continue
        return coords_i, int(sources[0])
    return None


def main() -> int:
    root = runtime.ROOT
    data_dir = root / "src/EventProp.Python/eventprop/yin_yang_data_set/publication_data"
    ckpt_dir = root / "src/EventProp.Python/paper"

    print("1. interpreter / native ABI")
    try:
        tag = runtime.ensure_native_abi()
        check("both backends loadable", True, f"ABI tag {tag!r}")
    except RuntimeError as exc:
        check("both backends loadable", False, "")
        print(str(exc))
        return 1

    from eventweb import analysis, dataset, network
    from eventweb.backends import BACKENDS, available

    print(f"2. backends import: {', '.join(available())}")
    check("both backends import", set(available()) == {"c", "eigen"})

    print("3. frozen dataset integrity")
    train = dataset.load_split(data_dir, "train")
    test = dataset.load_split(data_dir, "test")
    x, y = test.xy()
    check("train split is the published 5000", train.n == dataset.N_TRAIN, f"n={train.n}")
    check("test split size", test.n == dataset.N_TEST, f"n={test.n}")
    tx, ty = train.xy()
    check(
        "train labels agree with the yin-yang geometry",
        bool((dataset.which_class(tx, ty) == train.labels).all()),
    )
    check(
        "test labels agree with the yin-yang geometry",
        bool((dataset.which_class(x, y) == test.labels).all()),
    )
    check(
        "mirrored axis is exact (x2 = 1-x1, y2 = 1-y1)",
        bool(np.allclose(test.coords[:, 2:], 1.0 - test.coords[:, :2])),
    )
    counts = np.bincount(test.labels, minlength=3)
    check("three balanced classes", counts.min() > 200, f"counts={counts.tolist()}")

    print("4. shipped checkpoints vs the C library")
    be_c = BACKENDS["c"]
    ckpt = network.load_checkpoint(ckpt_dir, 0)
    stored = float(ckpt.accuracies[-1])

    # Decode the first-spike times the reference run itself stored: an independent path
    # to the same number that does not touch either binding.
    ref_pred, _ = dataset.decode(ckpt.first_spikes[-1])
    decoded = float((ref_pred == test.labels).mean())
    check("stored first-spike times decode to the stored accuracy", abs(decoded - stored) < 1e-9,
          f"decoded={decoded:.4f} stored={stored:.4f}")

    net = network.Net(be_c, *ckpt.weights_at(-1))
    ours = net.accuracy(test)
    check(
        "C library reproduces the reference accuracy",
        abs(ours - stored) < 0.005,
        f"C={ours:.4f} reference={stored:.4f} (delta {1e4 * abs(ours - stored):.1f} bp)",
    )

    print("5. snapshot/epoch alignment")
    for k in (0, 1, -1):
        epochs = ckpt.snapshot_epochs[k]
        wh, wo = ckpt.weights_at(k)
        acc = network.Net(be_c, wh, wo).accuracy(test)
        # accuracy[i] is measured after i epochs, so it is defined for i < 100 only;
        # the last snapshot (100 epochs) has no recorded accuracy and must be >= acc[-1].
        if epochs < len(ckpt.accuracies):
            ref = float(ckpt.accuracies[epochs])
            ok = abs(acc - ref) < 0.02
            detail = f"{epochs} epochs -> ours {acc:.3f} vs recorded {ref:.3f}"
        else:
            ok = acc >= stored - 0.02
            detail = f"{epochs} epochs -> ours {acc:.3f} (final, recorded {stored:.3f})"
        check(f"snapshot {k} is after {epochs} epochs", ok, detail)

    print("6. backend agreement")
    be_py = BACKENDS["eigen"]
    pred_c, _, _ = net.predict(test.coords)
    net_e = network.Net(be_py, *ckpt.weights_at(-1))
    pred_e = net_e.predict(test.coords)[0]
    agree = float((pred_c == pred_e).mean())
    check("C and Eigen predict identically on all 1000 test samples", agree == 1.0,
          f"agreement={agree:.3%}")

    print("7. exact gradient vs finite differences")
    # The headline claim of the 'Exact gradients' page. That page now has a backend picker,
    # so the claim is asserted on both bindings rather than only on the one that used to be
    # the only way to get traces. Several samples, because a single agreeable sample proves
    # little and a silent one proves the wrong thing.
    for be_key in ("c", "eigen"):
        short = BACKENDS[be_key].short
        worst = 0.0
        compared = 0
        skipped = 0
        for idx in (0, 42, 101, 555, 888):
            demo = analysis.gradient_demo(BACKENDS[be_key], test.coords[idx])
            if not demo.fired:
                check(f"{short}: sample {idx} fired", False,
                      "scenario produced no output spike")
                continue
            valid = demo.fd_valid.ravel()
            skipped += int((~valid).sum())
            picked = demo.rel_error.ravel()[valid & np.isfinite(demo.rel_error.ravel())]
            if picked.size:
                worst = max(worst, float(picked.max()))
                compared += int(picked.size)
        check(f"{short}: finite differences compared on every usable synapse", compared > 15,
              f"{compared} synapses compared, {skipped} skipped as discontinuous")
        check(f"{short}: exact gradient matches central differences", 0.0 <= worst < 1e-6,
              f"max relative error {worst:.2e}")

    print("8. trace parity (C vs Eigen)")
    # Traces are the visible half of the adjoint: the Gradient page plots lambda_i(t) as the
    # exact gradient's mechanism, so a backend that spikes identically and traces differently
    # would still mislead a reader. Errors are set on both sides because lambda_i is
    # loss-driven — zero errors make both traces zero and the comparison vacuous.
    probe = _trace_probe(net_e, test)
    if probe is None:
        check("found a probe point where the output layer fires", False,
              "no spiking sample in the first 20 test points")
    else:
        coords_i, target = probe
        got = {}
        for key in ("c", "eigen"):
            n = network.Net(BACKENDS[key], *ckpt.weights_at(-1))
            n.spikes_for(coords_i)
            post = n.output.post_batch[0]
            _set_post_errors(post, [1.0 + 0.5 * j for j in range(len(post.times))])
            _, v = n.hidden.get_voltage_trace_for_neuron(
                0, target, t_max=n.t_max, dt=TRACE_DT)
            _, lam = n.output.get_lambda_i_trace_for_neuron(
                0, target, t_max=n.t_max, dt=TRACE_DT)
            got[key] = (v, lam, n)
        vc, lc, net_probe = got["c"]
        ve, le, _ = got["eigen"]
        check("voltage traces match across backends",
              bool(np.allclose(vc, ve, rtol=1e-6, atol=1e-8)),
              f"max |delta| {np.abs(vc - ve).max():.2e}")
        check("adjoint traces match across backends",
              bool(np.allclose(lc, le, rtol=1e-6, atol=1e-8)),
              f"max |delta| {np.abs(lc - le).max():.2e}, peak |lambda| {np.abs(lc).max():.4g}")
        # The pointwise query and the grid share one sweep in C, so they should agree exactly.
        half = int(round((net_probe.t_max / 2) / TRACE_DT))
        one = net_probe.output.get_lambda_i_for_neuron(0, target, half * TRACE_DT)
        check("pointwise adjoint equals the trace it samples",
              one == lc[half], f"{one:.10g} vs {lc[half]:.10g} at t={half * TRACE_DT:.4f}")

    print("9. trace capability")
    for key, be in BACKENDS.items():
        supports = be.supports_traces()
        check(f"{be.short} backend exposes traces from Python", supports,
              "yes" if supports else "no — the views hide any trace panel for this backend")

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
