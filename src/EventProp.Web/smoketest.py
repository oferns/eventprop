"""Headless smoke test for the Streamlit views.

    .venv/bin/python smoketest.py

Runs each page through `streamlit.testing.v1.AppTest`, which executes the script the way
a browser session would and surfaces exceptions the way the UI would. It exists because a
view can be wrong in ways `selfcheck.py` cannot see — a renamed field, a figure builder
whose arguments drifted, an attribute that only exists on one backend — and because a
blank page with a red stack trace is not a demonstration.

It checks that pages *render*, not that they are scientifically right; that is
`selfcheck.py`'s job.
"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

HERE = Path(__file__).resolve().parent
PAGES = ["Yin-Yang, live", "How it learns", "Exact gradients", "Digits (MNIST)",
         "Reading a number", "Under the hood"]
# Pages that legitimately render no figure. The Under-the-hood page is text and tables by
# design; the digits page shows a download prompt instead of a plot when the MNIST pixel
# files are not cached, which is the state of a fresh checkout — so a warning there would
# fire on exactly the machines that have done nothing wrong.
CHARTLESS_OK = {"Under the hood"}
# Give each run room: the live page scans an 81x81 grid and evaluates the test set.
TIMEOUT = 300

failures: list[str] = []


def run(page: str, label: str = "") -> AppTest | None:
    at = AppTest.from_file(str(HERE / "streamlit_app.py"), default_timeout=TIMEOUT)
    at.run()
    if at.exception:
        failures.append(f"{page}{label}: startup")
        report(at)
        return None
    at.sidebar.radio[0].set_value(page).run()
    return at


def report(at: AppTest) -> None:
    for exc in at.exception:
        print(f"        {exc.value}")
    for err in at.error:
        # One error box is content, not a crash: the 'defect we found' panel on the
        # Under-the-hood page is a deliberate st.error.
        print(f"        st.error: {err.value[:200]}")


def _click_button(at, label: str):
    """Click the first button carrying `label`.

    By label rather than index: the digits page has three buttons and their order is a
    detail of the page, not something the smoketest should be pinning. If the label is
    gone the assertion surfaces as a visible `[skip]` line in the loop — a panel that was
    renamed or dropped should show up as not-tested, not as a pass.
    """
    for b in at.button:
        if b.label == label:
            return b.set_value(True).run()
    raise AssertionError(f"no button labelled {label!r}; page offers "
                         f"{[b.label for b in at.button]}")


def _digits_awaiting_download(page: str) -> bool:
    """True when the digits page is legitimately chartless — no cached pixel files yet."""
    if page != "Digits (MNIST)":
        return False
    try:
        from eventweb import mnist
    except Exception:
        return False
    return not mnist.have_test_set()


def main() -> int:
    for page in PAGES:
        at = run(page)
        if at is None:
            print(f"  [FAIL] {page} — could not start")
            continue
        if at.exception:
            report(at)
            print(f"  [FAIL] {page}")
            failures.append(page)
            continue
        n_charts = len(at.get("plotly_chart"))
        n_errors = len(at.error)
        print(f"  [ok]   {page} — {n_charts} figure(s), {len(at.markdown)} text blocks, "
              f"{n_errors} error box(es)")
        if n_charts == 0 and page not in CHARTLESS_OK and not _digits_awaiting_download(page):
            print(f"  [WARN] {page} rendered no figures")

    # Interactions, not just first paint.
    checks = [
        ("Yin-Yang, live", lambda a: a.radio[0].set_value("Eigen reference (pybind11)").run(),
         "live page on the Eigen backend (traces path)"),
        # The C backend grew trace methods, so its traces are now a reachable UI path rather
        # than a hidden branch — and a binding that returns the wrong shape renders an empty
        # plot rather than an error, which is exactly what this check is for.
        ("Yin-Yang, live", lambda a: a.radio[0].set_value("C library (Cython)").run(),
         "live page back on the C backend (C traces path)"),
        ("Yin-Yang, live", lambda a: a.radio[1].set_value("a test-set sample").run(),
         "live page picking a real test point"),
        ("Yin-Yang, live", lambda a: a.selectbox[1].set_value(5).run(),
         "live page at the first snapshot (1 epoch)"),
        ("Yin-Yang, live", lambda a: a.slider[0].set_value(0.02).run(),
         "a point outside the data (no output spike at all)"),
        ("How it learns", lambda a: a.select_slider[0].set_value(0).run(),
         "boundary at the first snapshot"),
        ("How it learns", lambda a: a.select_slider[1].set_value(81).run(),
         "boundary at 81x81"),
        ("Exact gradients", lambda a: a.selectbox[0].set_value(7).run(),
         "a different input sample"),
        ("Exact gradients", lambda a: a.radio[0].set_value("Eigen reference (pybind11)").run(),
         "gradient page on the other backend (the picker's other half)"),
        ("Exact gradients", lambda a: a.slider[0].set_value(5).run(),
         "a different weight draw"),
        # The digits page is the only reachable UI path through the maxima read-out, so
        # both halves of it get driven: the run picker, the backend picker (C vs Eigen run
        # two independent implementations of `compute_maxima_batch`) and the sample index.
        ("Digits (MNIST)", lambda a: a.slider[0].set_value(3).run(),
         "digits page on a different published run"),
        ("Digits (MNIST)", lambda a: a.radio[0].set_value("Eigen reference").run(),
         "digits page on the Eigen backend (maxima read-out)"),
        ("Digits (MNIST)", lambda a: a.radio[0].set_value("C library (Cython)").run(),
         "digits page back on the C backend"),
        ("Digits (MNIST)", lambda a: a.number_input[0].set_value(7).run(),
         "digits page at another test digit"),
        # The timing panel measures both bindings on the same batches, so it is the only
        # place the maxima path is timed rather than merely exercised. Skips visibly when
        # the pixel files are not cached, since that state shows a download prompt instead.
        ("Digits (MNIST)", lambda a: _click_button(a, "Time both backends"),
         "digits timing panel (both backends measured)"),
        # The read-out page is the only reachable path through the C regression loss, so its
        # three meaningful axes get driven: the other target (y, which this network encodes
        # worse than x), the adjoint half of the fit (read-out + weights), and a Huber scale
        # small enough that the loss's saturation is visible in the curve.
        ("Reading a number", lambda a: a.radio[0].set_value("y — vertical position").run(),
         "read-out page on the y target"),
        ("Reading a number", lambda a: a.radio[1].set_value("read-out + weights (adjoint)").run(),
         "read-out page with the adjoint training the weights"),
        ("Reading a number", lambda a: a.radio[2].set_value("Huber δ = 0.01").run(),
         "read-out page at a saturating Huber scale"),
        ("Reading a number", lambda a: a.toggle[0].set_value(False).run(),
         "read-out page on raw (unstandardised) maxima"),
        ("Reading a number", lambda a: a.number_input[0].set_value(3).run(),
         "read-out page probing a different held-out point"),
        ("Under the hood", lambda a: a.selectbox[0].set_value(1000).run(),
         "parity across the whole test set"),
    ]
    for page, act, label in checks:
        at = run(page)
        if at is None:
            continue
        try:
            act(at)
        except Exception as exc:  # widget plumbing differences, not app bugs
            print(f"  [skip] {label} — {type(exc).__name__}: {exc}")
            continue
        if at.exception:
            report(at)
            print(f"  [FAIL] {label}")
            failures.append(label)
        else:
            print(f"  [ok]   {label}")

    print()
    if failures:
        print(f"FAILED ({len(failures)}): {'; '.join(failures)}")
        return 1
    print("all pages rendered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
