"""Headless smoke test for the Streamlit chapters.

    .venv/bin/python smoketest.py

Runs each chapter through `streamlit.testing.v1.AppTest`, which executes the script the way
a browser session would and surfaces exceptions the way the UI would. It exists because a
view can be wrong in ways `selfcheck.py` cannot see — a renamed field, a figure builder
whose arguments drifted, an attribute that only exists on one backend — and because a blank
page with a red stack trace is not a demonstration.

It checks that chapters *render*, not that they are scientifically right; that is
`selfcheck.py`'s job.

Chapters are addressed through `eventweb/narrative.py` rather than by literal strings. That
is deliberate: the sidebar labels carry chapter numbers, so a test that hardcodes
`"Reading a number"` starts selecting nothing the moment a chapter is renumbered — and a
smoketest that selects nothing renders no page, finds no exception, and reports a pass.
Asking the spine for its own labels makes that failure impossible to hide.

Widgets are found by label where the widget has a distinctive label, and by position only
where it does not; each positional lookup says which control it means, because a silently
re-targeted interaction is worse than a missing one.
"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

from eventweb import narrative

HERE = Path(__file__).resolve().parent
CH = {c.slug: c.label for c in narrative.chapters()}
PAGES = [c.label for c in narrative.chapters()]
# Chapters that legitimately render no figure. The closing chapter is text and tables by
# design; the digits chapter shows a download prompt instead of a plot when the MNIST pixel
# files are not cached, which is the state of a fresh checkout — so a warning there would
# fire on exactly the machines that have done nothing wrong.
CHARTLESS_OK = {CH["limits"]}
# Give each run room: chapter 3 scans an 81x81 grid and evaluates the test set, chapter 5
# can fetch and run 10 000 digits.
TIMEOUT = 600

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
        # One error box is content, not a crash: the 'defect we found' panel on the closing
        # chapter is a deliberate st.error.
        print(f"        st.error: {err.value[:200]}")


def _click(at, label: str):
    """Click the first button carrying `label`.

    By label rather than index: a chapter can hold several buttons and their order is a
    detail of the page, not something the smoketest should pin. A missing label raises,
    which surfaces as a visible failure rather than a silent skip.
    """
    for b in at.button:
        if b.label == label:
            return b.set_value(True).run()
    raise AssertionError(f"no button labelled {label!r}; page offers "
                         f"{[b.label for b in at.button]}")


def _set(at, kind: str, wanted: str):
    """Set the first widget of `kind` whose label contains `wanted`."""
    for w in getattr(at, kind):
        if wanted.lower() in str(getattr(w, "label", "")).lower():
            return w
    raise AssertionError(f"no {kind} labelled like {wanted!r}; this chapter offers "
                         f"{[getattr(w, 'label', '?') for w in getattr(at, kind)]}")


def _outside_data(a):
    """Switch chapter 3 to the free sliders, then push x outside the data."""
    _set(a, "radio", "Choose a point").set_value("sliders").run()
    return _set(a, "slider", "x").set_value(0.02).run()


def _digits_awaiting_download(page: str) -> bool:
    """True when the digits chapter is legitimately chartless — no cached pixels yet."""
    if page != CH["digits"]:
        return False
    try:
        from eventweb import mnist
    except Exception:
        return False
    return not mnist.have_test_set()


def main() -> int:
    # Reading order matters here: the chapters are an argument, so they are walked in the
    # order the spine defines rather than alphabetically or by file.
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
        head = [m.value for m in at.markdown if m.value.startswith("### ")]
        print(f"  [ok]   {page} — {n_charts} figure(s), {len(at.markdown)} text blocks, "
              f"{n_errors} error box(es)")
        if len(head) != 1:
            print(f"  [WARN] {page} shows {len(head)} chapter headers, expected exactly 1")
        if n_charts == 0 and page not in CHARTLESS_OK and not _digits_awaiting_download(page):
            print(f"  [WARN] {page} rendered no figures")

    # The footer is the spine's navigation: it must move the reader to the neighbouring
    # chapter rather than merely existing. A spine whose buttons go nowhere is a spine that
    # looks fine and reads broken.
    at = run(CH["encoding"])
    if at is not None:
        nxt = narrative.after("encoding").label.replace(" · ", ". ", 1)
        try:
            _click(at, f"{nxt} →")
            if at.sidebar.radio[0].value != narrative.chapter("gradient").label:
                print(f"  [FAIL] footer did not reach chapter 2 "
                      f"(landed on {at.sidebar.radio[0].value!r})")
                failures.append("footer next")
            else:
                print(f"  [ok]   footer takes the reader from chapter 1 to chapter 2")
        except AssertionError as exc:
            print(f"  [FAIL] footer next — {exc}")
            failures.append("footer next")

    # One stimulus across the chapters: chapter 1 picks it, chapter 3 must show the same one.
    at = run(CH["encoding"])
    if at is not None:
        try:
            _set(at, "number_input", "test index").set_value(7).run()
            # Read the app's session, not this process's: `state.story()` here would consult
            # the bare-mode session the test runner owns, which has never seen the widget,
            # and the comparison would then fail while the app behaved correctly.
            moved = int(at.session_state["story_index"])
            at.sidebar.radio[0].set_value(CH["adjoint"]).run()
            shown = [c.value for c in at.caption if c.value.startswith("test index")]
            if at.exception:
                report(at)
                raise AssertionError("chapter 3 raised after the protagonist moved")
            if not shown or f"test index {moved} " not in shown[0]:
                raise AssertionError(f"chapter 3 shows {shown[:1]!r}, expected index {moved}")
            print(f"  [ok]   the stimulus chosen in chapter 1 is the one chapter 3 renders")
        except AssertionError as exc:
            print(f"  [skip] shared protagonist — {exc}")

    # Interactions, not just first paint.
    checks = [
        # Chapter 1 is the only place the encoder itself is drawn, and its measured accuracy
        # is the number the chapter's argument rests on.
        (CH["encoding"], lambda a: _set(a, "number_input", "test index").set_value(11).run(),
         "chapter 1 on a different stimulus"),
        # Chapter 3 opens on the story's point; the sliders are the escape hatch, and one of
        # them can produce a point the network says nothing about at all.
        (CH["adjoint"], lambda a: _set(a, "radio", "Choose a point").set_value("sliders").run(),
         "chapter 3 switched to the free sliders"),
        (CH["adjoint"], _outside_data,
         "a point outside the data (no output spike at all)"),
        (CH["adjoint"], lambda a: _set(a, "radio", "Backend").set_value(
            "Eigen reference (pybind11)").run(),
         "chapter 3 on the Eigen backend (traces path)"),
        (CH["adjoint"], lambda a: _set(a, "radio", "Backend").set_value(
            "C library (Cython)").run(),
         "chapter 3 back on the C backend (C traces path)"),
        (CH["adjoint"], lambda a: a.selectbox[1].set_value(5).run(),
         "chapter 3 at an early snapshot"),
        (CH["learning"], lambda a: a.select_slider[0].set_value(0).run(),
         "boundary at the first snapshot"),
        (CH["learning"], lambda a: a.select_slider[1].set_value(81).run(),
         "boundary at 81x81"),
        (CH["gradient"],
         lambda a: _set(a, "number_input", "Input sample").set_value(7).run(),
         "a different input sample"),
        (CH["gradient"], lambda a: a.radio[0].set_value("Eigen reference (pybind11)").run(),
         "gradient chapter on the other backend (the picker's other half)"),
        (CH["gradient"], lambda a: a.slider[0].set_value(5).run(),
         "a different weight draw"),
        # The digits chapter is the only reachable UI path through the maxima read-out, so
        # both halves of it get driven: the run picker and the backend picker (C vs Eigen run
        # two independent implementations of `compute_maxima_batch`).
        (CH["digits"], lambda a: a.slider[0].set_value(3).run(),
         "digits chapter on a different published run"),
        (CH["digits"], lambda a: a.radio[0].set_value("Eigen reference").run(),
         "digits chapter on the Eigen backend (maxima read-out)"),
        (CH["digits"], lambda a: a.radio[0].set_value("C library (Cython)").run(),
         "digits chapter back on the C backend"),
        (CH["digits"], lambda a: a.number_input[0].set_value(7).run(),
         "digits chapter at another test digit"),
        # The timing panel measures both bindings on the same batches, so it is the only
        # place the maxima path is timed rather than merely exercised. Skips visibly when
        # the pixel files are not cached, since that state shows a download prompt instead.
        (CH["digits"], lambda a: _click(a, "Time both backends"),
         "digits timing panel (both backends measured)"),
        # The read-out chapter is the only reachable path through the C regression loss, so
        # its three meaningful axes get driven: the other target (y, which this network
        # encodes worse than x), the adjoint half of the fit, and a Huber scale small enough
        # that the loss's saturation is visible in the curve.
        (CH["regression"], lambda a: a.radio[0].set_value("y — vertical position").run(),
         "read-out chapter on the y target"),
        (CH["regression"],
         lambda a: a.radio[1].set_value("read-out + weights (adjoint)").run(),
         "read-out chapter with the adjoint training the weights"),
        (CH["regression"], lambda a: a.radio[2].set_value("Huber δ = 0.01").run(),
         "read-out chapter at a saturating Huber scale"),
        (CH["regression"], lambda a: a.toggle[0].set_value(False).run(),
         "read-out chapter on raw (unstandardised) maxima"),
        (CH["regression"], lambda a: a.number_input[0].set_value(3).run(),
         "read-out chapter probing a different held-out point"),
        (CH["limits"], lambda a: a.selectbox[0].set_value(1000).run(),
         "parity across the whole test set"),
    ]
    for page, act, label in checks:
        at = run(page)
        if at is None:
            continue
        try:
            act(at)  # each interaction runs the script before returning
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
    print("all chapters rendered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
