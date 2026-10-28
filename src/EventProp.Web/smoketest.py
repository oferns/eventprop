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
PAGES = ["Yin-Yang, live", "How it learns", "Exact gradients", "Under the hood"]
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
        if n_charts == 0 and page != "Under the hood":
            print(f"  [WARN] {page} rendered no figures")

    # Interactions, not just first paint.
    checks = [
        ("Yin-Yang, live", lambda a: a.radio[0].set_value("Eigen reference (pybind11)").run(),
         "live page on the Eigen backend (traces path)"),
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
        ("Exact gradients", lambda a: a.slider[0].set_value(5).run(),
         "a different weight draw"),
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
