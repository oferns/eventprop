"""EventProp Visualizer — Streamlit entry point.

Starts with an ABI check rather than an import: both backends are pre-built extension
modules owned by sibling projects, and loading them with the wrong interpreter fails as a
`ModuleNotFoundError` from inside `eventweb.config`-style imports, which reads as a broken
app. `eventweb/runtime.py` turns that into one sentence naming the interpreter, the tag
wanted and the tags present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eventweb import runtime  # noqa: E402

st.set_page_config(page_title="EventProp Visualizer", page_icon="⚡", layout="wide")

try:
    runtime.ensure_native_abi()
    runtime.add_backend_paths()
except RuntimeError as exc:
    st.title("EventProp Visualizer")
    st.error(f"{exc}\n\nThe app needs both native backends to run.", exc=False)
    st.stop()

st.title("EventProp Visualizer")
st.caption(
    "Exact gradients through spike times — running on the real library, on the task "
    "EventProp was published with."
)

page = st.sidebar.radio(
    "Explore",
    ["Yin-Yang, live", "How it learns", "Exact gradients", "Under the hood"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.caption(
    "Ten published Yin-Yang networks ship in this checkout, so nothing on these pages "
    "trains while you wait. `selfcheck.py` verifies the numbers the plots show."
)
st.sidebar.caption(f"Interpreter: {sys.version.split()[0]}")

if page == "Yin-Yang, live":
    from views import live

    live.render()
elif page == "How it learns":
    from views import learning

    learning.render()
elif page == "Exact gradients":
    from views import gradient

    gradient.render()
else:
    from views import internals

    internals.render()
