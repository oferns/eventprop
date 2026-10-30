"""EventProp Visualizer — Streamlit entry point, and the chapter spine.

The app reads as one argument, not six panels: `eventweb/narrative.py` owns the chapter
order, the question each chapter asks and what pins its numbers, and this file is the shell
that renders that spine — a shared header, the selected view, and a prev/next footer whose
"next" caption is the previous chapter's unresolved tension. Each `views/*.py` renders its
own content but no longer its own title, so the header appears exactly once per chapter.

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

from eventweb import narrative, runtime  # noqa: E402

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
    "Exact gradients through spike times, on the real library, told in order: what a spike "
    "train encodes, why its gradient looks impossible, what the adjoint is, and whether any "
    "of it survives published data."
)

labels = narrative.labels()
# The footer buttons drive the same key through `on_click`, because a callback runs before
# the next script run instantiates the radio: assigning a widget's state after it exists is
# an error, and a button that sets it inline would be exactly that.
picked = st.sidebar.radio("Chapters", labels, key="chapter_label",
                          label_visibility="collapsed")
chapter = narrative.chapter(narrative.slug_of_label(picked))

st.sidebar.divider()
st.sidebar.caption(
    "Ten published Yin-Yang networks ship in this checkout, so nothing here trains while you "
    "wait. `selfcheck.py` verifies the numbers the plots show, and every chapter names the "
    "checks behind its claims."
)
st.sidebar.caption(f"Interpreter: {sys.version.split()[0]}")

# --- Chapter header, rendered once here so no view repeats it ---------------------------
st.markdown(f"### {chapter.number}. {chapter.title}")
st.caption(chapter.question)

module = __import__(f"views.{chapter.view}", fromlist=["render"])
module.render()

# --- Footer: the previous chapter's tension becomes the next chapter's reason -----------
def _goto(label: str) -> None:
    st.session_state["chapter_label"] = label



prev_ch = narrative.before(chapter.slug)
next_ch = narrative.after(chapter.slug)
st.divider()
left, right = st.columns(2)
if prev_ch is not None:
    left.button(f"← {prev_ch.number}. {prev_ch.title}", key="prev",
                on_click=_goto, args=(prev_ch.label,))
if next_ch is not None:
    right.button(f"{next_ch.number}. {next_ch.title} →", key="next",
                 on_click=_goto, args=(next_ch.label,))
if next_ch is not None and chapter.hands_off:
    st.caption(f"*{chapter.hands_off}*")
