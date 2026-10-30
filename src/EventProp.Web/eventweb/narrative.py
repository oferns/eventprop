"""The chapter order, the question each chapter asks, and what pins its numbers.

Only the spine lives here — the pages keep their own prose, because copy that is separated
from the figure it explains is copy that rots. What this module owns is the *order* and the
*argument*: a dashboard is a set of panels you can visit in any order, a narrative is a set
of chapters where each one leaves a debt the next one pays.

Two rules make that checkable rather than aspirational:

*   Every chapter states a `question` that a control cannot answer on its own, and a
    `hands_off` — the sentence that hands the next chapter its tension. A chapter with no
    hands-off is where the reading stops, so the list refuses to load with a hole in it
    (`_check_arc`).
*   Every chapter names the `selfcheck.py` steps that pin the numbers it quotes. That is a
    pointer, not a second measurement: selfcheck owns the measurements, and prose that
    cites no step is prose nothing verifies. `uncited_claims()` is what catches that.

Deliberately Streamlit-free: the spine has to be importable by `selfcheck.py` and by any
future non-interactive renderer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chapter:
    number: int
    slug: str
    title: str
    view: str
    question: str
    hands_off: str
    pinned_by: tuple[str, ...]
    #: module under `views/` that renders this chapter. Owned by the spine rather than by
    #: the app shell, so `selfcheck.py` can verify every chapter still resolves to a real
    #: page instead of keeping a second copy of the mapping that could drift.

    @property
    def label(self) -> str:
        """What the sidebar and the footer call this chapter."""
        return f"{self.number} · {self.title}"


CHAPTERS: tuple[Chapter, ...] = (
    Chapter(
        number=1,
        slug="encoding",
        title="A spike train is a number",
        view="encoding",
        question="A point in this plane becomes four numbers: when each neuron fires. "
                 "What can a read-out possibly make of them?",
        hands_off="The answer is read off a race — whichever output neuron fires first. A "
                  "spike is a step, and a step has no gradient to give.",
        pinned_by=("selfcheck step 4 (shipped checkpoints run through the C library)",
                   "selfcheck step 3 (the frozen split against its own geometry)",
                   "selfcheck step 6 (both backends on the same spikes)"),
    ),
    Chapter(
        number=2,
        slug="gradient",
        title="You cannot backpropagate a spike",
        view="gradient",
        question="The output is a step function of the membrane voltage. Where is its "
                 "derivative, and what is a network supposed to do with a zero?",
        hands_off="So the gradient does not come from the output at all. It comes from "
                  "inside: from the one instant the voltage crossed threshold, and from a "
                  "signal that runs backwards through time to meet it.",
        pinned_by=("selfcheck step 7 (analytic gradient against central differences, "
                   "both backends)"),
    ),
    Chapter(
        number=3,
        slug="adjoint",
        title="The signal that runs backwards",
        view="live",
        question="Something has to carry the error from the read-out back to every crossing. "
                 "What is it, and why is it zero almost everywhere?",
        hands_off="That signal is what every plotted gradient is made of — and it is only as "
                  "honest as the network that was trained to answer with it. So: does the "
                  "training actually work?",
        pinned_by=("selfcheck step 8 (voltage and adjoint traces against the reference)",
                   "selfcheck step 9 (which backend can produce them at all)"),
    ),
    Chapter(
        number=4,
        slug="learning",
        title="Watch it learn",
        view="learning",
        question="Does the rule described above actually move a network, on the task it was "
                 "published with — and where does it still get things wrong?",
        hands_off="Nine-tenths of a percent of the test set is still wrong, on a synthetic "
                  "shape drawn with two equations. Does any of this survive data that was "
                  "never designed for it?",
        pinned_by=("selfcheck step 5 (snapshot/epoch alignment)",
                   "selfcheck step 10 (the checkpoint reader names the layout it parsed)",
                   "selfcheck step 6 (C against Eigen on every test sample)"),
    ),
    Chapter(
        number=5,
        slug="digits",
        title="Real data",
        view="digits",
        question="MNIST was not built for first-spike times. What does the same machinery "
                 "reproduce, and where does it disagree with the published record?",
        hands_off="Classification reads the *order* of spikes. But the same hidden layer "
                  "also records *how far* each neuron got — and that is a continuous number "
                  "nobody has asked this network to predict.",
        pinned_by=("selfcheck step 11 (the published MNIST/VMax checkpoints)",
                   "selfcheck step 12 (real digits, byte-identical encoder)",
                   "selfcheck step 13 (the timing panel agrees with a stopwatch)"),
    ),
    Chapter(
        number=6,
        slug="regression",
        title="Reading a number",
        view="regression",
        question="Can the same spikes fit a target that is not a class at all — and what "
                 "does a fitted curve have to be compared against?",
        hands_off="Every number on these pages came from a library call or a checkpoint. "
                  "Which of them are measurements, which are reproductions, and what is "
                  "known to be broken?",
        pinned_by=("selfcheck step 15 (the read-out, its chain rule, its baselines)"),
    ),
    Chapter(
        number=7,
        slug="limits",
        title="What this cannot show you",
        view="internals",
        question="What is a measurement here, what is a reproduction, what is a placeholder, "
                 "and what is a known defect?",
        hands_off="",  # the last chapter's job is to stop promising
        pinned_by=("selfcheck step 1 (the interpreter and ABI the app insists on)",
                   "selfcheck step 14 (what the two bindings hand you)"),
    ),
)


def _check_arc(chapters: tuple[Chapter, ...]) -> None:
    """Fail loudly if the spine has a hole in it.

    A chapter that hands off to nothing is where the reader stops reading, which is exactly
    the failure mode this module exists to prevent — so it is a load-time error rather than
    a review comment. Only the final chapter may hand off to nothing.
    """
    numbers = [c.number for c in chapters]
    if numbers != list(range(1, len(chapters) + 1)):
        raise ValueError(f"chapter numbers must be 1..{len(chapters)}, got {numbers}")
    slugs = [c.slug for c in chapters]
    if len(set(slugs)) != len(slugs):
        raise ValueError(f"chapter slugs must be unique, got {slugs}")
    for c in chapters:
        if not getattr(c, "view", "").strip():
            raise ValueError(f"chapter {c.number} ({c.slug}) has no view module")
    for c in chapters[:-1]:
        if not c.hands_off.strip():
            raise ValueError(f"chapter {c.number} ({c.slug}) hands off to nothing")
        if not c.pinned_by:
            raise ValueError(f"chapter {c.number} ({c.slug}) cites no selfcheck step")


def chapters() -> tuple[Chapter, ...]:
    return CHAPTERS


def labels() -> list[str]:
    """Sidebar/footer labels, in reading order."""
    return [c.label for c in CHAPTERS]


def chapter(slug: str) -> Chapter:
    for c in CHAPTERS:
        if c.slug == slug:
            return c
    raise KeyError(f"no chapter {slug!r}; there are {', '.join(c.slug for c in CHAPTERS)}")


def slug_of_label(label: str) -> str:
    """The chapter a sidebar/footer label names."""
    for c in CHAPTERS:
        if c.label == label:
            return c.slug
    raise KeyError(f"{label!r} is not a chapter label")


def number_of(label: str) -> int:
    """Which chapter a sidebar label refers to (the label is the app's page key)."""
    for c in CHAPTERS:
        if c.label == label:
            return c.number
    raise KeyError(f"{label!r} is not a chapter label")


def after(slug: str) -> Chapter | None:
    c = chapter(slug)
    nxt = [d for d in CHAPTERS if d.number == c.number + 1]
    return nxt[0] if nxt else None


def before(slug: str) -> Chapter | None:
    c = chapter(slug)
    prv = [d for d in CHAPTERS if d.number == c.number - 1]
    return prv[0] if prv else None


def claim_coverage() -> list[tuple[str, tuple[str, ...]]]:
    """`(slug, selfcheck steps)` for every chapter — the spine's audit trail.

    `selfcheck.py` walks this: it fails if a chapter cites a step that no longer exists, so
    renaming or deleting a check cannot silently leave seven pages of prose quoting it.
    """
    return [(c.slug, c.pinned_by) for c in CHAPTERS]


def uncited_claims() -> list[str]:
    """Slugs whose number of pinned steps is zero — empty by construction, kept honest by
    `_check_arc`; exposed so the claim is a testable statement rather than a comment."""
    return [c.slug for c in CHAPTERS if not c.pinned_by]


_check_arc(CHAPTERS)
