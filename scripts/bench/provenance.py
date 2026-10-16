"""provenance.py — the vcs fields every record's meta carries.

`vcs_dirty` is meant to answer one question: **was the tracked source tree modified
when this was measured?** It used to be computed as `bool(git status --porcelain)`,
which on this repo is true on every machine that can run the python suite:

  * the read-only reference submodule MUST carry local patches to build
    (`scripts/submodule-state.sh` owns them), so it always reports
    ` M src/EventProp.Python`;
  * untracked tool output counts too — session logs, generated benchmark reports.

A flag that is true on every run cannot gate a claim, and `report.py`'s staleness
banner reads it as exactly that: "the numbers do not correspond to any committed
state". The committed darwin report says `vcs_dirty: True` for this reason, not
because anyone was editing source.

So the three signals are separated, one field each:

  vcs_dirty        tracked modifications. Submodule dirt excluded, untracked ignored.
                   This is the one the staleness banner is about.
  submodule_dirty  the submodule working tree is modified. EXPECTED — the patches are
                   required to build the reference library. A fact to state, not a
                   warning to raise.
  untracked        count of untracked files. Tool output; informative, not a defect.

No git repository (an exported tree, a worktree without git metadata) yields None for
every field: the absence of a repository is neither a clean tree nor a dirty one, and
reporting False there would be a claim that nothing was checked.

Stdlib only, like the rest of scripts/bench/.
"""

import subprocess

__all__ = ["vcs_state"]


def _git(*args, cwd=None):
    """Run git, returning stdout. None on any failure: no git, not a repo, timeout."""
    try:
        p = subprocess.run(("git",) + args, cwd=cwd, capture_output=True,
                           text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return p.stdout if p.returncode == 0 else None


def _toplevel(cwd):
    """The superproject root, so every git call below sees one directory.

    Measured: run from scripts/bench, `git status --porcelain` reports the submodule as
    `../../src/EventProp.Python` (paths are relative to cwd) while `git submodule
    status` reports `src/EventProp.Python` (relative to the superproject root), so the
    path comparison missed and submodule_dirty read False from a subdirectory and True
    from the root — same repository, same state, two answers. Pinning every call to the
    top level removes the class, and keeps vcs_dirty repo-wide rather than scoped to
    whatever directory the adapter happened to run in.
    """
    top = _git("rev-parse", "--show-toplevel", cwd=cwd)
    if top is None:
        return cwd
    top = top.strip()
    return top or cwd


def _submodule_paths(root):
    out = _git("submodule", "status", "--recursive", cwd=root)
    if out is None:
        return None
    paths = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            # git prints forward slashes on every platform; normalise the comparison.
            paths.add(parts[1].replace("\\", "/"))
    return paths


def vcs_state(cwd=None):
    """Return {"vcs_rev", "vcs_dirty", "submodule_dirty", "untracked"}.

    Every value is None when it cannot be determined, so a record never asserts
    cleanliness it did not observe.
    """
    root = _toplevel(cwd)
    rev = _git("rev-parse", "--short", "HEAD", cwd=root)
    if rev is None:
        return {"vcs_rev": None, "vcs_dirty": None,
                "submodule_dirty": None, "untracked": None}

    # Tracked changes only: -uno drops untracked, --ignore-submodules=dirty drops the
    # submodule's patched working tree, which is the repo's required state.
    tracked = _git("status", "--porcelain", "--ignore-submodules=dirty", "-uno", cwd=root)
    # Everything, including submodule content and untracked files, so the two signals
    # that used to pollute vcs_dirty can be reported on their own terms.
    everything = _git("status", "--porcelain", "--ignore-submodules=none", "-uall", cwd=root)

    vcs_dirty = bool(tracked.strip()) if tracked is not None else None

    submodule_dirty = None
    sub_paths = _submodule_paths(root)
    if everything is not None and sub_paths is not None:
        submodule_dirty = False
        for line in everything.splitlines():
            if len(line) < 4:
                continue
            xy, path = line[:2], line[3:].replace("\\", "/")
            # porcelain v1 XY: ' M' = modified in the worktree, 'M ' = staged,
            # 'MM' = both. A gitlink entry with any of those means the submodule's
            # recorded SHA moved, or its working tree is modified.
            if path in sub_paths and ("M" in xy or "m" in xy):
                submodule_dirty = True
                break

    untracked = None
    if everything is not None:
        untracked = sum(1 for line in everything.splitlines() if line.startswith("?? "))

    return {"vcs_rev": rev.strip(), "vcs_dirty": vcs_dirty,
            "submodule_dirty": submodule_dirty, "untracked": untracked}
