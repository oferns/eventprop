"""Interpreter/ABI guard and sys.path wiring for the EventProp Visualizer.

Both native backends are *pre-built* CPython extension modules owned by sibling
projects (`src/EventProp.Python` builds the pybind11/Eigen core, `src/EventProp.Native.Python`
builds the Cython layer over the C library). This app does not build them and must not
try to.

A CPython extension module carries its ABI tag in its *filename*
(`eventprop_cpp.cpython-314-darwin.so`), which means an extension built for another
interpreter is simply invisible: the app dies with
`ModuleNotFoundError: No eventprop.eventprop_cpp` several frames into `config.py`, which
reads like a broken app rather than like the wrong interpreter. `build.sh` asks the same
question for the same reason (see its Python-layers step).

So this module answers one question up front — *can this interpreter load both backends*
— and turns a confusing import chain into one actionable sentence. The EXT_SUFFIX probe is
the portable form: it carries the platform tag on every platform this repo builds on
(`.cpython-314-darwin.so`, `.cpython-314-x86_64-linux-gnu.so`, `.cp314-win_amd64.pyd`).
"""

from __future__ import annotations

import glob
import os
import sys
import sysconfig
from pathlib import Path


def _find_root() -> Path:
    """Walk up to the repository root instead of counting parent directories.

    Depth arithmetic (`parent.parent.parent`) breaks the moment this file moves, and it
    fails as a missing-path error from deep inside a view rather than at import.
    """
    here = Path(__file__).resolve().parent
    for cand in (here, *here.parents):
        if (cand / "src" / "EventProp.C").is_dir():
            return cand
    raise RuntimeError(
        f"could not locate the EventProp repository root above {here}; "
        "the Visualizer must stay inside the checkout because it loads the "
        "pre-built backends from its sibling projects"
    )


ROOT = _find_root()
BACKEND_PATHS = {
    "pybind11": ROOT / "src" / "EventProp.Python",
    "cython": ROOT / "src" / "EventProp.Native.Python",
}
# Where each backend's built extension module lives, relative to its project root.
EXT_GLOBS = {
    "pybind11": "eventprop/eventprop_cpp.*",
    "cython": "eventprop_c/_native.*",
}


def abi_tag() -> str:
    """The ABI tag this interpreter expects an extension module to carry.

    `EXT_SUFFIX` is the full suffix (e.g. `.cpython-314-darwin.so`); the tag is the part
    between the first dot and the platform-specific tail (e.g. `cpython-314`).
    """
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ""
    parts = suffix.lstrip(".").split(".")
    return parts[0] if parts and parts[0] else ""


def find_extensions() -> dict[str, list[str]]:
    """Every built extension per backend, whatever ABI it was built for."""
    out: dict[str, list[str]] = {}
    for name, rel in EXT_GLOBS.items():
        pattern = str(BACKEND_PATHS[name] / rel)
        out[name] = sorted(
            os.path.basename(p)
            for p in glob.glob(pattern)
            # `eventprop_cpp.cpython-314-darwin.so` is a build artefact; the `.pyi`
            # and `.py` siblings are not extension modules.
            if p.rsplit(".", 1)[-1] in ("so", "dylib", "pyd")
        )
    return out


def _matches(filename: str, tag: str) -> bool:
    """True if `filename` is an extension built for ABI `tag`.

    Matches the tag as a delimited component so `cpython-31` cannot accidentally match
    `cpython-314`. Windows spells the tag `cp314` inside `.cp314-win_amd64.pyd`.
    """
    stem = filename.split(".")[1:-1]
    return tag in stem


def ensure_native_abi() -> str:
    """Raise unless *this* interpreter can load both pre-built backends.

    Returns the ABI tag on success, so callers can log it.
    """
    tag = abi_tag()
    found = find_extensions()
    missing = [
        name for name, files in found.items() if not any(_matches(f, tag) for f in files)
    ]
    if not missing:
        return tag

    lines = [
        f"This interpreter ({sys.version.split()[0]}, ABI tag '{tag}') cannot load the "
        f"pre-built extension module(s) for: {', '.join(missing)}.",
        "C extension filenames carry their ABI tag, so a module built for another "
        "interpreter is not merely stale — it is invisible.",
    ]
    for name in missing:
        present = found[name] or ["<none built yet>"]
        lines.append(f"  {name}: found {', '.join(present)}")
    lines.append(
        f"Fix: create the venv with an interpreter whose tag matches, e.g.\n"
        f"  python3.14 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt\n"
        f"...or rebuild the extensions for THIS interpreter:\n"
        f"  ./build.sh --python {Path(sys.executable).name}"
    )
    raise RuntimeError("\n".join(lines))


def add_backend_paths() -> None:
    """Put both backend projects on sys.path (idempotent)."""
    for path in BACKEND_PATHS.values():
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)
