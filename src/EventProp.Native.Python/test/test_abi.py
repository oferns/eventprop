"""ABI self-check: the extension's compiled struct layouts vs the library's.

Every consumer of the C library COMPILES its struct layouts into its own binary —
this extension via `cdef extern struct`, `EventProp.Native` via `[StructLayout]`,
the C++ consumers via the headers. A header change that moves a field offset is
invisible at link time: the consumer keeps reading the offsets it compiled, against a
library whose fields moved, and reads plausible-looking garbage.

The failure this pins was measured, not imagined: a `_native.cp311-win_amd64.pyd`
built against July headers, loaded against a library rebuilt from October headers,
produced an access violation inside `spikes.py __getitem__` via
`loss_layer._find_first_spikes` — 40 minutes into a benchmark run, in a frame with no
hint that a header/binary offset mismatch was the cause. Nothing in the build checked
that the extension was newer than the headers it was compiled from.

`_native.pyx` now refuses to import on a mismatch. These tests assert the check
passes, that it COVERS every public struct (a struct added to the headers without an
entry is the drift that reintroduces the bug), and — via ctypes, which loads the DLL
without consulting the extension's compiled offsets — that the comparison is a real
cross-binary one rather than a tautology.
"""

import ctypes
import os
import sys

import pytest

from eventprop_c._native import abi_check, abi_struct_sizes
from eventprop_c._library_path import candidate_directories, registered_directories

# The public structs a consumer embeds by value. Kept here as well as in
# eventprop_abi.c and the C++/C# tests on purpose: adding a struct to the headers
# must fail a test in every consumer, which is what forces the check to be extended.
STRUCT_NAMES = {
    "EventPropInput",
    "SpikeOutput",
    "SumsOutput",
    "MaximaOutput",
    "BackwardOutput",
    "SpikesBatch",
    "MaximaBatch",
    "BackwardsBatch",
    "GradientOutput",
    "DropoutOutput",
    "DropoutBatch",
    "AccumulatedBackwardResult",
    "VoltageTraceInput",
    "VoltageTraceOutput",
    "LambdaITraceInput",
    "LambdaITraceOutput",
}

_LIB_NAMES = ("eventprop_c_api.dll",
              "libeventprop_c_api.dylib",
              "libeventprop_c_api.so")


def _load_c_library():
    """Load the C library through ctypes, independently of the extension.

    On Windows the bare name resolves via the directories
    `eventprop_c/_library_path.py` registered with os.add_dll_directory — the same
    mechanism the extension's own dependency resolution uses, so this loads the DLL
    the extension loaded.
    """
    tried = []
    for name in _LIB_NAMES:
        # Full paths from the repo layout first, then the bare name (search rules).
        for directory in list(registered_directories()) + list(candidate_directories()) + [""]:
            candidate = os.path.join(directory, name) if directory else name
            if directory and not os.path.exists(candidate):
                tried.append(candidate)
                continue
            try:
                return ctypes.CDLL(candidate)
            except OSError as exc:
                tried.append(f"{candidate}: {exc}")
    pytest.skip(f"could not load the C library for the cross-binary check; tried: {tried[:6]}")


def _abi_symbols(lib):
    lib.eventprop_abi_version.argtypes = []
    lib.eventprop_abi_version.restype = ctypes.c_int
    lib.eventprop_abi_struct_size.argtypes = [ctypes.c_char_p]
    lib.eventprop_abi_struct_size.restype = ctypes.c_size_t
    return lib


def test_abi_check_passes():
    ok, details = abi_check()
    assert ok, f"ABI mismatch: {details['mismatches']}"


def test_versions_are_positive_and_agree():
    _, details = abi_check()
    assert details["library_abi_version"] >= 1
    assert details["compiled_abi_version"] == details["library_abi_version"]


def test_every_public_struct_is_measured():
    """A struct added to the headers without an entry in the check is the drift that
    reintroduces the offset-mismatch bug, so the coverage is asserted, not assumed."""
    sizes = abi_struct_sizes()
    assert set(sizes) == STRUCT_NAMES, (
        f"missing={sorted(STRUCT_NAMES - set(sizes))} extra={sorted(set(sizes) - STRUCT_NAMES)}")
    assert all(size > 0 for size in sizes.values())


def test_sizes_match_the_library_loaded_via_ctypes():
    """The comparison the import-time guard performs, asserted from outside the
    extension: ctypes reads the DLL's own sizeof, the extension reports its compiled
    sizeof. Two independent compilations of the same header must agree."""
    lib = _abi_symbols(_load_c_library())
    _, details = abi_check()
    assert lib.eventprop_abi_version() == details["compiled_abi_version"]
    for name, compiled_size in abi_struct_sizes().items():
        assert lib.eventprop_abi_struct_size(name.encode("utf-8")) == compiled_size, name


def test_unknown_struct_name_reports_zero():
    """A consumer asking about a struct the library does not know must fail its own
    check, not read 0 as a match."""
    lib = _abi_symbols(_load_c_library())
    assert lib.eventprop_abi_struct_size(b"NotAStruct") == 0
    assert lib.eventprop_abi_struct_size(b"") == 0


def test_check_is_exposed_from_the_package():
    """The guard runs at import, but the package must also expose it: a caller that
    loads the library late (a frozen app, a plugin host) needs a way to ask."""
    import eventprop_c

    assert callable(eventprop_c.abi_check)
    assert callable(eventprop_c.abi_struct_sizes)
    ok, details = eventprop_c.abi_check()
    assert ok and isinstance(details, dict)


@pytest.mark.skipif(sys.platform != "win32",
                    reason="POSIX builds resolve the library by rpath; no DLL directory to register")
def test_the_library_under_test_is_the_one_the_extension_loaded():
    """The extension links eventprop_c_api at build time by rpath/dll-directory. If a
    stale copy shadowed the fresh one, abi_check() would compare the extension against
    the stale library and pass, while the numbers came from the stale code. This asserts
    the DLL ctypes resolves is in a directory the registration chose."""
    lib = _load_c_library()
    path = ctypes.cast(lib._handle, ctypes.c_void_p).value
    # GetModuleFileNameW on the HMODULE.
    buf = ctypes.create_unicode_buffer(32768)
    ctypes.windll.kernel32.GetModuleFileNameW.restype = ctypes.c_uint
    ctypes.windll.kernel32.GetModuleFileNameW.argtypes = [ctypes.c_void_p,
                                                          ctypes.c_wchar_p, ctypes.c_uint]
    n = ctypes.windll.kernel32.GetModuleFileNameW(path, buf, 32768)
    assert n > 0, "GetModuleFileNameW failed"
    loaded = os.path.normcase(os.path.abspath(buf.value))
    assert os.path.basename(loaded) == "eventprop_c_api.dll"
    # It must be a build output of this repo, not a copy of an older build sitting on
    # PATH — the registration in _library_path.py is what makes that the case, and a
    # stale library that passes the ABI check still produces stale numbers.
    parts = os.path.normcase(loaded).split(os.sep)
    assert "build" in parts or "build-werror" in parts, (
        f"loaded {loaded}, which is not under a build directory of this repo")
