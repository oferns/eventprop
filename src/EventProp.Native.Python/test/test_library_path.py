"""Pins the Windows DLL search-path setup that makes _native.pyd importable.

The failure this guards: Python 3.8+ loads extension modules with
LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, whose search set does NOT include PATH. So
eventprop_c_api.dll is found only if it sits next to the extension, in a system
directory, or in a directory registered with os.add_dll_directory.

Measured on Windows/x64:
    PATH prepended with the directory holding eventprop_c_api.dll
        -> ImportError: DLL load failed while importing _native
    os.add_dll_directory on that same directory
        -> imports

The PATH recipe is what scripts/bench/run_all.py used to rely on, which is why
the python suite could not run on Windows while its prereq check passed.
"""

import ctypes
import os
import sys
import unittest
from unittest import mock

from eventprop_c import _library_path
from eventprop_c import _native

_WINDOWS = os.name == "nt"
_LINUX = sys.platform == "linux"


def _extension_dir():
    return os.path.dirname(os.path.abspath(_native.__file__))


def _dll_next_to_extension():
    """True when the loader needs no registration: the DLL is in the extension's
    own directory, which LOAD_LIBRARY_SEARCH_DEFAULT_DIRS always searches."""
    return os.path.isfile(os.path.join(_extension_dir(), "eventprop_c_api.dll"))


class NativeExtensionImportTest(unittest.TestCase):
    """Cross-platform: the extension imports and exposes the batch API."""

    def test_extension_imports_and_exposes_the_batch_api(self):
        for name in ("compute_spikes_batch", "compute_maxima_batch",
                     "backward_spikes_batch_accumulate",
                     "backward_maxima_batch_accumulate", "dropout_spikes_batch"):
            self.assertTrue(hasattr(_native, name), f"_native has no {name}")


@unittest.skipUnless(_WINDOWS, "os.add_dll_directory is Windows-only")
class WindowsDllDirectoryTest(unittest.TestCase):

    def setUp(self):
        if _dll_next_to_extension():
            self.skipTest("eventprop_c_api.dll sits next to the extension; "
                          "registration is not the mechanism in play here")

    def test_package_import_registered_a_directory_holding_the_dll(self):
        registered = _library_path.registered_directories()
        self.assertTrue(registered,
                        "package import registered no DLL directory")
        with_dll = [d for d in registered
                    if os.path.isfile(os.path.join(d, "eventprop_c_api.dll"))]
        self.assertTrue(
            with_dll,
            f"no registered directory contains eventprop_c_api.dll: {registered}")

    def test_registration_is_idempotent(self):
        first = _library_path.register_c_library_dirs()
        second = _library_path.register_c_library_dirs()
        self.assertEqual(first, second)

    def test_env_var_entries_are_candidates_and_come_first(self):
        with mock.patch.dict(os.environ, {"EVENTPROP_C_LIB_DIR": "C:\\definitely-not-a-dir"}):
            candidates = list(_library_path.candidate_directories())
        self.assertEqual(candidates[0], "C:\\definitely-not-a-dir")

    def test_env_var_is_pathsep_split(self):
        env = "C:\\first-dir" + os.pathsep + "C:\\second-dir"
        with mock.patch.dict(os.environ, {"EVENTPROP_C_LIB_DIR": env}):
            candidates = list(_library_path.candidate_directories())
        self.assertEqual(candidates[:2], ["C:\\first-dir", "C:\\second-dir"])

    def test_repo_layout_candidates_cover_the_multi_config_subdirs(self):
        # MSVC's generator writes build/Release; single-config writes build/.
        with mock.patch.dict(os.environ, {"EVENTPROP_C_LIB_DIR": ""}):
            candidates = list(_library_path.candidate_directories())
        self.assertTrue(any(c.endswith(os.path.join("build", "Release")) for c in candidates),
                        candidates)
        self.assertTrue(any(c.endswith("build") for c in candidates), candidates)


@unittest.skipUnless(_LINUX, "the Linux mechanism is the RTLD_GLOBAL preload")
class LinuxLibraryLookupTest(unittest.TestCase):
    """Pins what makes _native*.so importable on Linux.

    The failure this guards, measured on the first Linux build: after a build.sh that
    reported every step green, `import eventprop_c` died with
        ImportError: libeventprop_c_api.so: cannot open shared object file
    because the extension's DT_NEEDED is the bare soname, nothing pointed a RUNPATH at
    src/EventProp.C/build, and nothing had preloaded the library. The cross-platform
    import test in this file is the one that had been failing.
    """

    def test_package_import_preloaded_the_c_library(self):
        preloaded = _library_path.preloaded_libraries()
        if not preloaded:
            self.skipTest("no preload happened; the extension's RUNPATH resolved the "
                          "library on its own — the import test above is the real guard")
        for path in preloaded:
            self.assertTrue(os.path.isfile(path), f"preloaded a path that is not a file: {path}")

    def test_preload_is_idempotent(self):
        first = _library_path.register_c_library_dirs()
        second = _library_path.register_c_library_dirs()
        self.assertEqual(first, second)

    def test_repo_layout_candidate_is_the_single_config_build_dir(self):
        # CMake presets write ${sourceDir}/build on single-config generators, which is
        # every non-MSVC platform; the multi-config subdirs are yielded as well.
        with mock.patch.dict(os.environ, {"EVENTPROP_C_LIB_DIR": ""}):
            candidates = list(_library_path.candidate_directories())
        self.assertTrue(any(c.endswith(os.path.join("EventProp.C", "build")) for c in candidates),
                        candidates)


if __name__ == "__main__":
    unittest.main()
