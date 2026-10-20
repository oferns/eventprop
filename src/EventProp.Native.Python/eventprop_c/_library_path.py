"""Make eventprop_c_api.dll findable by the native extension on Windows.

`eventprop_c/_native.pyd` links against `eventprop_c_api.dll`, and the loader has
to find that DLL when the extension is imported.

Since Python 3.8, CPython loads extension modules with
`LOAD_LIBRARY_SEARCH_DEFAULT_DIRS`. The search set that flag enables is: the
extension's own directory, the process directory, the system directories, and
directories registered with `os.add_dll_directory`. **PATH is not in that set.**
The usual "prepend the library directory to PATH" recipe therefore cannot work
for a pyd — measured on Windows: with PATH pointing at the directory that
contains eventprop_c_api.dll, `import eventprop_c._native` fails with
"ImportError: DLL load failed while importing _native"; with
`os.add_dll_directory` on the same directory it imports.

On Linux this module PRELOADS libeventprop_c_api.so by absolute path before the
extension is imported, which is the POSIX counterpart of os.add_dll_directory: a library
already loaded with RTLD_GLOBAL satisfies a later extension's DT_NEEDED by soname.
setup.py now also bakes an rpath for the repo layout, so the preload is the fallback for
the case an rpath cannot cover — an installed eventprop_c, or a checkout that moved after
the extension was built — and it is what makes EVENTPROP_C_LIB_DIR mean something here.
Before both changes, a plain `import eventprop_c` after a green build failed with
"ImportError: libeventprop_c_api.so: cannot open shared object file" unless the caller
happened to export LD_LIBRARY_PATH.

On macOS nothing is registered: the extension is linked with -Wl,-rpath at build time,
and DYLD_LIBRARY_PATH works there.

`os.add_dll_directory` returns a handle that REMOVES the directory from the
search path when it is closed or garbage collected, so the handles are kept in
`_handles` for the lifetime of the process. Registration is process-wide and
irreversible from Python's point of view, so this runs once, at package import,
before `._native` is imported.
"""

import ctypes
import os
import sys

__all__ = ["register_c_library_dirs", "registered_directories",
           "preloaded_libraries", "candidate_directories"]

# Handles must outlive this module's function calls: dropping the last reference
# unregisters the directory. Module globals live as long as sys.modules does.
_handles = []
_registered = []
# Loaded libraries (Linux). Kept for the same reason as _handles and for diagnostics;
# a library stays in the process regardless, but the list is what a test can assert on.
_preloaded = []

# What the loader calls the C library on each platform, as the extension's DT_NEEDED
# records it — the exact name the search has to satisfy.
_LIBRARY_FILENAMES = {
    "nt": "eventprop_c_api.dll",
    "darwin": "libeventprop_c_api.dylib",
}


def _library_name():
    return _LIBRARY_FILENAMES.get(sys.platform, "libeventprop_c_api.so")


def candidate_directories():
    """Directories that may hold eventprop_c_api.dll, in preference order.

    EVENTPROP_C_LIB_DIR (os.pathsep-separated) wins, because an installed
    eventprop_c has no repo layout to derive the path from, and that is the
    documented escape hatch. The repo layout is then derived from this file:
    <root>/src/EventProp.Native.Python/eventprop_c/ -> <root>/src/EventProp.C/build.
    Release and Debug are probed because MSVC's CMake generator is multi-config
    and puts the DLL in build/Release; single-config generators put it in build/.
    """
    env = os.environ.get("EVENTPROP_C_LIB_DIR")
    if env:
        for entry in env.split(os.pathsep):
            entry = entry.strip()
            if entry:
                yield os.path.abspath(os.path.expanduser(entry))

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    c_build = os.path.normpath(os.path.join(pkg_dir, "..", "..", "EventProp.C", "build"))
    for sub in ("Release", "Debug", ""):
        yield os.path.join(c_build, sub) if sub else c_build


def register_c_library_dirs():
    """Make the C library findable, by whatever mechanism this platform has.

    Windows -> registers each candidate directory with os.add_dll_directory and returns
    those directories. Linux -> preloads libeventprop_c_api.so from each candidate and
    returns the library paths it loaded. macOS -> no-op returning [] (build-time rpath).
    Non-3.8+ Windows has no os.add_dll_directory, and is a no-op too.

    Idempotent: repeated calls register or load nothing new.
    """
    if os.name == "nt":
        if not hasattr(os, "add_dll_directory"):
            return []
        for directory in candidate_directories():
            if not os.path.isdir(directory):
                continue
            if directory in _registered:
                continue
            _handles.append(os.add_dll_directory(directory))
            _registered.append(directory)
        return list(_registered)

    if sys.platform == "linux":
        name = _library_name()
        for directory in candidate_directories():
            library = os.path.join(directory, name)
            if not os.path.isfile(library):
                continue
            if library in _preloaded:
                continue
            try:
                # RTLD_GLOBAL so the already-loaded library answers the extension's
                # DT_NEEDED when CPython loads _native.*.so a moment later. Errors are
                # swallowed on purpose: if this library cannot load, the very next line
                # the package runs — `from ._native import ...` — fails with the loader's
                # own message, which names the real problem. A second error here would
                # only compete with it.
                ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue
            _preloaded.append(library)
        return list(_preloaded)

    return []


def preloaded_libraries():
    """Library paths preloaded by register_c_library_dirs() on Linux, in order."""
    return list(_preloaded)


def registered_directories():
    """The directories registered by register_c_library_dirs(), in order."""
    return list(_registered)
