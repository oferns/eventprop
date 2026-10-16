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

On POSIX this module does nothing: the extension embeds an rpath at build time
(see `_native.pyx`), and DYLD_LIBRARY_PATH / LD_LIBRARY_PATH do work there.

`os.add_dll_directory` returns a handle that REMOVES the directory from the
search path when it is closed or garbage collected, so the handles are kept in
`_handles` for the lifetime of the process. Registration is process-wide and
irreversible from Python's point of view, so this runs once, at package import,
before `._native` is imported.
"""

import os

__all__ = ["register_c_library_dirs", "registered_directories", "candidate_directories"]

# Handles must outlive this module's function calls: dropping the last reference
# unregisters the directory. Module globals live as long as sys.modules does.
_handles = []
_registered = []


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
    """Register every candidate directory that exists. Returns the registered list.

    A no-op returning [] on non-Windows and on Python < 3.8 (no
    os.add_dll_directory). Idempotent: repeated calls register nothing new.
    """
    if not (os.name == "nt" and hasattr(os, "add_dll_directory")):
        return []

    for directory in candidate_directories():
        if not os.path.isdir(directory):
            continue
        if directory in _registered:
            continue
        _handles.append(os.add_dll_directory(directory))
        _registered.append(directory)
    return list(_registered)


def registered_directories():
    """The directories registered by register_c_library_dirs(), in order."""
    return list(_registered)
