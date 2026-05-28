# PyInstaller runtime hook: register Qt DLL search paths before any Qt import.
# Python 3.8+ on Windows ignores PATH for extension-module DLL loading.


def _pyi_rthook():
    import os
    import sys

    if not getattr(sys, "frozen", False):
        return

    base = getattr(sys, "_MEIPASS", "")
    if not base:
        return

    for sub in ("", "PySide6", "shiboken6"):
        path = os.path.join(base, sub) if sub else base
        if os.path.isdir(path):
            os.add_dll_directory(path)

    prepend = os.pathsep.join(
        p for p in (os.path.join(base, "PySide6"), os.path.join(base, "shiboken6"), base) if os.path.isdir(p)
    )
    os.environ["PATH"] = prepend + os.pathsep + os.environ.get("PATH", "")


_pyi_rthook()
del _pyi_rthook
