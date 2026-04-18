import os
import site
import sys
from pathlib import Path

_DLL_HANDLES = []


def add_ortools_dll_directory() -> None:
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return

    candidates = []
    for package_dir in site.getsitepackages():
        candidates.append(Path(package_dir) / "ortools" / ".libs")

    user_site = site.getusersitepackages()
    if user_site:
        candidates.append(Path(user_site) / "ortools" / ".libs")

    for dll_dir in candidates:
        if dll_dir.exists():
            _DLL_HANDLES.append(os.add_dll_directory(str(dll_dir)))
            return
