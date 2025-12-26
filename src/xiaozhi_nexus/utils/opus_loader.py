from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
import sys
from pathlib import Path


def _patch_find_library(name: str, path: str) -> None:
    original = ctypes.util.find_library

    def patched_find_library(query: str):
        if query == name:
            return path
        return original(query)

    ctypes.util.find_library = patched_find_library  # type: ignore[assignment]


def _repo_root() -> Path:
    # .../src/xiaozhi_nexus/utils/opus_loader.py -> repo root is parents[3]
    return Path(__file__).resolve().parents[3]


def _candidate_paths() -> list[Path]:
    candidates: list[Path] = []

    env = os.getenv("XIAOZHI_OPUS_LIB")
    if env:
        candidates.append(Path(env))

    root = _repo_root()

    # 1) local vendored path (if user copies libs into this repo)
    candidates.append(root / "libs" / "libopus" / "win" / "x64" / "opus.dll")

    # 2) sibling simple-xiaozhi checkout (common in this workspace)
    candidates.append(
        root.parent / "simple-xiaozhi" / "libs" / "libopus" / "win" / "x64" / "opus.dll"
    )

    # 3) system name (let ctypes resolve if installed)
    return candidates


def setup_opus() -> bool:
    """
    Ensure libopus is discoverable for `opuslib` (Windows: opus.dll).
    - Uses `XIAOZHI_OPUS_LIB` if provided.
    - Falls back to common workspace locations.
    """

    if getattr(sys, "_xiaozhi_opus_loaded", False):
        return True

    system = platform.system().lower()
    is_windows = system.startswith("win")

    # Try explicit / workspace DLL locations first.
    for path in _candidate_paths():
        if path.exists():
            if is_windows:
                dll_dir = str(path.parent)
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(dll_dir)
                    except Exception:
                        pass
                os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")

            _patch_find_library("opus", str(path))
            ctypes.CDLL(str(path))
            sys._xiaozhi_opus_loaded = True
            return True

    # Finally try system-installed opus.
    found = ctypes.util.find_library("opus")
    if found:
        try:
            ctypes.CDLL(found)
            sys._xiaozhi_opus_loaded = True
            return True
        except Exception:
            return False

    return False

