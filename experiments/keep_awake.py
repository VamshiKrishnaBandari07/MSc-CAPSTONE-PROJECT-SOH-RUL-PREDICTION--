"""Prevent Windows sleep during long CPU training (lock screen OK)."""

from __future__ import annotations

import sys


class KeepSystemAwake:
    """Request ES_SYSTEM_REQUIRED so sleep does not stop training."""

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __init__(self) -> None:
        self._enabled = False
        self._kernel32 = None
        if sys.platform == "win32":
            import ctypes

            self._kernel32 = ctypes.windll.kernel32

    def __enter__(self) -> "KeepSystemAwake":
        if self._kernel32 is not None:
            self._kernel32.SetThreadExecutionState(
                self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED
            )
            self._enabled = True
            print("[Runtime] Keep-awake enabled (lock OK; avoid Sleep/Hibernate).")
        return self

    def __exit__(self, *args) -> None:
        if self._enabled and self._kernel32 is not None:
            self._kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)
            self._enabled = False
