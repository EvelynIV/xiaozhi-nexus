from __future__ import annotations

import time
from typing import Optional

import numpy as np

_START_TS = time.monotonic()
_PERIOD_SEC = 6.0
_TRUE_SEC = 1.0
_FALSE_SEC = _PERIOD_SEC - _TRUE_SEC


def get_is_speech(pcm_f32: np.ndarray) -> Optional[bool]:
    """Stub for VAD/speech detection with a repeating false/true cadence."""
    elapsed = time.monotonic() - _START_TS
    phase = elapsed % _PERIOD_SEC
    return phase >= _FALSE_SEC
