from __future__ import annotations

from typing import Optional

import numpy as np


def get_is_speech(pcm_f32: np.ndarray) -> Optional[bool]:
    """Stub for VAD/speech detection. Returns None when unavailable."""
    return None
