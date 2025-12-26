from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np


@dataclass
class SineWaveTTS:
    sample_rate: int = 24000
    seconds_per_token: float = 0.2
    start_hz: float = 440.0
    step_hz: float = 20.0
    amplitude: float = 0.2
    _token_index: int = field(default=0, init=False)

    def synthesize(self, text: str) -> Iterator[np.ndarray]:
        tokens = [t for t in str(text).split() if t]
        for _token in tokens:
            freq = self.start_hz + self.step_hz * self._token_index
            self._token_index += 1
            yield self._sine(freq_hz=freq)

    def _sine(self, freq_hz: float) -> np.ndarray:
        n = int(self.sample_rate * self.seconds_per_token)
        t = np.arange(n, dtype=np.float32) / float(self.sample_rate)
        wave = np.sin(2.0 * np.pi * float(freq_hz) * t) * float(self.amplitude)
        return wave.astype(np.float32)

