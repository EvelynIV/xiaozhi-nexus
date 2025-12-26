from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np


@dataclass
class StreamIASRnferencer:
    """
    Stub ASR:
    - input: iterator of PCM float32 (mono)
    - output: iterator of incremental transcripts ("1", "1 2", ...)
    """

    sample_rate: int = 16000
    seconds_per_digit: float = 1.0
    _digits: list[str] = field(default_factory=list, init=False)
    _counter: int = field(default=1, init=False)

    def __call__(self, audio_iter: Iterator[np.ndarray]) -> Iterator[str]:
        needed = int(self.sample_rate * self.seconds_per_digit)
        buffered = np.zeros((0,), dtype=np.float32)

        for chunk in audio_iter:
            chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
            if chunk.size == 0:
                continue
            buffered = np.concatenate([buffered, chunk], axis=0)

            while buffered.size >= needed:
                buffered = buffered[needed:]
                digit = str(self._counter % 10)
                self._counter += 1
                self._digits.append(digit)
                yield " ".join(self._digits)

