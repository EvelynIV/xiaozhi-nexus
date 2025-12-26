from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from xiaozhi_nexus.utils.opus_loader import setup_opus


def _float32_to_int16(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)


@dataclass(frozen=True)
class OpusDecoder:
    sample_rate: int
    channels: int
    frame_size: int

    def __post_init__(self) -> None:
        if not setup_opus():
            raise RuntimeError(
                "libopus not found. Set XIAOZHI_OPUS_LIB to opus.dll or provide libs/libopus."
            )
        import opuslib

        object.__setattr__(
            self, "_decoder", opuslib.Decoder(self.sample_rate, self.channels)
        )

    def decode_to_float32(self, packet: bytes) -> np.ndarray:
        pcm_bytes = self._decoder.decode(packet, self.frame_size, decode_fec=False)
        pcm_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        pcm_f32 = pcm_i16.astype(np.float32) / 32768.0
        if self.channels > 1:
            pcm_f32 = pcm_f32.reshape(-1, self.channels).mean(axis=1)
        return pcm_f32


@dataclass(frozen=True)
class OpusEncoder:
    sample_rate: int
    channels: int
    frame_duration_ms: int = 20
    bitrate: int = 24000

    def __post_init__(self) -> None:
        if self.frame_duration_ms not in (10, 20, 40, 60):
            raise ValueError("frame_duration_ms must be one of 10/20/40/60")
        if not setup_opus():
            raise RuntimeError(
                "libopus not found. Set XIAOZHI_OPUS_LIB to opus.dll or provide libs/libopus."
            )
        import opuslib

        enc = opuslib.Encoder(
            self.sample_rate, self.channels, opuslib.APPLICATION_AUDIO
        )
        enc.bitrate = int(self.bitrate)
        object.__setattr__(self, "_encoder", enc)

    @property
    def frame_size(self) -> int:
        return int(self.sample_rate * (self.frame_duration_ms / 1000))

    def encode_pcm_float32(self, pcm: np.ndarray) -> Iterator[bytes]:
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if self.channels != 1:
            raise ValueError("Only mono PCM supported by this stub encoder")

        frame_size = self.frame_size
        total = int(pcm.shape[0])
        idx = 0
        while idx < total:
            frame = pcm[idx : idx + frame_size]
            if frame.shape[0] < frame_size:
                frame = np.pad(frame, (0, frame_size - frame.shape[0]))
            idx += frame_size
            pcm_i16 = _float32_to_int16(frame)
            packet = self._encoder.encode(pcm_i16.tobytes(), frame_size)
            yield packet
