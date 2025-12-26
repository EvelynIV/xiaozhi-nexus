from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np

from xiaozhi_nexus.audio.opus import OpusEncoder
from xiaozhi_nexus.stubs.asr import StreamIASRnferencer
from xiaozhi_nexus.stubs.tts import SineWaveTTS


@dataclass
class StreamSession:
    publish_json: Callable[[dict], None]
    publish_bytes: Callable[[bytes], None]
    inferencer: StreamIASRnferencer
    tts: SineWaveTTS
    encoder: OpusEncoder
    input_maxsize: int = 200

    def __post_init__(self) -> None:
        self._audio_q: queue.Queue[np.ndarray | None] = queue.Queue(
            maxsize=self.input_maxsize
        )
        self._thread: threading.Thread | None = None
        self._running = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        while True:
            try:
                self._audio_q.put(None, timeout=0.05)
                break
            except queue.Full:
                try:
                    self._audio_q.get_nowait()
                except queue.Empty:
                    break
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def push_audio(self, pcm_f32: np.ndarray) -> None:
        if not self._running.is_set():
            return
        try:
            self._audio_q.put_nowait(np.asarray(pcm_f32, dtype=np.float32))
        except queue.Full:
            pass

    def _audio_iter(self):
        while self._running.is_set():
            item = self._audio_q.get()
            if item is None:
                break
            yield item

    def _worker(self) -> None:
        last_transcript: str | None = None
        for transcript in self.inferencer(self._audio_iter()):
            if not self._running.is_set():
                break

            self.publish_json({"type": "stt", "text": transcript})
            token = transcript.split()[-1] if transcript else ""

            self.publish_json({"type": "tts", "state": "start", "text": token})
            for pcm in self.tts.synthesize(token):
                for packet in self.encoder.encode_pcm_float32(pcm):
                    self.publish_bytes(packet)
            self.publish_json({"type": "tts", "state": "stop"})

            if transcript != last_transcript:
                self.publish_json({"type": "llm", "emotion": "neutral"})
                last_transcript = transcript
