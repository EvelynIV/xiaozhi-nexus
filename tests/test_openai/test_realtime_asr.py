"""
测试 OpenAI Realtime ASR 推理器

运行方式:
    python -m pytest tests/test_openai/test_realtime_asr.py -v
    或直接运行:
    python tests/test_openai/test_realtime_asr.py
"""

from __future__ import annotations

import os
import asyncio

import numpy as np
import soundfile as sf
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

TEST_AUDIO_FILE = os.getenv("TEST_AUDIO_FILE")
if not TEST_AUDIO_FILE:
    raise ValueError("环境变量 TEST_AUDIO_FILE 未设置")


def load_audio_as_float32(file_path: str, sample_rate: int = 16000) -> np.ndarray:
    """读取音频文件并转换为 float32 格式"""
    import librosa

    audio, sr = sf.read(file_path, dtype="float32")
    # 如果是多声道，取第一个声道
    if audio.ndim > 1:
        audio = audio[:, 0]
    print(f"Loaded audio: {sr=} shape={audio.shape}")
    print(f"Duration: {len(audio) / sr:.2f}s")

    # 重采样到目标采样率
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

    return audio


def audio_chunk_iterator(audio_data: np.ndarray, chunk_size: int = 1600):
    """将音频数据切分成块的迭代器"""
    for i in range(0, len(audio_data), chunk_size):
        yield audio_data[i : i + chunk_size]


async def async_audio_chunk_iterator(audio_data: np.ndarray, chunk_size: int = 1600):
    """异步音频块迭代器"""
    for i in range(0, len(audio_data), chunk_size):
        yield audio_data[i : i + chunk_size]
        await asyncio.sleep(0.01)  # 模拟实时音频流


async def test_async_inferencer():
    """测试异步推理器"""
    from xiaozhi_nexus.inferencers.stream_asr import OpenAIRealtimeASRInferencerAsync

    if not os.path.exists(TEST_AUDIO_FILE):
        print(f"Audio file not found: {TEST_AUDIO_FILE}")
        return

    print("Loading audio...")
    audio_data = load_audio_as_float32(TEST_AUDIO_FILE)
    print(f"Audio shape: {audio_data.shape}, dtype: {audio_data.dtype}")

    # 创建推理器（禁用 SSL 验证用于测试）
    inferencer = OpenAIRealtimeASRInferencerAsync(
        verify_ssl=False,
    )

    print("\nStarting transcription...")
    async for transcript in inferencer.transcribe(
        async_audio_chunk_iterator(audio_data),
    ):
        print(f"Transcript: {transcript}")

    print("\nDone!")


def test_sync_inferencer():
    """测试同步推理器"""
    from xiaozhi_nexus.inferencers.stream_asr import OpenAIRealtimeASRInferencer

    if not os.path.exists(TEST_AUDIO_FILE):
        print(f"Audio file not found: {TEST_AUDIO_FILE}")
        return

    print("Loading audio...")
    audio_data = load_audio_as_float32(TEST_AUDIO_FILE)
    print(f"Audio shape: {audio_data.shape}, dtype: {audio_data.dtype}")

    # 创建推理器（禁用 SSL 验证用于测试）
    inferencer = OpenAIRealtimeASRInferencer(
        verify_ssl=False,
    )

    print("\nStarting transcription...")
    for transcript in inferencer(audio_chunk_iterator(audio_data)):
        print(f"Transcript: {transcript}")

    print("\nDone!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "sync":
        test_sync_inferencer()
    else:
        asyncio.run(test_async_inferencer())
