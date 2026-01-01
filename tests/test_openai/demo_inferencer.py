"""OpenAI Realtime ASR 同步推理器最简 Demo"""

import os
import numpy as np
import soundfile as sf
from dotenv import load_dotenv

from xiaozhi_nexus.inferencers.stream_asr import OpenAIRealtimeASRInferencer

load_dotenv()

AUDIO_FILE = os.getenv("TEST_AUDIO_FILE")
if not AUDIO_FILE:
    raise ValueError("环境变量 TEST_AUDIO_FILE 未设置")


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """加载音频文件为 float32 格式"""
    import librosa
    audio, sr = sf.read(path, dtype="float32")
    # 如果是多声道，取第一个声道
    if audio.ndim > 1:
        audio = audio[:, 0]
    # 重采样到目标采样率
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


def chunk_audio(audio: np.ndarray, chunk_size: int = 1600):
    """将音频切分成块"""
    for i in range(0, len(audio), chunk_size):
        yield audio[i : i + chunk_size]


if __name__ == "__main__":
    # 加载音频
    audio = load_audio(AUDIO_FILE)
    print(f"音频时长: {len(audio) / 16000:.2f}s")

    # 创建推理器
    inferencer = OpenAIRealtimeASRInferencer(verify_ssl=False)

    # 流式转录
    print("\n开始转录...")
    for text in inferencer(chunk_audio(audio)):
        print(text)

    print("\n\n完成!")
