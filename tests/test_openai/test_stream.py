import os
from openai import OpenAI
import httpx
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "your-api-key"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    http_client=httpx.Client(
        verify=False  # 🔴 关键:关闭证书校验
    ),
)

audio_file_path = "data-bin/huaqiang/403369728_nb2-1-30280_left_16k.wav"

with open(audio_file_path, "rb") as audio_file:
    stream = client.audio.transcriptions.create(
        file=audio_file,
        model="gpt-4o-transcribe",
        stream=True,          # 👈 关键
        language="zh",
    )

    print("流式识别结果：")
    for event in stream:
        # 兼容 OpenAI / vLLM / FastAPI 实现
        if hasattr(event, "text") and event.text:
            print(event.text, end="", flush=True)