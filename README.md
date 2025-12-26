# xiaozhi-nexus

一个用于对齐 `simple-xiaozhi` WebSocket 协议的 FastAPI 服务端最小实现（ASR/TTS stub）。

## 运行服务端

```bash
poetry install
poetry run xiaozhi-nexus serve --host 127.0.0.1 --port 8000
```

WebSocket 地址：`ws://127.0.0.1:8000/ws`

> Windows 如遇到 `Could not find Opus library`：设置 `XIAOZHI_OPUS_LIB` 指向 `opus.dll`，或确保工作区存在 `simple-xiaozhi/libs/libopus/win/x64/opus.dll`。

## 使用 simple-xiaozhi 的 SimpleClient 验证

在 `D:\workspace\simple-xiaozhi` 下设置环境变量（示例）：

```powershell
$env:XIAOZHI_WS_URL="ws://127.0.0.1:8000/ws"
$env:XIAOZHI_ACCESS_TOKEN="dev"
$env:XIAOZHI_DEVICE_ID="dev-device"
$env:XIAOZHI_CLIENT_ID="dev-client"
poetry run python .\tests\simple_client.py
```

## Stub 行为

- ASR：累计收到约 1 秒音频（按 `hello.audio_params.sample_rate` 计算）输出一个数字，增量文本为 `"1" / "1 2" / "1 2 3" ...`
- TTS：每个 token 生成一段正弦波，频率从 440Hz 开始，每个 token +20Hz，并以 Opus 二进制帧下行
