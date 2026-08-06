# OpenAI Router Shim

## 起動方法

```bash
uv run --project ./sever --system-certs python ./sever/src/openai_classifier_server.py \
  --host 127.0.0.1 \
  --port 18001 \
  --model-name router \
  --adapter ./poc/artifacts/qwen-router-lora
```

起動後、Router endpoint は次を指定する。

```text
http://127.0.0.1:18001/v1
```
