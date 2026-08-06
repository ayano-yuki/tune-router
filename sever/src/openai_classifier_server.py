from __future__ import annotations

import argparse
import json
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
POC_SRC = REPO_ROOT / "poc" / "src"
DEFAULT_ADAPTER = REPO_ROOT / "poc" / "artifacts" / "qwen-router-lora"

if str(POC_SRC) not in sys.path:
    sys.path.insert(0, str(POC_SRC))

from config import LABELS, ROUTER_BASE_MODEL  # noqa: E402
from qwen_router import import_inference_deps, load_qwen_router  # noqa: E402


MAX_BODY_BYTES = 1 << 20


class RouterClassifier:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model_name = args.model_name
        self.max_length = args.max_length
        self.deps = import_inference_deps()
        self.tokenizer, self.model = load_qwen_router(
            args.base_model,
            resolve_path(args.adapter),
            self.deps,
        )
        self.torch = self.deps["torch"]
        self.labels = list(LABELS)

        device = resolve_device(self.torch, args.device)
        self.model.to(device)
        self.model.eval()

    def predict(self, text: str) -> tuple[str, float, dict[str, float]]:
        with self.torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
            logits = self.model(**inputs).logits[0]
            probabilities = self.torch.softmax(logits, dim=-1).detach().cpu().tolist()

        best_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        scores = {label: float(probabilities[index]) for index, label in enumerate(self.labels)}
        return self.labels[best_index], float(probabilities[best_index]), scores


class OpenAIClassifierHandler(BaseHTTPRequestHandler):
    classifier: RouterClassifier

    server_version = "TuneRouterOpenAIShim/0.1"

    def do_GET(self) -> None:
        if request_path(self.path) not in {"/v1/models", "/models"}:
            self.write_error(HTTPStatus.NOT_FOUND, "not_found", "unknown endpoint")
            return

        now = int(time.time())
        self.write_json(
            HTTPStatus.OK,
            {
                "object": "list",
                "data": [
                    {
                        "id": self.classifier.model_name,
                        "object": "model",
                        "created": now,
                        "owned_by": "local",
                    }
                ],
            },
        )

    def do_POST(self) -> None:
        if request_path(self.path) not in {"/v1/chat/completions", "/chat/completions"}:
            self.write_error(HTTPStatus.NOT_FOUND, "not_found", "unknown endpoint")
            return

        try:
            payload = self.read_json_body()
            message = extract_last_user_message(payload)
            label, confidence, scores = self.classifier.predict(message)
        except ValueError as exc:
            self.write_error(HTTPStatus.BAD_REQUEST, "invalid_request_error", str(exc))
            return
        except Exception as exc:
            self.write_error(HTTPStatus.INTERNAL_SERVER_ERROR, "server_error", str(exc))
            return

        created = int(time.time())
        self.write_json(
            HTTPStatus.OK,
            {
                "id": f"chatcmpl-router-{created}",
                "object": "chat.completion",
                "created": created,
                "model": self.classifier.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": label},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 1,
                    "total_tokens": 1,
                },
                "router": {
                    "confidence": round(confidence, 6),
                    "scores": {name: round(score, 6) for name, score in scores.items()},
                },
            },
        )

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def read_json_body(self) -> dict[str, Any]:
        length_header = self.headers.get("Content-Length")
        if not length_header:
            raise ValueError("Content-Length is required")
        try:
            length = int(length_header)
        except ValueError as exc:
            raise ValueError("Content-Length must be an integer") from exc
        if length > MAX_BODY_BYTES:
            raise ValueError("request body is too large")

        body = self.rfile.read(length)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError(f"request body must be JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def write_error(self, status: HTTPStatus, error_type: str, message: str) -> None:
        self.write_json(status, {"error": {"message": message, "type": error_type}})


def extract_last_user_message(payload: dict[str, Any]) -> str:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages must be an array")

    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            text = extract_text_parts(content)
            if text:
                return text

    raise ValueError("messages must contain a non-empty user message")


def extract_text_parts(content: list[Any]) -> str:
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            text = item["text"].strip()
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def request_path(raw_path: str) -> str:
    path = urlparse(raw_path).path.rstrip("/")
    return path or "/"


def resolve_device(torch_module: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (Path.cwd() / candidate).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenAI-compatible server for the Qwen router classifier")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18001)
    parser.add_argument("--model-name", default="router")
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--base-model", default=ROUTER_BASE_MODEL)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    OpenAIClassifierHandler.classifier = RouterClassifier(args)
    server = ThreadingHTTPServer((args.host, args.port), OpenAIClassifierHandler)
    print(f"OpenAI-compatible router classifier listening on http://{args.host}:{args.port}/v1")
    print(f"model={args.model_name} labels={', '.join(LABELS)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
