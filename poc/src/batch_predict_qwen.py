from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from config import LABELS, ROUTER_BASE_MODEL
from qwen_router import import_training_deps, load_qwen_router


def _split_numbered_line(line: str) -> list[str]:
    """Split a line like '23. foo 24. bar' into ['foo', 'bar']."""
    # Only treat 1-2 digit list markers as inline item separators.
    # This avoids false splits like "port 443. Describe ...".
    matches = list(re.finditer(r"(?<!\d)\d{1,2}\.\s+", line))
    if not matches:
        return [line.strip()]

    chunks: list[str] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
        text = line[start:end].strip()
        if text:
            chunks.append(text)
    return chunks


def _extract_prompts_from_text(content: str) -> list[str]:
    prompts: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "---" or line.startswith("#"):
            continue
        if re.match(r"^\d+\.\s+.*\(\d+\s+items\)\s*$", line):
            continue

        if re.match(r"^\d{1,2}\.\s+", line):
            chunks = _split_numbered_line(line)
            prompts.extend(chunks)
            continue

        if prompts:
            prompts[-1] = f"{prompts[-1]} {line}"
        else:
            prompts.append(line)

    cleaned: list[str] = []
    for prompt in prompts:
        normalized = re.sub(r"\s+", " ", prompt).strip()
        if normalized:
            cleaned.append(normalized)
    return cleaned


def _extract_prompts(path: Path, fmt: str) -> list[str]:
    content = path.read_text(encoding="utf-8")

    if fmt in {"auto", "json"}:
        try:
            payload = json.loads(content)
            if isinstance(payload, list):
                return [str(item).strip() for item in payload if str(item).strip()]
            if isinstance(payload, dict):
                if isinstance(payload.get("records"), list):
                    prompts = []
                    for row in payload["records"]:
                        if isinstance(row, dict):
                            text = row.get("text") or row.get("prompt") or row.get("question")
                            if isinstance(text, str) and text.strip():
                                prompts.append(text.strip())
                    if prompts:
                        return prompts
                if isinstance(payload.get("prompts"), list):
                    return [str(item).strip() for item in payload["prompts"] if str(item).strip()]
                if isinstance(payload.get("questions"), list):
                    return [str(item).strip() for item in payload["questions"] if str(item).strip()]
        except json.JSONDecodeError:
            if fmt == "json":
                raise

    if fmt in {"auto", "jsonl"}:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        jsonl_prompts: list[str] = []
        ok = True
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                ok = False
                break
            if isinstance(row, dict):
                text = row.get("text") or row.get("prompt") or row.get("question")
                if isinstance(text, str) and text.strip():
                    jsonl_prompts.append(text.strip())
            elif isinstance(row, str) and row.strip():
                jsonl_prompts.append(row.strip())
        if ok and jsonl_prompts:
            return jsonl_prompts
        if fmt == "jsonl":
            raise ValueError("JSONLとして読み取れませんでした")

    return _extract_prompts_from_text(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch prediction with TuneRouter Qwen adapter")
    parser.add_argument("--input", required=True, help="Input file path (text/markdown/json/jsonl)")
    parser.add_argument("--output", default="poc/artifacts/predictions_batch.json", help="Output JSON file")
    parser.add_argument("--input-format", choices=["auto", "text", "json", "jsonl"], default="auto")
    parser.add_argument("--adapter", default="poc/artifacts/qwen-router-lora")
    parser.add_argument("--base-model", default=ROUTER_BASE_MODEL)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="Only run first N prompts")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    prompts = _extract_prompts(input_path, args.input_format)
    if args.limit is not None:
        prompts = prompts[: args.limit]

    if not prompts:
        raise ValueError("質問文を抽出できませんでした。入力形式を確認してください。")

    deps = import_training_deps()
    tokenizer, model = load_qwen_router(args.base_model, Path(args.adapter), deps)
    torch = deps["torch"]
    label_names = list(LABELS)

    records: list[dict] = []
    with torch.no_grad():
        model.eval()
        for index, text in enumerate(prompts, start=1):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            logits = model(**inputs).logits[0]
            probabilities = torch.softmax(logits, dim=-1).detach().cpu().tolist()
            best_index = max(range(len(probabilities)), key=probabilities.__getitem__)
            label = label_names[best_index]
            records.append(
                {
                    "index": index,
                    "text": text,
                    "predicted_label": label,
                    "target_model": LABELS[label]["target_model"],
                    "confidence": float(probabilities[best_index]),
                    "scores": {name: float(probabilities[i]) for i, name in enumerate(label_names)},
                }
            )

    counts = {name: 0 for name in label_names}
    for row in records:
        counts[row["predicted_label"]] += 1

    payload = {
        "metadata": {
            "input": str(input_path),
            "adapter": str(args.adapter),
            "base_model": args.base_model,
            "total": len(records),
            "label_counts": counts,
        },
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"saved: {output_path}")
    print(f"total: {len(records)}")
    print("label_counts:")
    for label in label_names:
        print(f"  {label}: {counts[label]}")


if __name__ == "__main__":
    main()
