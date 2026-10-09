"""Các bộ phát hiện tuỳ chọn: mô hình Hugging Face cục bộ và API bên ngoài.

Bật bằng biến môi trường:
  CHECKAI_LOCAL_MODELS=1        dùng mô hình Hugging Face (cần `pip install -r requirements-ml.txt`)
  CHECKAI_TEXT_MODEL=...        mô hình phân loại văn bản (mặc định: Hello-SimpleAI/chatgpt-detector-roberta)
  CHECKAI_IMAGE_MODEL=...       mô hình phân loại ảnh (mặc định: umm-maybe/AI-image-detector)
  SAPLING_API_KEY=...           API phát hiện văn bản AI của Sapling (https://sapling.ai)
  SIGHTENGINE_API_USER=... / SIGHTENGINE_API_SECRET=...   API phát hiện ảnh AI của Sightengine
"""

from __future__ import annotations

import io
import json
import logging
import os
import urllib.parse
import urllib.request
import uuid
from functools import lru_cache

from .scoring import Signal

log = logging.getLogger(__name__)

TIMEOUT = 20
AI_LABEL_HINTS = ("ai", "fake", "artificial", "generated", "chatgpt", "gpt", "machine", "label_1")
HUMAN_LABEL_HINTS = ("human", "real", "label_0")


def _local_enabled() -> bool:
    return os.getenv("CHECKAI_LOCAL_MODELS", "").lower() in ("1", "true", "yes")


@lru_cache(maxsize=2)
def _pipeline(task: str, model: str):
    from transformers import pipeline  # import chậm, chỉ khi cần

    return pipeline(task, model=model)


def _ai_prob_from_labels(preds: list[dict]) -> float | None:
    """Chuyển đầu ra [{label, score}, ...] thành xác suất AI."""
    for p in preds:
        if any(h in p["label"].lower() for h in AI_LABEL_HINTS):
            return float(p["score"])
    for p in preds:
        if any(h in p["label"].lower() for h in HUMAN_LABEL_HINTS):
            return 1.0 - float(p["score"])
    return None


def _post(url: str, data: bytes, headers: dict) -> dict:
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode())


# ---------------------------------------------------------------- văn bản

def text_model_signals(text: str) -> list[Signal]:
    out: list[Signal] = []
    if _local_enabled():
        model = os.getenv("CHECKAI_TEXT_MODEL", "Hello-SimpleAI/chatgpt-detector-roberta")
        try:
            clf = _pipeline("text-classification", model)
            preds = clf(text[:4000], truncation=True, top_k=None)
            p = _ai_prob_from_labels(preds)
            if p is not None:
                out.append(Signal("Mô hình cục bộ", p, 1.0, f"{model}: {p * 100:.1f}% AI", "model"))
        except Exception as e:  # noqa: BLE001
            log.warning("Text model failed: %s", e)

    key = os.getenv("SAPLING_API_KEY")
    if key:
        try:
            r = _post("https://api.sapling.ai/api/v1/aidetect",
                      json.dumps({"key": key, "text": text[:200000]}).encode(),
                      {"Content-Type": "application/json"})
            p = float(r["score"])
            out.append(Signal("Sapling AI Detector", p, 1.0, f"{p * 100:.1f}% AI", "api"))
        except Exception as e:  # noqa: BLE001
            log.warning("Sapling failed: %s", e)
    return out


# ---------------------------------------------------------------- hình ảnh

def image_model_signals(data: bytes, filename: str = "image") -> list[Signal]:
    out: list[Signal] = []
    if _local_enabled():
        model = os.getenv("CHECKAI_IMAGE_MODEL", "umm-maybe/AI-image-detector")
        try:
            from PIL import Image

            clf = _pipeline("image-classification", model)
            preds = clf(Image.open(io.BytesIO(data)).convert("RGB"))
            p = _ai_prob_from_labels(preds)
            if p is not None:
                out.append(Signal("Mô hình cục bộ", p, 1.0, f"{model}: {p * 100:.1f}% AI", "model"))
        except Exception as e:  # noqa: BLE001
            log.warning("Image model failed: %s", e)

    user, secret = os.getenv("SIGHTENGINE_API_USER"), os.getenv("SIGHTENGINE_API_SECRET")
    if user and secret:
        try:
            boundary = uuid.uuid4().hex
            fields = {"models": "genai", "api_user": user, "api_secret": secret}
            body = b"".join(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
                for k, v in fields.items())
            body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"media\"; "
                     f"filename=\"{urllib.parse.quote(filename)}\"\r\n"
                     "Content-Type: application/octet-stream\r\n\r\n").encode()
            body += data + f"\r\n--{boundary}--\r\n".encode()
            r = _post("https://api.sightengine.com/1.0/check.json", body,
                      {"Content-Type": f"multipart/form-data; boundary={boundary}"})
            p = float(r["type"]["ai_generated"])
            out.append(Signal("Sightengine GenAI", p, 1.0, f"{p * 100:.1f}% AI", "api"))
        except Exception as e:  # noqa: BLE001
            log.warning("Sightengine failed: %s", e)
    return out


def status() -> dict:
    return {
        "local_models": _local_enabled(),
        "sapling": bool(os.getenv("SAPLING_API_KEY")),
        "sightengine": bool(os.getenv("SIGHTENGINE_API_USER") and os.getenv("SIGHTENGINE_API_SECRET")),
    }
