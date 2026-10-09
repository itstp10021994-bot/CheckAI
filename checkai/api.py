"""Web API + giao diện. Chạy: uvicorn checkai.api:app --reload"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import __version__, providers
from .documents import extract_text
from .fetch import fetch_article
from .image_detector import analyze_image
from .text_detector import analyze_text

STATIC = Path(__file__).resolve().parent.parent / "static"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_TEXT_CHARS = 100_000

app = FastAPI(title="CheckAI", version=__version__,
              description="Ước lượng % khả năng văn bản / hình ảnh được tạo bởi AI.")


class TextIn(BaseModel):
    text: str | None = None
    url: str | None = None


@app.get("/api/health")
def health():
    return {"status": "ok", "version": __version__, "detectors": providers.status()}


@app.post("/api/text")
def check_text(body: TextIn):
    text = body.text
    source = None
    if body.url and not (text and text.strip()):
        try:
            text = fetch_article(body.url)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        source = body.url
    if not text or not text.strip():
        raise HTTPException(400, "Vui lòng nhập văn bản hoặc URL bài viết.")
    try:
        result = analyze_text(text[:MAX_TEXT_CHARS]).to_dict()
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if source:
        result["meta"] = {"source": source, "extracted_chars": len(text)}
    return result


@app.post("/api/image")
async def check_image(file: UploadFile = File(...)):
    data = await file.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Ảnh quá lớn (tối đa 20MB).")
    try:
        return analyze_image(data, file.filename or "image").to_dict()
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.post("/api/document")
async def check_document(file: UploadFile = File(...)):
    data = await file.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Tệp quá lớn (tối đa 20MB).")
    try:
        text, info = extract_text(data, file.filename or "")
        result = analyze_text(text[:MAX_TEXT_CHARS])
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    result.meta.update(info)
    return result.to_dict()


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
