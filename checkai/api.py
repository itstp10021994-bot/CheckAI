"""Web API + giao diện. Chạy: uvicorn checkai.api:app --reload"""

from __future__ import annotations

import ipaddress
import socket
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import __version__, providers
from .image_detector import analyze_image
from .text_detector import analyze_text

STATIC = Path(__file__).resolve().parent.parent / "static"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_TEXT_CHARS = 100_000
MAX_PAGE_BYTES = 5 * 1024 * 1024

app = FastAPI(title="CheckAI", version=__version__,
              description="Ước lượng % khả năng văn bản / hình ảnh được tạo bởi AI.")


class TextIn(BaseModel):
    text: str | None = None
    url: str | None = None


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg"}
    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "article", "section", "tr"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def _check_public_host(host: str) -> None:
    """Chặn truy cập địa chỉ nội bộ (SSRF)."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise HTTPException(400, "Không phân giải được tên miền.") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise HTTPException(400, "Không cho phép truy cập địa chỉ nội bộ.")


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        host = urllib.parse.urlparse(newurl).hostname
        if host:
            _check_public_host(host)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_opener = urllib.request.build_opener(_SafeRedirect)


def fetch_article(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise HTTPException(400, "URL không hợp lệ (chỉ hỗ trợ http/https).")
    _check_public_host(parsed.hostname)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 CheckAI"})
    try:
        with _opener.open(req, timeout=15) as r:
            raw = r.read(MAX_PAGE_BYTES)
            charset = r.headers.get_content_charset() or "utf-8"
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Không tải được trang: {e}") from e
    parser = _TextExtractor()
    parser.feed(raw.decode(charset, errors="ignore"))
    lines = [" ".join(line.split()) for line in "".join(parser.parts).split("\n")]
    # Giữ các dòng đủ dài - thường là nội dung chính của bài viết.
    paras = [line for line in lines if len(line.split()) >= 8]
    return "\n\n".join(paras)


@app.get("/api/health")
def health():
    return {"status": "ok", "version": __version__, "detectors": providers.status()}


@app.post("/api/text")
def check_text(body: TextIn):
    text = body.text
    source = None
    if body.url and not (text and text.strip()):
        text = fetch_article(body.url)
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


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
