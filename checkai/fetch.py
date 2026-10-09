"""Tải một trang web và trích phần nội dung chính của bài viết."""

from __future__ import annotations

import ipaddress
import socket
import urllib.parse
import urllib.request
from html.parser import HTMLParser

MAX_PAGE_BYTES = 5 * 1024 * 1024


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
        raise ValueError("Không phân giải được tên miền.") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise ValueError("Không cho phép truy cập địa chỉ nội bộ.")


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
        raise ValueError("URL không hợp lệ (chỉ hỗ trợ http/https).")
    _check_public_host(parsed.hostname)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 CheckAI"})
    try:
        with _opener.open(req, timeout=15) as r:
            raw = r.read(MAX_PAGE_BYTES)
            charset = r.headers.get_content_charset() or "utf-8"
    except ValueError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Không tải được trang: {e}") from e
    parser = _TextExtractor()
    parser.feed(raw.decode(charset, errors="ignore"))
    lines = [" ".join(line.split()) for line in "".join(parser.parts).split("\n")]
    # Giữ các dòng đủ dài - thường là nội dung chính của bài viết.
    paras = [line for line in lines if len(line.split()) >= 8]
    return "\n\n".join(paras)
