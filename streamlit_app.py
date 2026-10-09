"""Giao diện Streamlit cho CheckAI. Chạy: streamlit run streamlit_app.py"""

from __future__ import annotations

import html
import os
import re

import streamlit as st

# Đưa Secrets của Streamlit (SAPLING_API_KEY, SIGHTENGINE_API_USER...) vào biến môi trường
# để checkai.providers đọc được.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, (str, int, float, bool)):
            os.environ.setdefault(_k, str(_v))
except Exception:  # noqa: BLE001 - không có secrets.toml
    pass

from checkai import providers  # noqa: E402
from checkai.documents import extract_text  # noqa: E402
from checkai.fetch import fetch_article  # noqa: E402
from checkai.image_detector import analyze_image  # noqa: E402
from checkai.scoring import Result  # noqa: E402
from checkai.text_detector import analyze_text  # noqa: E402

MAX_TEXT_CHARS = 100_000
MAX_IMAGE_MB = 20
MAX_DOC_MB = 20

st.set_page_config(page_title="CheckAI", page_icon="🔍", layout="centered")


def color(p: float) -> str:
    return "#d64545" if p >= 65 else "#d99a1e" if p >= 40 else "#2e9d5b"


@st.cache_data(show_spinner=False, max_entries=50)
def cached_fetch(url: str) -> str:
    return fetch_article(url)


def render(r: Result) -> None:
    p = r.ai_probability
    c = color(p)
    st.markdown(
        f"""<div style="display:flex;gap:20px;align-items:center;flex-wrap:wrap;margin:8px 0 16px">
        <div style="position:relative;width:130px;height:130px;flex:none">
          <div style="position:absolute;inset:0;border-radius:50%;
               background:conic-gradient({c} {p * 3.6}deg, rgba(128,128,128,.25) 0);
               -webkit-mask:radial-gradient(farthest-side, transparent 78%, #000 79%);
               mask:radial-gradient(farthest-side, transparent 78%, #000 79%)"></div>
          <div style="position:absolute;inset:0;display:grid;place-items:center;
               font-size:26px;font-weight:700">{p}%</div></div>
        <div><div style="font-size:21px;font-weight:600;color:{c}">{html.escape(r.verdict)}</div>
        <div>Khả năng do AI tạo: <b>{p}%</b> · Độ tin cậy: <b>{html.escape(r.confidence)}</b></div></div>
        </div>""",
        unsafe_allow_html=True,
    )
    for n in r.notes:
        st.caption("ℹ️ " + n)

    st.subheader("Chi tiết các dấu hiệu")
    for s in r.signals:
        v = round(s.score * 100)
        tag = "" if s.source == "heuristic" else f" `{s.source}`"
        st.markdown(f"**{s.name}**{tag} — {v}% giống AI")
        st.progress(v / 100)
        st.caption(s.detail)

    if r.highlights:
        st.subheader("Câu có dấu hiệu AI")
        items = []
        for h in r.highlights[:30]:
            t = html.escape(h["text"])
            for w in h["reasons"]:
                t = re.sub(re.escape(html.escape(w)), lambda m: f"<mark>{m.group(0)}</mark>", t, flags=re.I)
            items.append(f"<li>{t}</li>")
        st.markdown("<ul>" + "".join(items) + "</ul>", unsafe_allow_html=True)

    if r.meta:
        with st.expander("Thông tin"):
            st.json(r.meta)


st.title("🔍 CheckAI")
st.caption("Ước lượng khả năng bài viết, nội dung hoặc hình ảnh được tạo bởi AI.")

tab_text, tab_doc, tab_image = st.tabs(["📝 Văn bản / Bài viết", "📄 Tệp Word / PDF", "🖼️ Hình ảnh"])

with tab_text:
    text = st.text_area("Nội dung", height=240,
                        placeholder="Dán nội dung bài viết vào đây (nên từ 150 từ trở lên)...")
    url = st.text_input("...hoặc URL bài viết", placeholder="https://...")
    st.caption(f"{len(re.findall(r'\w+', text))} từ")
    if st.button("Kiểm tra", type="primary", key="go_text"):
        try:
            with st.spinner("Đang phân tích..."):
                if not text.strip() and url.strip():
                    text = cached_fetch(url.strip())
                    with st.expander(f"Nội dung trích từ URL ({len(text)} ký tự)"):
                        st.write(text[:5000])
                if not text.strip():
                    raise ValueError("Vui lòng nhập văn bản hoặc URL bài viết.")
                result = analyze_text(text[:MAX_TEXT_CHARS])
            render(result)
        except ValueError as e:
            st.error(str(e))

with tab_doc:
    doc = st.file_uploader("Chọn tệp Word hoặc PDF", type=["docx", "pdf", "txt", "md"], key="doc")
    st.caption("Hỗ trợ .docx, .pdf (có lớp chữ, không phải ảnh scan), .txt, .md — tối đa 20MB.")
    if doc is not None and st.button("Kiểm tra tệp", type="primary", key="go_doc"):
        try:
            data = doc.getvalue()
            if len(data) > MAX_DOC_MB * 1024 * 1024:
                raise ValueError(f"Tệp quá lớn (tối đa {MAX_DOC_MB}MB).")
            with st.spinner("Đang đọc tệp và phân tích..."):
                content, info = extract_text(data, doc.name)
                result = analyze_text(content[:MAX_TEXT_CHARS])
                result.meta.update(info)
            with st.expander(f"Nội dung trích từ tệp ({len(content)} ký tự)"):
                st.write(content[:5000] + ("…" if len(content) > 5000 else ""))
            render(result)
        except ValueError as e:
            st.error(str(e))

with tab_image:
    up = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png", "webp", "bmp", "tiff", "gif"])
    if up is not None:
        st.image(up)
        if st.button("Kiểm tra ảnh", type="primary", key="go_image"):
            data = up.getvalue()
            if len(data) > MAX_IMAGE_MB * 1024 * 1024:
                st.error(f"Ảnh quá lớn (tối đa {MAX_IMAGE_MB}MB).")
            else:
                try:
                    with st.spinner("Đang phân tích..."):
                        result = analyze_image(data, up.name)
                    render(result)
                except ValueError as e:
                    st.error(str(e))

st.divider()
d = providers.status()
on = [n for n, k in (("mô hình cục bộ", "local_models"), ("Sapling", "sapling"), ("Sightengine", "sightengine")) if d[k]]
st.caption(
    "Lưu ý: không công cụ nào phát hiện AI chính xác 100%. Kết quả là ước lượng, không nên dùng làm bằng chứng duy nhất. "
    + ("Đang dùng thêm: " + ", ".join(on) + "." if on else "Đang dùng chế độ heuristic (chưa bật mô hình/API).")
)
