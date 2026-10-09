import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, PngImagePlugin

from checkai.api import app
from checkai.documents import extract_text
from checkai.image_detector import analyze_image
from checkai.text_detector import analyze_text
from samples import AI_VI, HUMAN_VI, make_pdf

client = TestClient(app)


def _png(text_chunks=None, size=(1024, 1024)):
    arr = np.tile(np.linspace(0, 255, size[0]), (size[1], 1)).astype(np.uint8)
    info = PngImagePlugin.PngInfo()
    for k, v in (text_chunks or {}).items():
        info.add_text(k, v)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG", pnginfo=info)
    return buf.getvalue()


def _camera_jpeg():
    rng = np.random.default_rng(0)
    img = Image.fromarray(np.clip(rng.normal(128, 20, (900, 1200, 3)), 0, 255).astype(np.uint8))
    exif = img.getexif()
    exif[271], exif[272] = "Canon", "EOS 80D"
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    return buf.getvalue()


def test_ai_text_scores_higher_than_human_text():
    ai, human = analyze_text(AI_VI), analyze_text(HUMAN_VI)
    assert ai.ai_probability > 65
    assert human.ai_probability < 40
    assert ai.highlights


def test_short_text_is_low_confidence_and_near_neutral():
    r = analyze_text("Tóm lại, AI đóng vai trò quan trọng.")
    assert r.confidence == "Thấp"
    assert 35 <= r.ai_probability <= 65


def test_stable_diffusion_metadata_is_detected():
    r = analyze_image(_png({"parameters": "a cat, Steps: 20, Sampler: Euler"}))
    assert r.ai_probability >= 95
    assert r.confidence == "Cao"


def test_iptc_ai_marker_in_xmp_is_detected():
    xmp = "<x:xmpmeta><Iptc4xmpExt:DigitalSourceType>http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia</Iptc4xmpExt:DigitalSourceType></x:xmpmeta>"
    assert analyze_image(_png({"XML:com.adobe.xmp": xmp})).ai_probability >= 95


def test_camera_photo_scores_low():
    r = analyze_image(_camera_jpeg())
    assert r.ai_probability < 40
    assert r.meta["Make"] == "Canon"


def test_heuristic_only_image_is_not_overconfident():
    assert 20 <= analyze_image(_png()).ai_probability <= 80


def test_api_text_and_image():
    r = client.post("/api/text", json={"text": AI_VI})
    assert r.status_code == 200 and r.json()["kind"] == "text"
    r = client.post("/api/image", files={"file": ("a.png", _png({"prompt": "{}"}), "image/png")})
    assert r.status_code == 200 and r.json()["ai_probability"] >= 95


def test_api_rejects_bad_input():
    assert client.post("/api/text", json={}).status_code == 400
    assert client.post("/api/text", json={"url": "http://127.0.0.1/"}).status_code == 400
    assert client.post("/api/text", json={"url": "file:///etc/passwd"}).status_code == 400
    assert client.post("/api/image", files={"file": ("x.png", b"nope", "image/png")}).status_code == 400


def test_index_and_health():
    assert "CheckAI" in client.get("/").text
    assert client.get("/api/health").json()["status"] == "ok"


def _docx(paragraphs):
    import docx

    d = docx.Document()
    for p in paragraphs:
        d.add_paragraph(p)
    d.core_properties.author = "Tester"
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def test_extract_docx_keeps_paragraphs_and_metadata():
    text, meta = extract_text(_docx(AI_VI.split("\n\n")), "bai.docx")
    assert text.count("\n\n") == AI_VI.count("\n\n")
    assert meta["Tác giả"] == "Tester"
    assert analyze_text(text).ai_probability > 65


def test_extract_pdf_rejoins_wrapped_lines():
    pdf = make_pdf([
        "This is the first sentence of a paragraph that wraps across",
        "two lines in the PDF file. A second sentence follows here and",
        "also ends.",
        "",
        "New paragraph starts here.",
    ])
    text, meta = extract_text(pdf, "a.pdf")
    assert "wraps across two lines" in text
    assert meta["Số trang"] == "1"


def test_extract_rejects_unsupported_and_scanned():
    for data, name in [(b"x", "a.doc"), (b"x", "a.xlsx"), (make_pdf([]), "scan.pdf"), (b"junk", "a.pdf")]:
        try:
            extract_text(data, name)
        except ValueError:
            continue
        raise AssertionError(name)


def test_api_document():
    r = client.post("/api/document", files={"file": ("bai.docx", _docx(AI_VI.split("\n\n")), "application/octet-stream")})
    assert r.status_code == 200 and r.json()["meta"]["Tệp"] == "bai.docx"
    assert client.post("/api/document", files={"file": ("a.doc", b"x", "application/msword")}).status_code == 400
