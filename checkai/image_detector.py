"""Phát hiện ảnh do AI tạo: siêu dữ liệu (metadata, C2PA) + phân tích điểm ảnh.

Metadata là bằng chứng mạnh nhất khi còn nguyên (ảnh tải trực tiếp từ công cụ AI).
Phân tích điểm ảnh (phổ tần số, nhiễu cảm biến) chỉ là tín hiệu yếu.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import ExifTags, Image, UnidentifiedImageError

from . import providers
from .scoring import Result, Signal, blend, clamp, verdict_for, weighted_score

# Chuỗi trong metadata cho thấy ảnh do công cụ AI tạo ra.
GENERATOR_MARKERS = {
    b"trainedalgorithmicmedia": "IPTC DigitalSourceType = trainedAlgorithmicMedia (ảnh do AI tạo)",
    b"compositewithtrainedalgorithmicmedia": "IPTC: ảnh ghép có thành phần AI",
    b"stable diffusion": "Stable Diffusion",
    b"stable-diffusion": "Stable Diffusion",
    b"midjourney": "Midjourney",
    b"dall-e": "DALL·E",
    "dall·e".encode(): "DALL·E",
    b"adobe firefly": "Adobe Firefly",
    b"chatgpt": "ChatGPT / OpenAI",
    b"comfyui": "ComfyUI",
    b"novelai": "NovelAI",
    b"black-forest-labs": "FLUX (Black Forest Labs)",
    b"flux.1": "FLUX",
    b"ideogram": "Ideogram",
    b"leonardo.ai": "Leonardo.ai",
    b"google imagen": "Google Imagen",
    b"made with google ai": "Google AI",
    b"invokeai": "InvokeAI",
    b"automatic1111": "AUTOMATIC1111",
    b"sd-webui": "Stable Diffusion WebUI",
}
# Khoá text chunk của PNG do các giao diện Stable Diffusion ghi.
SD_PNG_KEYS = {"parameters", "prompt", "workflow", "negative_prompt", "sd-metadata", "invokeai_metadata"}
COMMON_AI_SIZES = {512, 576, 640, 704, 768, 832, 896, 960, 1024, 1152, 1216, 1280, 1344, 1536, 1792, 2048}
HEAD_SCAN = 1024 * 1024  # metadata nằm ở đầu tệp, tránh khớp nhầm trong dữ liệu nén


def _metadata(img: Image.Image, data: bytes) -> tuple[list[str], list[str], bool, dict]:
    """Trả về (bằng chứng AI, bằng chứng máy ảnh, có C2PA, thông tin hiển thị)."""
    ai_evidence: list[str] = []
    camera_evidence: list[str] = []
    info: dict = {"format": img.format, "size": f"{img.width}x{img.height}", "mode": img.mode}

    blob = data[:HEAD_SCAN].lower()
    for key, val in img.info.items():
        if isinstance(val, (str, bytes)):
            blob += b" " + (val.encode("utf-8", "ignore") if isinstance(val, str) else val).lower()
        if isinstance(key, str) and key.lower() in SD_PNG_KEYS:
            ai_evidence.append(f"PNG chứa trường '{key}' (prompt sinh ảnh)")

    for marker, label in GENERATOR_MARKERS.items():
        if marker in blob and label not in ai_evidence:
            ai_evidence.append(label)

    has_c2pa = b"c2pa" in blob or b"jumb" in blob[:HEAD_SCAN]

    exif = img.getexif()
    tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
    try:
        tags.update({ExifTags.TAGS.get(k, k): v for k, v in exif.get_ifd(0x8769).items()})
    except Exception:  # noqa: BLE001
        pass
    for t in ("Make", "Model", "Software", "DateTimeOriginal", "LensModel"):
        if t in tags:
            info[t] = str(tags[t]).strip("\x00 ")
    software = str(tags.get("Software", "")).lower()
    if any(m.decode(errors="ignore") in software for m in GENERATOR_MARKERS):
        ai_evidence.append(f"EXIF Software = {tags['Software']}")
    if tags.get("Make") or tags.get("Model"):
        camera_evidence.append(f"Thiết bị: {tags.get('Make', '')} {tags.get('Model', '')}".strip())
    shooting = [t for t in ("ExposureTime", "FNumber", "ISOSpeedRatings", "FocalLength") if t in tags]
    if shooting:
        camera_evidence.append("Thông số chụp: " + ", ".join(shooting))
    if 0x8825 in exif:
        camera_evidence.append("Có dữ liệu GPS")
    return ai_evidence, camera_evidence, has_c2pa, info


def _gray(img: Image.Image, max_side: int = 1024) -> np.ndarray:
    g = img.convert("L")
    g.thumbnail((max_side, max_side))
    return np.asarray(g, dtype=np.float64)


def spectral_peak_score(g: np.ndarray) -> tuple[float, float]:
    """Tìm đỉnh tuần hoàn ở tần số cao - dấu vết của upsampling trong mô hình sinh ảnh."""
    h, w = g.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    spec = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2((g - g.mean()) * win))))
    yy, xx = np.indices(spec.shape)
    r = np.hypot(yy - h / 2, xx - w / 2).astype(int)
    rmax = min(h, w) // 2
    mask = (r > rmax * 0.25) & (r < rmax)
    radial_mean = np.bincount(r.ravel(), spec.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    radial_sq = np.bincount(r.ravel(), (spec ** 2).ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    radial_std = np.sqrt(np.maximum(radial_sq - radial_mean ** 2, 1e-9))
    z = (spec - radial_mean[r]) / radial_std[r]
    frac = float((z[mask] > 5).mean())
    return clamp(frac / 0.002), frac


def noise_level(g: np.ndarray) -> float:
    """Ước lượng độ lệch chuẩn nhiễu (phương pháp Immerkær)."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    h, w = g.shape
    if h < 3 or w < 3:
        return 0.0
    conv = sum(k[i, j] * g[i:h - 2 + i, j:w - 2 + j] for i in range(3) for j in range(3))
    return float(np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6 * (w - 2) * (h - 2)))


def analyze_image(data: bytes, filename: str = "image") -> Result:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError("Không đọc được tệp ảnh.") from e

    ai_ev, cam_ev, has_c2pa, info = _metadata(img, data)
    g = _gray(img)
    signals: list[Signal] = []

    spec_score, frac = spectral_peak_score(g)
    signals.append(Signal("Đỉnh tuần hoàn trong phổ tần số", 0.35 + 0.5 * spec_score, 0.35,
                          f"{frac * 100:.3f}% điểm tần số cao bất thường (dấu vết upsampling)."))

    sigma = noise_level(g)
    signals.append(Signal("Nhiễu cảm biến", clamp(0.75 - (sigma - 0.8) * 0.2, 0.2, 0.75), 0.35,
                          f"Mức nhiễu ước lượng σ = {sigma:.2f} (ảnh máy ảnh thường có nhiễu, ảnh AI thường mịn)."))

    w, h = img.width, img.height
    ai_size = w % 64 == 0 and h % 64 == 0 and (w in COMMON_AI_SIZES or h in COMMON_AI_SIZES)
    signals.append(Signal("Kích thước ảnh", 0.7 if ai_size else 0.45, 0.15,
                          f"{w}x{h}" + (" - trùng kích thước phổ biến của mô hình sinh ảnh." if ai_size else ".")))

    no_meta = not cam_ev and not ai_ev
    signals.append(Signal("Siêu dữ liệu máy ảnh", 0.2 if cam_ev else 0.6, 0.15,
                          "; ".join(cam_ev) if cam_ev else
                          "Không có EXIF máy ảnh (có thể do AI, ảnh chụp màn hình, hoặc mạng xã hội đã xoá)."))

    if ai_ev:
        signals.append(Signal("Dấu vết công cụ AI trong metadata", 0.98, 3.0, "; ".join(ai_ev), "metadata"))

    h_score = weighted_score(signals)
    model_signals = providers.image_model_signals(data, filename)
    signals.extend(model_signals)
    p = blend(h_score, [s.score for s in model_signals])
    if ai_ev:
        p = max(p, 0.95)
    elif not model_signals:
        # Chỉ có tín hiệu điểm ảnh yếu: không đưa ra kết luận quá chắc chắn.
        p = clamp(p, 0.2, 0.45 if cam_ev else 0.8)

    notes = []
    if has_c2pa:
        notes.append("Ảnh có Content Credentials (C2PA). Kiểm tra chi tiết tại https://contentcredentials.org/verify")
    if no_meta:
        notes.append("Ảnh không có metadata; phân tích điểm ảnh chỉ là tín hiệu yếu.")
    if ai_ev:
        confidence = "Cao"
    elif model_signals:
        confidence = "Trung bình"
    else:
        confidence = "Thấp"
        notes.append("Bật mô hình cục bộ hoặc API Sightengine để tăng độ chính xác.")

    p *= 100
    return Result(kind="image", ai_probability=round(p, 1), verdict=verdict_for(p),
                  confidence=confidence, signals=signals, notes=notes, meta=info)
