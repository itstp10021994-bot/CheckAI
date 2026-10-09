"""Phát hiện văn bản do AI viết bằng các đặc trưng thống kê (hỗ trợ tiếng Việt và tiếng Anh).

Các heuristic này chỉ là tín hiệu yếu; kết quả đáng tin hơn khi bật thêm mô hình
hoặc API (xem checkai/providers.py).
"""

from __future__ import annotations

import re
import statistics

from . import providers
from .scoring import Result, Signal, blend, clamp, verdict_for, weighted_score

# Cụm từ "đặc sản" của văn bản do LLM viết.
AI_PHRASES = [
    # English
    "delve", "it's important to note", "it is important to note",
    "it's worth noting", "it is worth noting", "in today's fast-paced", "in today's digital",
    "in conclusion", "tapestry", "testament to", "plays a crucial role", "plays a vital role",
    "a pivotal role", "navigate the complexities", "ever-evolving", "ever-changing landscape",
    "landscape of", "unlock the potential", "harness the power", "embark on a journey",
    "seamlessly", "furthermore", "moreover", "in summary", "overall,", "as an ai",
    "i hope this helps", "let's dive", "dive into", "foster", "robust", "leverage",
    "multifaceted", "holistic", "underscores", "paramount", "elevate your",
    # Tiếng Việt
    "tóm lại", "nhìn chung", "điều quan trọng là", "cần lưu ý rằng", "đáng chú ý là",
    "đóng vai trò quan trọng", "đóng vai trò then chốt", "vai trò không thể thiếu",
    "không chỉ", "mà còn", "trong thời đại", "trong bối cảnh", "kỷ nguyên số",
    "hãy cùng", "cùng khám phá", "khám phá", "tối ưu hóa", "nâng tầm", "toàn diện",
    "bền vững", "hành trình", "góp phần", "mang lại", "một cách hiệu quả",
    "không ngừng phát triển", "ngày càng phát triển", "là một trong những",
    "hy vọng những thông tin", "dưới đây là", "với tư cách là một ai",
]

TRANSITIONS = {
    "however", "moreover", "furthermore", "additionally", "therefore", "consequently",
    "overall", "ultimately", "firstly", "secondly", "finally", "lastly", "thus",
}
# Các mở đầu chuyển ý nhiều chữ (tiếng Việt).
TRANSITION_PREFIXES = (
    "ngoài ra", "hơn nữa", "bên cạnh đó", "tuy nhiên", "do đó", "vì vậy", "cuối cùng",
    "đầu tiên", "thứ nhất", "thứ hai", "thứ ba", "tóm lại", "nhìn chung", "từ đó",
    "như vậy", "đồng thời", "mặt khác",
)

INFORMAL_MARKERS = [
    "lol", "haha", "hihi", "hehe", "kkk", "omg", "btw", "imo", "tbh", "idk", "ko ", "k ",
    "dc ", "đc", "vl", "vcl", "mình thấy", "tớ", "bọn tao", "ae ", "ạ", "nhé", "nha", "á ",
    ":)", ":(", ":d", "=))", "^^", "!!", "??", "...", "ủa", "trời ơi", "thật sự là",
]

_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")
_WORD = re.compile(r"\w+", re.UNICODE)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if len(_WORD.findall(s)) >= 3]


def _count_phrases(lower: str, phrases: list[str]) -> list[str]:
    found = []
    for p in phrases:
        p = p.strip()
        n = len(re.findall(r"(?<!\w)" + re.escape(p) + r"(?!\w)", lower))
        found.extend([p] * n)
    return found


def _starts_with_transition(sentence: str) -> bool:
    low = sentence.lower().lstrip("-•*#> ")
    if low.startswith(TRANSITION_PREFIXES):
        return True
    m = _WORD.match(low)
    return bool(m) and m.group(0) in TRANSITIONS


def heuristic_signals(text: str) -> tuple[list[Signal], list[dict], int]:
    lower = text.lower()
    words = _WORD.findall(lower)
    n_words = len(words)
    sentences = split_sentences(text)
    signals: list[Signal] = []

    # 1. Burstiness: con người viết câu dài ngắn thất thường, AI đều đặn hơn.
    lengths = [len(_WORD.findall(s)) for s in sentences]
    if len(lengths) >= 5:
        cv = statistics.pstdev(lengths) / statistics.mean(lengths)
        score = clamp((0.75 - cv) / 0.5)
        signals.append(Signal(
            "Độ biến thiên độ dài câu", score, 0.30,
            f"Hệ số biến thiên = {cv:.2f} (AI thường < 0.35, người thường > 0.6)."))
    else:
        signals.append(Signal("Độ biến thiên độ dài câu", 0.5, 0.05,
                              "Quá ít câu để đánh giá."))

    # 2. Mật độ cụm từ đặc trưng của AI.
    hits = _count_phrases(lower, AI_PHRASES)
    density = len(hits) / max(n_words, 1) * 100
    score = 0.3 + 0.7 * clamp(density / 2.0)
    top = ", ".join(sorted(set(hits))[:8]) or "không có"
    signals.append(Signal(
        "Cụm từ đặc trưng AI", score, 0.30,
        f"{len(hits)} lần ({density:.2f}/100 từ). Ví dụ: {top}."))

    # 3. Câu mở đầu bằng từ chuyển ý ("Ngoài ra", "Moreover"...).
    if sentences:
        ratio = sum(_starts_with_transition(s) for s in sentences) / len(sentences)
        signals.append(Signal(
            "Từ nối đầu câu", clamp((ratio - 0.05) / 0.3), 0.15,
            f"{ratio * 100:.0f}% câu mở đầu bằng từ chuyển ý."))

    # 4. Độ đồng đều độ dài đoạn văn.
    paras = [p for p in re.split(r"\n\s*\n", text) if len(_WORD.findall(p)) >= 5]
    if len(paras) >= 3:
        plens = [len(_WORD.findall(p)) for p in paras]
        cv = statistics.pstdev(plens) / statistics.mean(plens)
        signals.append(Signal(
            "Độ đồng đều đoạn văn", clamp((0.6 - cv) / 0.45), 0.10,
            f"{len(paras)} đoạn, hệ số biến thiên = {cv:.2f}."))

    # 5. Định dạng kiểu chatbot (tiêu đề markdown, gạch đầu dòng in đậm, gạch ngang dài).
    fmt = (len(re.findall(r"^\s*#{1,6}\s", text, re.M))
           + len(re.findall(r"^\s*[-*•]\s+\*\*[^*]+\*\*", text, re.M)) * 2
           + text.count("—"))
    signals.append(Signal(
        "Định dạng kiểu chatbot", clamp(0.35 + fmt / 8), 0.05,
        f"{fmt} dấu hiệu định dạng (tiêu đề markdown, **in đậm**, —)."))

    # 6. Dấu hiệu văn nói, viết tắt, cảm xúc - thường thấy ở người.
    informal = _count_phrases(" " + lower + " ", INFORMAL_MARKERS)
    idensity = len(informal) / max(n_words, 1) * 100
    signals.append(Signal(
        "Văn phong tự nhiên / không trang trọng", clamp(0.65 - idensity * 0.25), 0.10,
        f"{len(informal)} dấu hiệu văn nói/viết tắt ({idensity:.2f}/100 từ)."))

    # Câu nghi vấn: chứa cụm từ đặc trưng AI.
    highlights = []
    for s in sentences:
        found = sorted(set(_count_phrases(s.lower(), AI_PHRASES)))
        if found:
            highlights.append({"text": s, "reasons": found})

    return signals, highlights, n_words


def analyze_text(text: str) -> Result:
    text = (text or "").strip()
    if not text:
        raise ValueError("Văn bản trống.")

    signals, highlights, n_words = heuristic_signals(text)
    h = weighted_score(signals)
    # Văn bản ngắn: kéo điểm về mức trung lập vì thống kê không đủ tin cậy.
    h = 0.5 + (h - 0.5) * min(1.0, n_words / 150)

    model_signals = providers.text_model_signals(text)
    signals.extend(model_signals)
    p = blend(h, [s.score for s in model_signals]) * 100

    notes = []
    if n_words < 50:
        confidence = "Thấp"
        notes.append("Văn bản quá ngắn (< 50 từ); kết quả kém tin cậy.")
    elif model_signals:
        confidence = "Cao" if n_words >= 150 else "Trung bình"
    else:
        confidence = "Trung bình" if n_words >= 150 else "Thấp"
        notes.append("Chỉ dùng heuristic thống kê. Bật mô hình/API để tăng độ chính xác.")

    return Result(
        kind="text",
        ai_probability=round(p, 1),
        verdict=verdict_for(p),
        confidence=confidence,
        signals=signals,
        highlights=highlights,
        notes=notes,
    )
