"""Kiểu dữ liệu chung và hàm gộp điểm cho các bộ phát hiện."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


@dataclass
class Signal:
    """Một dấu hiệu riêng lẻ. score: 0 = giống người, 1 = giống AI."""

    name: str
    score: float
    weight: float
    detail: str
    source: str = "heuristic"


@dataclass
class Result:
    kind: str
    ai_probability: float  # 0..100
    verdict: str
    confidence: str
    signals: list[Signal] = field(default_factory=list)
    highlights: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def weighted_score(signals: list[Signal]) -> float:
    total = sum(s.weight for s in signals)
    if total <= 0:
        return 0.5
    return sum(s.score * s.weight for s in signals) / total


def verdict_for(p: float) -> str:
    if p >= 85:
        return "Rất có thể do AI tạo"
    if p >= 65:
        return "Có khả năng do AI tạo"
    if p >= 40:
        return "Không rõ ràng / có thể pha trộn"
    if p >= 20:
        return "Có khả năng do người tạo"
    return "Rất có thể do người tạo"


def blend(heuristic: float, model_scores: list[float], model_weight: float = 0.7) -> float:
    """Gộp điểm heuristic với điểm từ mô hình/API (nếu có). Mô hình được ưu tiên hơn."""
    if not model_scores:
        return heuristic
    m = sum(model_scores) / len(model_scores)
    return model_weight * m + (1 - model_weight) * heuristic
