from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Callable, Iterable


@dataclass(frozen=True)
class TimestampSample:
    offset_s: float
    timestamp: datetime | None
    raw_text: str | None = None


@dataclass(frozen=True)
class TimestampResult:
    recording_start: datetime | None
    recording_end: datetime | None
    source: str
    confidence: float
    samples: tuple[TimestampSample, ...]
    error: str | None = None

    def samples_json(self) -> str:
        values = []
        for sample in self.samples:
            row = asdict(sample)
            row["timestamp"] = sample.timestamp.isoformat(sep=" ") if sample.timestamp else None
            values.append(row)
        return json.dumps(values, ensure_ascii=False)


def validate_timestamp_samples(
    samples: Iterable[TimestampSample],
    duration_s: float,
    tolerance_s: float = 3.0,
) -> TimestampResult:
    """Fit overlay readings to video time and reject non-linear OCR results."""
    all_samples = tuple(samples)
    valid = tuple(sample for sample in all_samples if sample.timestamp is not None)
    if not valid:
        return TimestampResult(None, None, "unknown", 0.0, all_samples, "no_valid_ocr_sample")
    starts = [sample.timestamp - timedelta(seconds=sample.offset_s) for sample in valid]
    epoch = sorted(value.timestamp() for value in starts)[len(starts) // 2]
    residuals = [abs(value.timestamp() - epoch) for value in starts]
    inliers = [sample for sample, residual in zip(valid, residuals) if residual <= tolerance_s]
    if len(inliers) < 2 and len(all_samples) >= 2:
        return TimestampResult(None, None, "unknown", 0.0, all_samples, "ocr_timeline_inconsistent")
    start = datetime.fromtimestamp(epoch, tz=starts[0].tzinfo)
    confidence = min(1.0, len(inliers) / max(3, len(all_samples)))
    if len(inliers) == 1:
        confidence = min(confidence, 0.35)
    end = start + timedelta(seconds=max(0.0, float(duration_s)))
    return TimestampResult(start, end, "camera_ocr", confidence, all_samples)


def sample_offsets(duration_s: float, count: int = 3) -> list[float]:
    if duration_s <= 0:
        return [0.0]
    margin = min(1.0, duration_s / 10.0)
    if count <= 1 or duration_s <= 2 * margin:
        return [max(0.0, duration_s / 2.0)]
    return [margin, duration_s / 2.0, max(margin, duration_s - margin)]


def manual_timestamp(start: datetime, duration_s: float) -> TimestampResult:
    return TimestampResult(
        start,
        start + timedelta(seconds=max(0.0, duration_s)),
        "manual",
        1.0,
        (),
    )

