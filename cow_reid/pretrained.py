from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path


OPENCOWS2020_SOFTMAX_RTL_URL = (
    "https://github.com/CWOA/MetricLearningIdentification/raw/refs/heads/master/"
    "weights/SoftmaxRTL/50-50_fold-1.pkl"
)
OPENCOWS2020_SOFTMAX_RTL_SHA256 = "c126fcc5aa446e2f93134c9a013e47cad232f4147114c9e409ce21844598b34a"
OPENCOWS2020_SOFTMAX_RTL_BYTES = 103_197_184


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_opencows2020_weights(
    output_path: str | Path,
    url: str = OPENCOWS2020_SOFTMAX_RTL_URL,
    expected_sha256: str = OPENCOWS2020_SOFTMAX_RTL_SHA256,
) -> Path:
    """Download the public OpenCows2020 Softmax+RTL checkpoint atomically."""

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and sha256_file(output) == expected_sha256:
        return output
    temporary = output.with_suffix(output.suffix + ".download")
    if temporary.exists():
        temporary.unlink()
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "cow-tracklet-reid/0.3"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        actual = sha256_file(temporary)
        if actual != expected_sha256:
            raise RuntimeError(
                f"OpenCows2020 checkpoint checksum mismatch: expected {expected_sha256}, got {actual}."
            )
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output
