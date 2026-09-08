from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cow_reid.config import load_config
from cow_reid.embeddings import build_track_embeddings
from cow_reid.evaluation import evaluate_embeddings
from cow_reid.training import train_reid


def _image(cow: int, variation: int) -> np.ndarray:
    canvas = np.full((128, 256, 3), 238, dtype=np.uint8)
    if cow == 0:
        cv2.ellipse(canvas, (72 + variation, 64), (42, 48), 0, 0, 360, (15, 15, 15), -1)
        cv2.rectangle(canvas, (165, 28), (220, 102), (25, 25, 25), -1)
    else:
        cv2.rectangle(canvas, (30, 22), (105, 108), (20, 20, 20), -1)
        cv2.ellipse(canvas, (190 - variation, 68), (48, 30), 0, 0, 360, (25, 25, 25), -1)
    return canvas


def _make_run(run_dir: Path) -> None:
    tracks, frames, labels = [], [], []
    for cow in range(2):
        for session in range(2):
            tracklet_id = f"cow{cow}_session{session}"
            video_id, session_id = f"video_{session}", f"session_{session}"
            directory = run_dir / "tracklets" / tracklet_id
            directory.mkdir(parents=True, exist_ok=True)
            for rank in range(2):
                path = directory / f"crop_{rank}.jpg"
                cv2.imwrite(str(path), _image(cow, rank + session))
                frames.append({"tracklet_id": tracklet_id, "rank": rank, "quality": 0.9, "embedding_path": str(path)})
            tracks.append({"tracklet_id": tracklet_id, "video_id": video_id, "session_id": session_id, "valid": True, "start_s": float(cow)})
            labels.append({"tracklet_id": tracklet_id, "cow_id": f"COW_{cow + 1:04d}", "confirmed": True, "session_id": session_id, "video_id": video_id})
    pd.DataFrame(tracks).to_csv(run_dir / "tracklets.csv", index=False)
    pd.DataFrame(frames).to_csv(run_dir / "frames.csv", index=False)
    pd.DataFrame(labels).to_csv(run_dir / "labels.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="cow_metric_smoke_") as temporary:
        run_dir = Path(temporary)
        _make_run(run_dir)
        config = load_config()
        config["training"].update({"pretrained_checkpoint": str(Path(args.checkpoint).resolve()), "epochs": 1, "device": "cpu", "identities_per_batch": 2, "instances_per_identity": 2})
        output = run_dir / "smoke_metric.pt"
        train_reid(run_dir, run_dir / "labels.csv", config, output)
        config["embedding"].update({"backend": "metric", "checkpoint": str(output), "device": "cpu", "batch_size": 4})
        build_track_embeddings(run_dir, config)
        evaluation = evaluate_embeddings(run_dir, run_dir / "labels.csv")
        print(json.dumps({"checkpoint_exists": output.exists(), **evaluation}, indent=2))


if __name__ == "__main__":
    main()
