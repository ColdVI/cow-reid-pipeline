from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cow_reid.clustering import cluster_run
from cow_reid.config import load_config
from cow_reid.embeddings import build_track_embeddings
from cow_reid.report import generate_report
from cow_reid.utils import make_contact_sheet


def patterned_image(identity: int, variation: int) -> np.ndarray:
    image = np.full((160, 320, 3), 235, dtype=np.uint8)
    if identity == 0:
        cv2.ellipse(image, (90 + variation, 75), (45, 58), 10, 0, 360, (20, 20, 20), -1)
        cv2.rectangle(image, (190, 30), (260, 120), (25, 25, 25), -1)
    else:
        cv2.rectangle(image, (35, 25), (125, 130), (20, 20, 20), -1)
        cv2.ellipse(image, (235 - variation, 85), (55, 35), -15, 0, 360, (25, 25, 25), -1)
    return image


def create_synthetic_run(run_dir: Path) -> None:
    tracks = []
    frames = []
    specification = [("s1_a", "session_1", 0), ("s1_b", "session_1", 1), ("s2_a", "session_2", 0), ("s2_b", "session_2", 1)]
    for tracklet_id, session_id, identity in specification:
        directory = run_dir / "tracklets" / tracklet_id
        directory.mkdir(parents=True, exist_ok=True)
        images = []
        for rank in range(1, 5):
            image = patterned_image(identity, rank * 2)
            image_path = directory / f"rank_{rank:02d}.jpg"
            cv2.imwrite(str(image_path), image)
            images.append(image)
            frames.append(
                {
                    "tracklet_id": tracklet_id,
                    "video_id": session_id,
                    "session_id": session_id,
                    "rank": rank,
                    "frame_idx": rank,
                    "timestamp_s": float(rank),
                    "quality": 0.9,
                    "confidence": 0.9,
                    "overlap": 0.0,
                    "bbox": "0,0,320,160",
                    "torso_path": str(image_path.resolve()),
                    "full_path": str(image_path.resolve()),
                }
            )
        contact_path = directory / "contact.jpg"
        cv2.imwrite(str(contact_path), make_contact_sheet(images))
        tracks.append(
            {
                "tracklet_id": tracklet_id,
                "video_id": session_id,
                "session_id": session_id,
                "session_type": "test",
                "source_video": "synthetic",
                "source_filename": "synthetic",
                "valid": True,
                "rejection_reasons": "",
                "n_observations": 20,
                "n_best_frames": 4,
                "start_s": 0.0,
                "end_s": 4.0,
                "duration_s": 4.0,
                "x_start": 0.1,
                "x_end": 0.9,
                "x_span": 0.8,
                "net_dx": 0.8,
                "mean_confidence": 0.9,
                "mean_quality": 0.9,
                "contact_path": str(contact_path.resolve()),
                "metadata_path": "",
            }
        )
    pd.DataFrame(tracks).to_csv(run_dir / "tracklets.csv", index=False)
    pd.DataFrame(frames).to_csv(run_dir / "frames.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.output:
        run_dir = Path(args.output).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = Path(tempfile.mkdtemp(prefix="cow_reid_smoke_"))
    create_synthetic_run(run_dir)
    cfg = load_config()
    cfg["embedding"]["backend"] = "hist"
    cfg["matching"]["similarity_threshold"] = 0.90
    build_track_embeddings(run_dir, cfg)
    assignments, _ = cluster_run(run_dir, cfg)
    report = generate_report(run_dir)
    print(f"run={run_dir}")
    print(f"tracklets={len(assignments)} pseudo_ids={assignments.pseudo_id.nunique()}")
    print(f"report={report}")


if __name__ == "__main__":
    main()

