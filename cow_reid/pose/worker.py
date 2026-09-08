from __future__ import annotations

import argparse
import json
from pathlib import Path

from .lameness_dlc import LamenessDLCBackend


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True); parser.add_argument("--video", required=True); parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--checkpoint", default=None); parser.add_argument("--device", default="auto")
    args = parser.parse_args(); metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    backend = LamenessDLCBackend(args.repo, args.checkpoint, args.device)
    backend.infer_tracklet(args.video, float(metadata["start_s"]), float(metadata["end_s"]), metadata["records"], args.output)
    return 0


if __name__ == "__main__": raise SystemExit(main())

