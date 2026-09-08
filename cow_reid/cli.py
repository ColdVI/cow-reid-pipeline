from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .audit import audit_labels
from .clustering import cluster_run
from .config import load_config
from .embeddings import build_track_embeddings
from .evaluation import evaluate_embeddings
from .extract import extract_tracklets
from .gallery import build_gallery, identify_run
from .inventory import scan_videos
from .posture import import_posture_scores
from .db import init_database
from .registry import ingest_directory, pending_videos
from .pretrained import OPENCOWS2020_SOFTMAX_RTL_SHA256, OPENCOWS2020_SOFTMAX_RTL_URL, download_opencows2020_weights
from .report import generate_report
from .training import train_reid


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cow-reid", description="Tracklet-based cattle re-identification pipeline")
    parser.add_argument("--config", default=None, help="YAML configuration file")
    sub = parser.add_subparsers(dest="command", required=True)

    build_manifest = sub.add_parser(
        "build-manifest",
        help="Reconcile a raw inventory manifest with human-reviewed correction exports",
    )
    build_manifest.add_argument(
        "--raw",
        action="append",
        required=True,
        help="Raw inventory manifest CSV (e.g. data/video_manifest.csv), repeatable to concatenate multiple batches",
    )
    build_manifest.add_argument(
        "--corrections",
        action="append",
        default=[],
        help="Correction export CSV to overlay, in order (repeatable)",
    )
    build_manifest.add_argument("--output", required=True)

    inventory = sub.add_parser("inventory", help="Scan, OCR and deduplicate source videos")
    inventory.add_argument("--videos", required=True)
    inventory.add_argument("--output", required=True)
    inventory.add_argument("--no-ocr", action="store_true")
    inventory.add_argument("--camera", default="unknown")

    extract = sub.add_parser("extract", help="Detect, track and crop cow tracklets")
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--backend", choices=["yolo", "motion"], default=None)
    extract.add_argument("--max-videos", type=int, default=None)
    extract.add_argument("--max-seconds", type=float, default=None)

    embed = sub.add_parser("embed", help="Build one appearance embedding per tracklet")
    embed.add_argument("--run", required=True)
    embed.add_argument("--backend", choices=["opencows2020", "metric", "resnet18", "hist"], default=None)
    embed.add_argument("--checkpoint", default=None, help="Cattle-pretrained or farm fine-tuned checkpoint")

    train = sub.add_parser("train-reid", help="Fine-tune cattle Re-ID with classification plus metric loss")
    train.add_argument("--run", required=True)
    train.add_argument("--labels", required=True)
    train.add_argument("--output", default=None)
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--device", default=None)
    train.add_argument("--pretrained", default=None, help="OpenCows2020 initialization checkpoint")
    train.add_argument("--loss", choices=["reciprocal", "triplet"], default=None)
    train.add_argument("--stride-layout", choices=["torchvision", "opencows"], default=None, help="ResNet50 bottleneck downsampling stride placement")
    train.add_argument("--freeze-backbone", action="store_true", help="Freeze the backbone; only train the projection head and classifier")
    train.add_argument("--test-date", action="append", default=[], help="Recording date (YYYY-MM-DD) to hold out as test, repeatable")
    train.add_argument("--validation-date", action="append", default=[], help="Recording date (YYYY-MM-DD) to hold out as validation, repeatable")
    train.add_argument("--open-set-cow", action="append", default=[], help="Cow_ID to exclude entirely as an open-set test identity, repeatable")

    download = sub.add_parser("download-cattle-weights", help="Download verified OpenCows2020 Re-ID weights")
    download.add_argument("--output", default="weights/opencows2020_softmaxrtl.pkl")

    audit = sub.add_parser("audit-labels", help="Quarantine contradictory identity labels before training")
    audit.add_argument("--run", required=True)
    audit.add_argument("--labels", default=None)
    audit.add_argument("--output-dir", default=None)

    evaluate = sub.add_parser("evaluate-reid", help="Measure cross-session track-level retrieval")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--labels", required=True)
    evaluate.add_argument("--embeddings", default=None)
    evaluate.add_argument("--output", default=None)

    cluster = sub.add_parser("cluster", help="Create cross-session candidate pseudo identities")
    cluster.add_argument("--run", required=True)
    cluster.add_argument("--threshold", type=float, default=None)

    auto_confirm = sub.add_parser(
        "auto-confirm",
        help="Auto-confirm only mutual top-1 pairs where appearance AND geometry similarity agree",
    )
    auto_confirm.add_argument("--run", required=True)
    auto_confirm.add_argument("--cosine-threshold", type=float, default=0.95)
    auto_confirm.add_argument("--geometry-threshold", type=float, default=0.5)

    report = sub.add_parser("report", help="Generate an offline tracklet review report")
    report.add_argument("--run", required=True)

    run = sub.add_parser("run", help="Run inventory, extraction, embedding, clustering and report")
    run.add_argument("--videos", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--detector", choices=["yolo", "motion"], default=None)
    run.add_argument("--embedder", choices=["opencows2020", "metric", "resnet18", "hist"], default=None)
    run.add_argument("--max-videos", type=int, default=None)
    run.add_argument("--max-seconds", type=float, default=None)
    run.add_argument("--no-ocr", action="store_true")

    gallery = sub.add_parser("build-gallery", help="Build verified Cow_ID prototypes from labels.csv")
    gallery.add_argument("--run", required=True)
    gallery.add_argument("--labels", required=True)
    gallery.add_argument("--output", default=None)

    identify = sub.add_parser("identify", help="Match tracklets against a verified Cow_ID gallery")
    identify.add_argument("--run", required=True)
    identify.add_argument("--gallery", required=True)
    identify.add_argument("--output", default=None)

    review = sub.add_parser("review", help="Launch optional Streamlit review UI")
    review.add_argument("--run", default=None)
    review.add_argument("--database", default=None)

    review_bundle = sub.add_parser("review-bundle", help="Combine an established and a new run for human review")
    review_bundle.add_argument("--base", required=True)
    review_bundle.add_argument("--new", required=True)
    review_bundle.add_argument("--output", required=True)

    posture = sub.add_parser("import-posture", help="Attach keypoint posture scores to Cow_ID histories")
    posture.add_argument("--run", required=True)
    posture.add_argument("--scores", required=True, help="CSV with tracklet_id and arch_score/posture_score")
    posture.add_argument("--output", default=None)

    db_init = sub.add_parser("db-init", help="Initialize the durable registry schema")
    db_init.add_argument("--database", required=True)

    ingest = sub.add_parser("ingest", help="OCR, deduplicate and register incoming videos")
    ingest.add_argument("--videos", required=True)
    ingest.add_argument("--camera", required=True)
    ingest.add_argument("--database", required=True)
    ingest.add_argument("--manifest", default=None)
    ingest.add_argument("--no-ocr", action="store_true")

    timestamp_import = sub.add_parser("timestamp-import", help="Apply reviewed timestamp overrides and rebuild overlap groups")
    timestamp_import.add_argument("--manifest", required=True)
    timestamp_import.add_argument("--database", required=True)

    import_run = sub.add_parser("import-run", help="Register an existing file-based run without recomputing models")
    import_run.add_argument("--run", required=True)
    import_run.add_argument("--checkpoint", required=True)
    import_run.add_argument("--database", required=True)

    process_new = sub.add_parser("process-new", help="Process registered videos with resumable model stages")
    process_new.add_argument("--database", required=True)
    process_new.add_argument("--checkpoint", required=True)
    process_new.add_argument("--backend", choices=["opencows2020", "metric", "resnet18", "hist"], default=None)
    process_new.add_argument("--device", default="auto")
    process_new.add_argument("--output", default="data/processed")
    process_new.add_argument("--gallery", default=None)

    gallery_refresh = sub.add_parser("gallery-refresh", help="Refresh gallery metadata for a registered model")
    gallery_refresh.add_argument("--database", required=True)
    gallery_refresh.add_argument("--checkpoint", required=True)

    pose_run = sub.add_parser("pose-run", help="Run an external PoseBackend for selected tracklets")
    pose_run.add_argument("--database", default=None)
    pose_run.add_argument("--run", required=True)
    pose_run.add_argument("--backend", choices=["lameness_dlc"], default="lameness_dlc")
    pose_run.add_argument("--checkpoint", required=True)
    pose_run.add_argument("--repo", default="/Users/anil/Downloads/lameness-main")
    pose_run.add_argument("--tracklet", action="append", default=[])
    pose_run.add_argument("--max-tracklets", type=int, default=1)
    pose_run.add_argument("--python", dest="pose_python", default=None, help="Python executable containing DeepLabCut")
    pose_run.add_argument("--device", default="auto")

    health = sub.add_parser("health-refresh", help="Recompute longitudinal robust baselines")
    health.add_argument("--run", default=None)
    health.add_argument("--database", default=None)
    health.add_argument("--output", default=None)
    return parser


def _summary(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "build-manifest":
        from .manifest import build_canonical_manifest_files

        frame = build_canonical_manifest_files(args.raw, args.corrections, args.output)
        overlap_groups = frame.get("overlap_group_id")
        multi_member_groups = int(overlap_groups.value_counts().gt(1).sum()) if overlap_groups is not None else 0
        unverified = int(frame.get("overlap_unverified", False).sum()) if "overlap_unverified" in frame else 0
        print(json.dumps({
            "videos": len(frame),
            "overlap_groups_with_duplicates": multi_member_groups,
            "unverified_timestamp_rows": unverified,
            "manifest": str(Path(args.output).resolve()),
        }, indent=2))
    elif args.command == "inventory":
        frame = scan_videos(args.videos, args.output, use_ocr=not args.no_ocr, camera_id=args.camera)
        if frame.empty:
            raise SystemExit(f"No videos matched in {Path(args.videos).resolve()} (supported pattern: *.mp4, case-insensitive)")
        print(json.dumps({"files": len(frame), "unique": int((~frame["is_duplicate"]).sum()), "manifest": str(Path(args.output).resolve())}, indent=2))
    elif args.command == "extract":
        if args.backend:
            cfg["extract"]["backend"] = args.backend
        extract_tracklets(args.manifest, args.output, cfg, max_videos=args.max_videos, max_seconds=args.max_seconds)
        print(json.dumps(_summary(Path(args.output) / "extract_summary.json"), indent=2))
    elif args.command == "embed":
        if args.checkpoint:
            cfg["embedding"]["checkpoint"] = args.checkpoint
        build_track_embeddings(args.run, cfg, backend_override=args.backend)
        print(json.dumps(_summary(Path(args.run) / "embedding_summary.json"), indent=2))
    elif args.command == "train-reid":
        if args.epochs is not None:
            cfg["training"]["epochs"] = args.epochs
        if args.device is not None:
            cfg["training"]["device"] = args.device
        if args.pretrained is not None:
            cfg["training"]["pretrained_checkpoint"] = args.pretrained
        if args.loss is not None:
            cfg["training"]["metric_loss"] = args.loss
        if args.stride_layout is not None:
            cfg["training"]["stride_layout"] = args.stride_layout
        if args.freeze_backbone:
            cfg["training"]["freeze_backbone"] = True
        if args.test_date:
            cfg["splitting"]["test_dates"] = args.test_date
        if args.validation_date:
            cfg["splitting"]["validation_dates"] = args.validation_date
        if args.open_set_cow:
            cfg["splitting"]["open_set_cow_ids"] = args.open_set_cow
        checkpoint = train_reid(args.run, args.labels, cfg, args.output)
        print(json.dumps({"checkpoint": str(checkpoint), **_summary(Path(args.run) / "training_summary.json")}, indent=2))
    elif args.command == "download-cattle-weights":
        path = download_opencows2020_weights(args.output)
        print(json.dumps({"checkpoint": str(path), "source": OPENCOWS2020_SOFTMAX_RTL_URL, "sha256": OPENCOWS2020_SOFTMAX_RTL_SHA256}, indent=2))
    elif args.command == "audit-labels":
        outputs = audit_labels(args.run, args.labels, args.output_dir)
        print(json.dumps({**_summary(outputs["summary"]), **{key: str(value) for key, value in outputs.items()}}, indent=2))
    elif args.command == "evaluate-reid":
        print(json.dumps(evaluate_embeddings(args.run, args.labels, args.embeddings, args.output), indent=2))
    elif args.command == "cluster":
        if args.threshold is not None:
            cfg["matching"]["similarity_threshold"] = args.threshold
        cluster_run(args.run, cfg)
        print(json.dumps(_summary(Path(args.run) / "matching_summary.json"), indent=2))
    elif args.command == "auto-confirm":
        import pandas as pd

        from .identity import auto_confirm_dual_signal, load_identity_labels, save_identity_labels

        run_dir = Path(args.run).resolve()
        labels = load_identity_labels(run_dir)
        candidate_pairs = pd.read_csv(run_dir / "candidate_pairs.csv")
        updated, applied = auto_confirm_dual_signal(
            run_dir,
            labels,
            candidate_pairs,
            cosine_threshold=args.cosine_threshold,
            geometry_threshold=args.geometry_threshold,
        )
        save_identity_labels(updated, run_dir)
        print(json.dumps({"confirmed_pairs": len(applied), "pairs": applied}, indent=2))
    elif args.command == "report":
        path = generate_report(args.run)
        print(path)
    elif args.command == "run":
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        manifest = output / "video_manifest.csv"
        scan_videos(args.videos, manifest, use_ocr=not args.no_ocr)
        if args.detector:
            cfg["extract"]["backend"] = args.detector
        if args.embedder:
            cfg["embedding"]["backend"] = args.embedder
        extract_tracklets(manifest, output, cfg, max_videos=args.max_videos, max_seconds=args.max_seconds)
        build_track_embeddings(output, cfg)
        cluster_run(output, cfg)
        report_path = generate_report(output)
        print(json.dumps({"run": str(output), "report": str(report_path), **_summary(output / "matching_summary.json")}, indent=2))
    elif args.command == "build-gallery":
        path = build_gallery(args.run, args.labels, args.output)
        print(path)
    elif args.command == "identify":
        result = identify_run(args.run, args.gallery, cfg, args.output)
        print(json.dumps({"predictions": len(result), "output": str(Path(args.output).resolve()) if args.output else str(Path(args.run).resolve() / "identity_predictions.csv")}, indent=2))
    elif args.command == "review":
        if not args.run:
            raise SystemExit("Database-backed review UI requires a materialized run; pass --run for this release.")
        app = Path(__file__).with_name("review_app.py")
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), "--", "--run", str(Path(args.run).resolve())])
    elif args.command == "review-bundle":
        from .review_bundle import build_review_bundle
        print(json.dumps(build_review_bundle(args.base, args.new, args.output), indent=2))
    elif args.command == "import-posture":
        path = import_posture_scores(args.run, args.scores, args.output)
        print(path)
    elif args.command == "db-init":
        init_database(args.database)
        print(json.dumps({"database": args.database, "initialized": True}, indent=2))
    elif args.command == "ingest":
        print(json.dumps(ingest_directory(args.videos, args.camera, args.database, not args.no_ocr, args.manifest), indent=2))
    elif args.command == "timestamp-import":
        from .registry import import_timestamp_overrides
        print(json.dumps(import_timestamp_overrides(args.database, args.manifest), indent=2))
    elif args.command == "import-run":
        from .registry import import_existing_run
        print(json.dumps(import_existing_run(args.database, args.run, args.checkpoint), indent=2))
    elif args.command == "process-new":
        from .worker import process_new_videos
        result = process_new_videos(args.database, args.output, cfg, args.checkpoint, args.device, args.gallery, args.backend)
        print(json.dumps({"processed": len(result), "results": result}, indent=2))
    elif args.command == "gallery-refresh":
        from .registry import register_model
        checkpoint = Path(args.checkpoint).resolve()
        version = register_model(args.database, checkpoint)
        print(json.dumps({"database": args.database, "checkpoint": str(checkpoint), "model_version": version, "status": "registered"}, indent=2))
    elif args.command == "pose-run":
        import pandas as pd
        from .pose_runner import run_lameness_pose
        run = Path(args.run).resolve()
        tracklet_ids = list(args.tracklet)
        if not tracklet_ids:
            tracks = pd.read_csv(run / "tracklets.csv")
            if "valid" in tracks:
                from .utils import bool_mask
                tracks = tracks.loc[bool_mask(tracks["valid"], default=False)]
            tracks = tracks.sort_values("mean_quality", ascending=False)
            tracklet_ids = tracks["tracklet_id"].astype(str).head(args.max_tracklets).tolist()
        pose_python = args.pose_python or str(Path(".venv-dlc/bin/python").resolve())
        try:
            results = [run_lameness_pose(run, tracklet_id, args.repo, args.checkpoint, pose_python, args.device) for tracklet_id in tracklet_ids]
        except (FileNotFoundError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
            raise SystemExit(f"Pose run failed: {exc}") from exc
        print(json.dumps({"processed": len(results), "results": results}, indent=2))
    elif args.command == "health-refresh":
        if not args.run:
            raise SystemExit("Pass --run containing posture_observations.csv; direct DB refresh will be enabled with the PostgreSQL runtime.")
        from .posture import refresh_health_summaries
        path = refresh_health_summaries(args.run, args.output)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
