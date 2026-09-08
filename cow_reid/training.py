from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from math import ceil
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from PIL import Image

from .audit import audit_labels
from .config import resolve_model_path
from .evaluation import held_out_date_split
from .metric_model import build_metric_resnet50, build_metric_transform, load_opencows2020_initialization
from .review_state import load_excluded_tracklets
from .utils import bool_mask, l2_normalize, save_json


def _auto_last_day_split(manifest: pd.DataFrame) -> pd.DataFrame:
    """Default video-level split policy when no explicit test/validation dates are
    given: hold out the single most recent *verified* recording date as
    validation. Rows flagged ``overlap_unverified`` (see ``overlap.py``) are never
    eligible to be "the last day" — an unverified date can't safely anchor a
    held-out split, since we can't rule out that it secretly overlaps a day
    already used for training.
    """
    timestamp = manifest.get("recording_start", manifest.get("recording_timestamp"))
    if timestamp is None:
        raise RuntimeError("Manifest needs recording_start to auto-select a validation day.")
    recording_date = pd.to_datetime(timestamp, errors="coerce").dt.date.astype("string")
    unverified = bool_mask(manifest.get("overlap_unverified", pd.Series(False, index=manifest.index)), default=False)
    verified_dates = recording_date.loc[~unverified.to_numpy()].dropna().unique().tolist()
    if not verified_dates:
        raise RuntimeError(
            "No manifest rows have a verified recording date; cannot auto-select a "
            "validation day. Pass explicit validation dates (splitting.validation_dates "
            "in config, or --validation-date on the CLI)."
        )
    validation_day = max(verified_dates)
    return held_out_date_split(manifest, test_dates=[], validation_dates=[validation_day])


def _load_video_split(run_dir: Path, test_dates: list[str], validation_dates: list[str]) -> pd.DataFrame:
    """Resolve each video's train/validation/test assignment from the run's own
    video_manifest.csv (written by extract_tracklets from whatever manifest it was
    given — feed it the canonical, overlap-corrected manifest via
    'cow-reid build-manifest' for this to be leakage-safe)."""
    manifest_path = run_dir / "video_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} is missing. Re-run 'cow-reid extract' (it now writes the "
            "manifest it used into the run directory) with a manifest that carries real "
            "recording_start/overlap_group_id — see 'cow-reid build-manifest'."
        )
    manifest = pd.read_csv(manifest_path)
    if "video_id" not in manifest.columns:
        raise ValueError(f"{manifest_path} is missing a video_id column.")
    if test_dates or validation_dates:
        return held_out_date_split(manifest, test_dates, validation_dates)
    return _auto_last_day_split(manifest)


def _derive_tracklet_split(
    labels: pd.DataFrame,
    tracks: pd.DataFrame,
    video_split: pd.DataFrame,
    excluded_tracklets: set[str] | None = None,
    open_set_cow_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Every labelled tracklet inherits its split from its own video's (== its
    overlap group's) date-derived split. This replaces the old per-cow
    lexicographic-session heuristic, which had no notion of overlapping
    recordings or real dates and could split a single physical passage's
    tracklets across train and validation.

    A cow whose confirmed tracklets are all on one assigned day is correctly
    train-only (never force-split mid-session, unlike the old behavior); a cow
    with tracklets on more than one assigned day naturally gets both splits.
    Tracklets a human flagged in tracklet_reviews.csv, and cows listed as
    open-set test identities, are excluded from train/val entirely.
    """
    excluded_tracklets = excluded_tracklets or set()
    open_set_cow_ids = {str(cow_id) for cow_id in (open_set_cow_ids or set())}

    labelled = labels.copy()
    labelled["cow_id"] = labelled["cow_id"].fillna("").astype(str).str.strip()
    labelled = labelled.loc[labelled["cow_id"].ne("")]
    if "confirmed" in labelled.columns:
        labelled = labelled.loc[bool_mask(labelled["confirmed"])]
    if "training_eligible" in labelled.columns:
        labelled = labelled.loc[bool_mask(labelled["training_eligible"])]
    labelled = labelled[["tracklet_id", "cow_id"]].drop_duplicates("tracklet_id")
    labelled = labelled.loc[~labelled["tracklet_id"].astype(str).isin(excluded_tracklets)]
    labelled = labelled.merge(tracks[["tracklet_id", "session_id", "video_id"]], on="tracklet_id", how="inner")

    video_to_split = video_split.set_index(video_split["video_id"].astype(str))["split"].to_dict()
    labelled["video_split"] = labelled["video_id"].astype(str).map(video_to_split)
    labelled = labelled.loc[labelled["video_split"].notna()].copy()

    split_map = {"train": "train", "validation": "val", "test": "test"}
    pieces: list[pd.DataFrame] = []
    for cow_id, group in labelled.groupby("cow_id"):
        group = group.sort_values(["video_id", "tracklet_id"]).drop_duplicates("tracklet_id").copy()
        if cow_id in open_set_cow_ids:
            group["split"] = "open_set_test"
        else:
            group["split"] = group["video_split"].map(split_map).fillna("train")
        pieces.append(group)
    if not pieces:
        raise RuntimeError("Need at least two confirmed, conflict-free tracklets per Cow_ID.")

    split = pd.concat(pieces, ignore_index=True).drop(columns=["video_split"])
    trainval_candidates = split.loc[~split["split"].eq("open_set_test")]
    trainval = trainval_candidates.loc[trainval_candidates["split"].isin(["train", "val"])]
    counts = trainval.groupby("cow_id")["split"].nunique()
    keep_ids = set(counts[counts == 2].index)
    result = split.loc[split["cow_id"].isin(keep_ids) | split["split"].eq("open_set_test")].reset_index(drop=True)
    # Only enforce train+val presence when there was actually some non-open-set
    # data to split -- a request for open-set-only data has nothing to train
    # or validate on by design, that's not an error.
    if not trainval_candidates.empty and (not (result["split"] == "train").any() or not (result["split"] == "val").any()):
        raise RuntimeError(
            "Need at least two confirmed, conflict-free tracklets per Cow_ID spanning both "
            "a train day and a validation day."
        )
    return result


class _CropDataset:
    def __init__(self, rows: pd.DataFrame, class_to_index: dict[str, int], transform: Any):
        self.rows = rows.reset_index(drop=True)
        self.class_to_index = class_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        with Image.open(str(row["image_path"])) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.class_to_index[str(row["cow_id"])], str(row["tracklet_id"])


class _PKBatchSampler:
    """Sample P identities and K crops, spreading K across tracklets when possible."""

    def __init__(self, rows: pd.DataFrame, identities: int, instances: int, seed: int = 42):
        self.identities = max(2, min(int(identities), rows["cow_id"].nunique()))
        self.instances = max(2, int(instances))
        self.seed = int(seed)
        self.iteration = 0
        self.by_identity: dict[str, dict[str, list[int]]] = {}
        for cow_id, cow_rows in rows.reset_index(drop=True).groupby("cow_id"):
            self.by_identity[str(cow_id)] = {
                str(tracklet_id): group.index.astype(int).tolist()
                for tracklet_id, group in cow_rows.groupby("tracklet_id")
            }
        self.batches = max(1, ceil(len(rows) / (self.identities * self.instances)))

    def __len__(self) -> int:
        return self.batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.iteration)
        self.iteration += 1
        cow_ids = np.asarray(sorted(self.by_identity), dtype=object)
        for _ in range(self.batches):
            selected_cows = rng.choice(cow_ids, size=self.identities, replace=False)
            batch: list[int] = []
            for cow_id in selected_cows:
                tracks = self.by_identity[str(cow_id)]
                track_ids = np.asarray(sorted(tracks), dtype=object)
                chosen_tracks = rng.choice(track_ids, size=self.instances, replace=len(track_ids) < self.instances)
                for tracklet_id in chosen_tracks:
                    batch.append(int(rng.choice(tracks[str(tracklet_id)])))
            rng.shuffle(batch)
            yield batch


def _resolve_device(value: str) -> str:
    if value != "auto":
        return value
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _batch_hard_metric_loss(embeddings, targets, mode: str, margin: float):
    """Batch-hard triplet or reciprocal loss for L2-normalized embeddings."""

    import torch

    distances = torch.clamp(2.0 - 2.0 * embeddings @ embeddings.T, min=0.0)
    same = targets[:, None].eq(targets[None, :])
    diagonal = torch.eye(len(targets), dtype=torch.bool, device=targets.device)
    positives = same & ~diagonal
    negatives = ~same
    valid = positives.any(dim=1) & negatives.any(dim=1)
    if not bool(valid.any()):
        return embeddings.sum() * 0.0
    hardest_positive = distances.masked_fill(~positives, float("-inf")).max(dim=1).values[valid]
    hardest_negative = distances.masked_fill(~negatives, float("inf")).min(dim=1).values[valid]
    if mode == "triplet":
        return torch.relu(hardest_positive - hardest_negative + float(margin)).mean()
    if mode == "reciprocal":
        return (hardest_positive + 1.0 / hardest_negative.clamp_min(0.05)).mean()
    raise ValueError(f"Unsupported metric loss: {mode}")


def _freeze_backbone(model) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False


def _prototype_retrieval_metrics(
    train_embeddings: np.ndarray,
    train_cow_ids: list[str],
    val_embeddings: np.ndarray,
    val_cow_ids: list[str],
) -> dict[str, float | int | None]:
    """Query-vs-gallery retrieval, restricted to this epoch's validation split:
    one L2-normalized mean prototype per train-side cow_id (the gallery),
    every val embedding scored against all prototypes (the queries). Same
    shape gallery.py::identify_run uses in production. This -- not
    classification accuracy -- is what checkpoint selection is based on, per
    the instruction that checkpoint choice must reflect target retrieval
    success rather than classification accuracy.
    """
    if len(val_cow_ids) == 0 or len(train_cow_ids) == 0:
        return {"top1": None, "mAP": None, "queries": 0}
    train_cow_array = np.asarray(train_cow_ids)
    unique_cows = sorted(set(train_cow_ids))
    prototypes = np.vstack(
        [l2_normalize(np.mean(train_embeddings[train_cow_array == cow_id], axis=0)) for cow_id in unique_cows]
    )
    cow_index = {cow_id: index for index, cow_id in enumerate(unique_cows)}
    scores = val_embeddings @ prototypes.T
    top1_hits = 0
    average_precisions: list[float] = []
    eligible = 0
    for row, cow_id in zip(scores, val_cow_ids):
        if cow_id not in cow_index:
            continue  # no train-side prototype for this cow -- can't evaluate retrieval
        eligible += 1
        order = np.argsort(row)[::-1]
        ranked_cows = np.asarray(unique_cows)[order]
        relevant = ranked_cows == cow_id
        top1_hits += int(relevant[0])
        positive_ranks = np.flatnonzero(relevant) + 1
        precision_at_positive = np.cumsum(relevant)[positive_ranks - 1] / positive_ranks
        average_precisions.append(float(np.mean(precision_at_positive)))
    if eligible == 0:
        return {"top1": None, "mAP": None, "queries": 0}
    return {"top1": top1_hits / eligible, "mAP": float(np.mean(average_precisions)), "queries": eligible}


def train_reid(
    run_dir: str | Path,
    labels_csv: str | Path,
    config: dict[str, Any],
    output_path: str | Path | None = None,
) -> Path:
    """Fine-tune a cattle Re-ID embedding with CE plus batch-hard metric loss."""

    import torch
    from torch.utils.data import DataLoader

    run_dir = Path(run_dir).resolve()
    labels_path = Path(labels_csv).resolve()
    audit_outputs = audit_labels(run_dir, labels_path)
    labels = pd.read_csv(audit_outputs["safe_labels"], dtype={"cow_id": str})
    tracks = pd.read_csv(run_dir / "tracklets.csv")
    frames = pd.read_csv(run_dir / "frames.csv")

    splitting_cfg = config.get("splitting", {}) or {}
    test_dates = [str(value) for value in (splitting_cfg.get("test_dates") or [])]
    validation_dates = [str(value) for value in (splitting_cfg.get("validation_dates") or [])]
    open_set_cow_ids = {str(value) for value in (splitting_cfg.get("open_set_cow_ids") or [])}
    video_split = _load_video_split(run_dir, test_dates, validation_dates)
    excluded_tracklets = load_excluded_tracklets(run_dir)
    split = _derive_tracklet_split(labels, tracks, video_split, excluded_tracklets, open_set_cow_ids)
    classes = sorted(split.loc[split["split"].isin(["train", "val"]), "cow_id"].unique().tolist())
    if len(classes) < 2:
        raise RuntimeError("Need conflict-free tracklets for at least two different Cow_ID values.")

    feature_column = "embedding_path" if "embedding_path" in frames.columns else "torso_path"
    frame_rows = frames[["tracklet_id", feature_column]].rename(columns={feature_column: "image_path"})
    samples = split.merge(frame_rows, on="tracklet_id", how="inner")
    samples = samples.loc[samples["image_path"].map(lambda value: Path(str(value)).exists())].copy()
    train_rows = samples.loc[samples["split"].eq("train")].reset_index(drop=True)
    val_rows = samples.loc[samples["split"].eq("val")].reset_index(drop=True)
    if train_rows.empty or val_rows.empty:
        raise RuntimeError("The safe labels did not resolve to both training and validation crop files.")

    training_cfg = config.get("training", {})
    seed = int(training_cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _resolve_device(str(training_cfg.get("device", "auto")))
    epochs = int(training_cfg.get("epochs", 30))
    learning_rate = float(training_cfg.get("learning_rate", 0.00001))
    weight_decay = float(training_cfg.get("weight_decay", 0.0001))
    embedding_dim = int(training_cfg.get("embedding_dim", 128))
    metric_mode = str(training_cfg.get("metric_loss", "reciprocal"))
    metric_weight = float(training_cfg.get("metric_weight", 0.01))
    triplet_margin = float(training_cfg.get("triplet_margin", 0.2))
    identities_per_batch = int(training_cfg.get("identities_per_batch", 4))
    instances_per_identity = int(training_cfg.get("instances_per_identity", 4))
    stride_layout = str(training_cfg.get("stride_layout", "torchvision"))
    freeze_backbone = bool(training_cfg.get("freeze_backbone", False))

    pretrained_value = training_cfg.get("pretrained_checkpoint")
    pretrained = resolve_model_path(str(pretrained_value), config) if pretrained_value else None
    require_pretrained = bool(training_cfg.get("require_pretrained", True))
    if pretrained and not Path(pretrained).exists():
        raise FileNotFoundError(
            f"Cattle pretrained checkpoint not found: {pretrained}. Run: cow-reid download-cattle-weights"
        )
    if require_pretrained and not pretrained:
        raise RuntimeError("A cattle pretrained checkpoint is required. Run: cow-reid download-cattle-weights")

    if pretrained:
        input_profile = str(training_cfg.get("input_profile", "opencows2020_legacy"))
        model = build_metric_resnet50(len(classes), embedding_dim=embedding_dim, imagenet=False, stride_layout=stride_layout)
        initialization = load_opencows2020_initialization(model, pretrained)
    else:
        input_profile = str(training_cfg.get("input_profile", "imagenet"))
        model = build_metric_resnet50(len(classes), embedding_dim=embedding_dim, imagenet=True, stride_layout=stride_layout)
        initialization = {"source": "ImageNet torchvision", "loaded_tensors": 0, "skipped_tensors": 0}

    class_to_index = {cow_id: index for index, cow_id in enumerate(classes)}
    index_to_class = {index: cow_id for cow_id, index in class_to_index.items()}
    train_dataset = _CropDataset(train_rows, class_to_index, build_metric_transform(input_profile, training=True))
    val_dataset = _CropDataset(val_rows, class_to_index, build_metric_transform(input_profile, training=False))
    # Evaluated with the same no-augmentation transform as val, purely to get
    # a clean per-tracklet train-side embedding for the retrieval prototypes
    # below -- distinct from train_loader, which uses PK-sampling and
    # augmentation for the actual gradient steps.
    train_eval_dataset = _CropDataset(train_rows, class_to_index, build_metric_transform(input_profile, training=False))
    sampler = _PKBatchSampler(train_rows, identities_per_batch, instances_per_identity, seed=seed)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=0)
    eval_batch_size = max(1, identities_per_batch * instances_per_identity)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=0)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=0)

    model.to(device)
    if freeze_backbone:
        _freeze_backbone(model)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
    classifier_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    best_metric = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_val_tracklet_accuracy = -1.0
    best_retrieval: dict[str, float | int | None] = {"top1": None, "mAP": None, "queries": 0}
    history: list[dict[str, float | int | None]] = []
    print(
        f"[train] model=metric_resnet50 stride_layout={stride_layout} freeze_backbone={freeze_backbone} "
        f"device={device} cows={len(classes)} "
        f"train_tracks={split.loc[split['split'].eq('train'), 'tracklet_id'].nunique()} "
        f"val_tracks={split.loc[split['split'].eq('val'), 'tracklet_id'].nunique()}",
        flush=True,
    )

    def _collect(loader) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]], dict[str, int]]:
        logits_by_track: dict[str, list[np.ndarray]] = defaultdict(list)
        embeddings_by_track: dict[str, list[np.ndarray]] = defaultdict(list)
        target_by_track: dict[str, int] = {}
        with torch.inference_mode():
            for images, targets, tracklet_ids in loader:
                embedding, logits = model(images.to(device))
                embedding_values = embedding.detach().cpu().numpy()
                logit_values = logits.detach().cpu().numpy()
                for tracklet_id, row_logits, row_embedding, target in zip(
                    tracklet_ids, logit_values, embedding_values, targets.numpy().tolist()
                ):
                    logits_by_track[str(tracklet_id)].append(row_logits)
                    embeddings_by_track[str(tracklet_id)].append(row_embedding)
                    target_by_track[str(tracklet_id)] = int(target)
        return logits_by_track, embeddings_by_track, target_by_track

    for epoch in range(1, epochs + 1):
        model.train()
        if freeze_backbone:
            # model.train() recursively re-enables BatchNorm running-stat
            # updates on every submodule, including the frozen backbone --
            # re-pin it to eval() so a tiny/skewed batch can't drift its
            # running statistics even though its weights don't update.
            model.backbone.eval()
        total_loss = total_ce = total_metric = 0.0
        total_items = 0
        for images, targets, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            embedding, logits = model(images)
            ce_loss = classifier_loss(logits, targets)
            metric_loss = _batch_hard_metric_loss(embedding, targets, metric_mode, triplet_margin)
            loss = ce_loss + metric_weight * metric_loss
            loss.backward()
            optimizer.step()
            items = len(images)
            total_loss += float(loss.detach().cpu()) * items
            total_ce += float(ce_loss.detach().cpu()) * items
            total_metric += float(metric_loss.detach().cpu()) * items
            total_items += items

        model.eval()
        val_logits, val_embeddings_by_track, val_targets = _collect(val_loader)
        train_logits, train_embeddings_by_track, train_targets = _collect(train_eval_loader)

        correct = sum(
            int(np.mean(np.vstack(track_logits), axis=0).argmax()) == val_targets[tracklet_id]
            for tracklet_id, track_logits in val_logits.items()
        )
        validation_accuracy = correct / max(len(val_logits), 1)

        val_tracklet_ids = list(val_embeddings_by_track.keys())
        val_pooled = l2_normalize(np.vstack([np.mean(np.vstack(val_embeddings_by_track[t]), axis=0) for t in val_tracklet_ids]))
        val_cow_ids = [index_to_class[val_targets[t]] for t in val_tracklet_ids]
        train_tracklet_ids = list(train_embeddings_by_track.keys())
        train_pooled = l2_normalize(np.vstack([np.mean(np.vstack(train_embeddings_by_track[t]), axis=0) for t in train_tracklet_ids]))
        train_cow_ids = [index_to_class[train_targets[t]] for t in train_tracklet_ids]
        retrieval = _prototype_retrieval_metrics(train_pooled, train_cow_ids, val_pooled, val_cow_ids)

        # Checkpoint selection is based on retrieval success against a
        # train-side gallery -- the target-deployment metric -- not
        # classification accuracy; classification accuracy is still tracked
        # and reported for comparison, and used as a fallback only when a
        # split is too degenerate to produce any eligible retrieval query.
        selection_metric = retrieval["top1"] if retrieval["top1"] is not None else validation_accuracy

        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(total_items, 1),
                "classification_loss": total_ce / max(total_items, 1),
                "metric_loss": total_metric / max(total_items, 1),
                "val_tracklet_accuracy": validation_accuracy,
                "retrieval_top1": retrieval["top1"],
                "retrieval_mAP": retrieval["mAP"],
                "retrieval_queries": retrieval["queries"],
            }
        )
        print(
            f"[train] epoch={epoch}/{epochs} loss={history[-1]['train_loss']:.4f} "
            f"metric={history[-1]['metric_loss']:.4f} val_track_acc={validation_accuracy:.3f} "
            f"retrieval_top1={retrieval['top1']} retrieval_mAP={retrieval['mAP']} (n={retrieval['queries']})",
            flush=True,
        )
        if selection_metric > best_metric:
            best_metric = selection_metric
            best_val_tracklet_accuracy = validation_accuracy
            best_retrieval = retrieval
            best_state = deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    output = Path(output_path).resolve() if output_path else run_dir / "farm_metric_resnet50.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": "metric_resnet50",
            "state_dict": best_state,
            "class_names": classes,
            "embedding_dim": embedding_dim,
            "input_profile": input_profile,
            "stride_layout": stride_layout,
            "freeze_backbone": freeze_backbone,
            "initialization": initialization,
            "metric_loss": metric_mode,
            "metric_weight": metric_weight,
            "best_val_tracklet_accuracy": best_val_tracklet_accuracy,
            "best_retrieval_top1": best_retrieval["top1"],
            "best_retrieval_mAP": best_retrieval["mAP"],
            "best_retrieval_queries": best_retrieval["queries"],
        },
        output,
    )
    split.to_csv(run_dir / "training_split.csv", index=False)
    save_json(
        run_dir / "training_summary.json",
        {
            "checkpoint": str(output),
            "model": "metric_resnet50",
            "embedding_dim": embedding_dim,
            "initialization": initialization,
            "input_profile": input_profile,
            "stride_layout": stride_layout,
            "freeze_backbone": freeze_backbone,
            "metric_loss": metric_mode,
            "metric_weight": metric_weight,
            "cow_ids": len(classes),
            "train_tracklets": int(split.loc[split["split"].eq("train"), "tracklet_id"].nunique()),
            "validation_tracklets": int(split.loc[split["split"].eq("val"), "tracklet_id"].nunique()),
            "train_crops": int(len(train_rows)),
            "validation_crops": int(len(val_rows)),
            "best_val_tracklet_accuracy": best_val_tracklet_accuracy,
            "best_retrieval_top1": best_retrieval["top1"],
            "best_retrieval_mAP": best_retrieval["mAP"],
            "best_retrieval_queries": best_retrieval["queries"],
            "device": device,
            "epochs": epochs,
            "audit_summary": str(audit_outputs["summary"]),
            "history": history,
        },
    )
    return output
