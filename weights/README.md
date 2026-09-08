# Model weights

## Included in this repo (via Git LFS)

- `ablation/20260907_stage_a_torchvision_finetune.pt` — the current, working
  Re-ID checkpoint (torchvision ResNet50 stride layout, full fine-tune,
  winner of the 2×2 ablation on 2026-09-07 — see `../VALIDATION_NOTES.md`).
  This is what `cow-reid embed --backend metric --checkpoint
  weights/ablation/20260907_stage_a_torchvision_finetune.pt` and the
  `identify`/`process-new` recipe in the main README expect.

Run `git lfs pull` after cloning if the file above looks like a small text
pointer instead of a ~98MB binary.

## Not included — fetch these instead

- **`opencows2020_softmaxrtl.pkl`** (public OpenCows2020 checkpoint, used as
  the fine-tune starting point): `cow-reid download-cattle-weights`
- **`yolo11n-seg.pt`** (Ultralytics detector, used for tracklet extraction):
  auto-downloaded by `ultralytics` on first `cow-reid extract` run if not
  present at the repo root. Not covered by this repo's license either way —
  review the Ultralytics license before use.

## Not included — superseded/experimental, kept local only

- `farm_metric_resnet50.pt` — the pre-foundation-pass checkpoint. Documented
  in `VALIDATION_NOTES.md` as trained on a leaky train/val split; kept
  locally only as a historical reference, never as something to build on.
- `ablation/20260907_stage_{b,c,d}_*.pt` — the three losing ablation
  variants (opencows stride, and/or frozen backbone). Useful only if you want
  to re-inspect the ablation itself; not needed to run the model.
- `backups/*.pt` — pre-change safety snapshots from earlier training runs.

If you need any of the above, ask for them directly rather than expecting
them on GitHub — they were left out of this repo on purpose (size and, for
the legacy checkpoint, correctness).
