# Devralan Ekip İçin Güncel Başlangıç

Bu dosya tarihsel `START_HERE_TR.md` yerine 2026-09-11 itibarıyla önerilen ilk çalışma akışını özetler.

## Önce oku

1. `docs/PROJECT_HANDOFF_2026-09-11.md`
2. `VALIDATION_NOTES.md`
3. `weights/README.md`
4. `configs/default.yaml`

## Kritik model uyarısı

Kullanılacak mevcut ReID checkpoint'i:

```text
weights/ablation/20260907_stage_a_torchvision_finetune.pt
```

Eski `weights/farm_metric_resnet50.pt` leaky train/validation split nedeniyle yalnız tarihsel referanstır.

## Temiz kurulum

```bash
git clone <repo-url> cow_reid_pipeline
cd cow_reid_pipeline
git lfs install
git lfs pull
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[deep,ui,dev]"
cow-reid download-cattle-weights
python -m pytest -q
```

## Yeni gün işleme

Önce ham video inventory + reviewed timestamp düzeltmelerinden kanonik manifest oluşturun. Dosya adını recording time kabul etmeyin.

```bash
cow-reid build-manifest \
  --raw data/video_manifest.csv \
  --corrections exports/reviewed_manifest.csv \
  --output data/canonical_manifest.csv

cow-reid --config configs/default.yaml extract \
  --manifest data/canonical_manifest.csv \
  --output runs/new_day

cow-reid embed \
  --run runs/new_day \
  --backend metric \
  --checkpoint weights/ablation/20260907_stage_a_torchvision_finetune.pt

cow-reid identify \
  --run runs/new_day \
  --gallery <gallery-dir>/cow_gallery.npz

cow-reid review --run runs/new_day
```

## İnsan kontrolü

Bütün frameleri veya binlerce pair'i label'lamayın. İnsan review şu durumlara odaklanmalıdır:

- `UNKNOWN`
- `AMBIGUOUS`
- tracking/detection hatası
- yeni cow identity şüphesi
- periyodik `KNOWN` spot-check

Model tahmini `human_labeled` değildir. Yalnız reviewer tarafından doğrulanan kimlikler training/gallery ground truth'a alınmalıdır.

## Debug

`cow-reid review` içindeki **Video ile inek kontrolü** sekmesinde `COW_0012` gibi bir kimlik arayın. Farklı video/gün tracklet'larını izleyebilir; bbox, trajectory, identity/status/confidence overlay bulunan debug klibi açabilirsiniz.

## Re-training ne zaman?

Her yeni video geldiğinde model eğitmeyin. Yeterli yeni gün, confirmed identity veya belirgin domain shift biriktiğinde yeniden eğitim planlayın. Yeniden eğitimden sonra gallery aynı checkpoint ve preprocessing contract ile yeniden embed edilmelidir.

## İlk gerçek araştırma görevi

Tamamen held-out bir gün üzerinde human ground truth oluşturun; closed-set retrieval yanında open-set unknown rejection ve threshold/margin calibration yapın. Temiz ablasyon sonucu `%92.9 Top-1 (13/14)` yalnız küçük-n yönsel bulgudur, production SLA değildir.
