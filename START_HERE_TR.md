# Anıl — buradan devam et

Şimdilik daha fazla çift etiketleme. Mevcut `runs/all_videos_120s` korunacak.

## Güvenli güncelleme

```bash
cd /Users/anil/Downloads
unzip cow_reid_pipeline_v0.3.0.zip

rsync -a \
  --exclude '.venv/' \
  --exclude 'runs/' \
  --exclude 'weights/' \
  --exclude 'models/' \
  cow_reid_pipeline_v0.3.0/ cow_reid_pipeline/

cd cow_reid_pipeline
source .venv/bin/activate
python -m pip install -e ".[deep,ui]"
```

Finder'da eski klasöre `Replace` deme; bu yöntem `.venv`, `runs` ve model
ağırlıklarını korur.

## Sırayla çalıştır

```bash
cow-reid audit-labels \
  --run runs/all_videos_120s \
  --labels runs/all_videos_120s/labels.csv
```

```bash
cow-reid review --run runs/all_videos_120s
```

Yalnızca `Çelişki denetimi` sekmesine bak. Handoff'taki karışmış gruplar:
`COW_0009`, `COW_0019`, `COW_0025`, `COW_0028`. Bunları sonra da düzeltebilirsin;
diğer 28 güvenli kimlikle eğitim başlayabilir.

```bash
cow-reid download-cattle-weights
```

```bash
cow-reid --config configs/default.yaml train-reid \
  --run runs/all_videos_120s \
  --labels runs/all_videos_120s/labels.csv \
  --pretrained weights/opencows2020_softmaxrtl.pkl \
  --output weights/farm_metric_resnet50.pt \
  --device mps \
  --epochs 30
```

```bash
cow-reid --config configs/default.yaml embed \
  --run runs/all_videos_120s \
  --backend metric \
  --checkpoint weights/farm_metric_resnet50.pt

cow-reid --config configs/default.yaml cluster \
  --run runs/all_videos_120s

cow-reid evaluate-reid \
  --run runs/all_videos_120s \
  --labels runs/all_videos_120s/labels.csv

cow-reid review --run runs/all_videos_120s
```

Son arayüzde `İnek odaklı etiketle` sekmesine geç. Bir COW seçildiğinde başka
videolardan yalnızca en yakın altı aday gösterilir; binlerce çift gezilmez.

`weights/opencows2020_softmaxrtl.pkl` public cattle-ReID başlangıcıdır.
`weights/farm_metric_resnet50.pt` ise senin güvenli etiketlerinle oluşan modeldir.
YOLO yalnızca detection/tracking yapar.
