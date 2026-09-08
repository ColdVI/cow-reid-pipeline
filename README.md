# Cow Tracklet Re-ID + Longitudinal Posture Pipeline v1.0.0

Sabit sağımhane kamerasındaki inek geçişlerini tracklet'lara ayıran, gövde/benek
görüntülerinden kimlik arayan ve daha sonra keypoint/kamburluk skorlarını aynı
kimlik geçmişine bağlayan araştırma pipeline'ı.

## ⚠️ Bilinen sızıntı: `weights/farm_metric_resnet50.pt` yalnızca tarihsel

`runs/all_videos_120s` altında eğitilmiş mevcut `weights/farm_metric_resnet50.pt`
checkpoint'inin raporlanan `best_val_tracklet_accuracy` değeri **gerçek
görülmemiş-gün doğruluğu değildir**. `training_split.csv` incelendiğinde, 12/14
etiketli ineğin bütün `val` bölümü `vid_eef4a44f4b99`'dan geliyor; bu video da
`exports/current_14_manifest_v1.csv`'ye göre `vid_e2b59d770f3b`'nin (aynı ineklerin
`train` bölümünün kaynağı) 17 saniye kaydırılmış aynı kaydıdır — eski
`training._tracklet_split` bu örtüşmeyi göremiyordu (session metnini
lexicographic sıralayıp sonuncusunu val seçiyordu, gerçek tarih veya
`overlap_group_id` kullanmıyordu). Bu sorun bu geliştirme turunda `cow_reid/overlap.py`,
`cow_reid/manifest.py` ve `cow_reid/training.py`'deki örtüşme/tarih-farkında
ayrım mantığıyla düzeltildi (bkz. aşağıdaki `cow-reid build-manifest` adımı),
fakat **mevcut checkpoint temiz ayrımla yeniden eğitilmedi**. Bu checkpoint'i
yalnızca tarihsel karşılaştırma noktası olarak kullanın; üretim/galeri
kararları için önce `cow-reid build-manifest` ile kanonik manifesti üretip
temiz ayrımla yeniden eğitim yapın.

## v1 mimarisi

- Kamera zamanı başlangıç/orta/son kare OCR sonuçlarının doğrusal tutarlılığıyla
  kabul edilir. Dosya adı yalnızca indirme metadata'sıdır; kayıt zamanı değildir.
- SHA-256, zaman aralığı ve görsel parmak izi birlikte tutulur. Örtüşen klipler
  aynı `overlap_group_id` altında kalır.
- Registry; video durum makinesi, kalıcı `cow_uuid`, alias, identity edge,
  keypoint run ve append-only postür tablolarını içerir.
- Gallery, inek başına birden fazla doğrulanmış tracklet prototipi taşır. Eşik
  veya Top-2 margin yetersizse sonuç `UNKNOWN`/`AMBIGUOUS` olur.
- `cow_reid.pose` harici keypoint repoları için adapter sözleşmesini;
  `cow_reid.health` kalite özeti, dorsal arch ve rolling median/MAD baseline'ını
  sağlar.

PoC registry standart kütüphane ile SQLite üzerinde doğrudan çalışır. Üretim
PostgreSQL + pgvector şeması `migrations/postgresql.sql` dosyasındadır.

```bash
cow-reid db-init --database sqlite:///data/cow_reid.db

cow-reid ingest \
  --videos data/incoming \
  --camera side_01 \
  --database sqlite:///data/cow_reid.db \
  --manifest data/video_manifest.csv

cow-reid timestamp-import \
  --manifest exports/current_14_manifest_v1.csv \
  --database postgresql:///cow_reid

# Reconcile the raw inventory manifest with reviewed correction exports into
# one canonical manifest -- real overlap groups, real dates, before extract/
# train ever see the data. See the leakage warning above for why this matters.
cow-reid build-manifest \
  --raw data/video_manifest.csv \
  --corrections exports/current_14_manifest_v1.csv \
  --output data/canonical_manifest.csv

cow-reid import-run \
  --run runs/all_videos_120s \
  --checkpoint weights/farm_metric_resnet50.pt \
  --database postgresql:///cow_reid

cow-reid process-new \
  --database sqlite:///data/cow_reid.db \
  --checkpoint weights/farm_metric_resnet50.pt \
  --backend metric \
  --output data/processed \
  --device mps

cow-reid health-refresh --run runs/all_videos_120s
```

`/Users/anil/Downloads/lameness-main` içindeki özel 15 noktalı DeepLabCut modeli
`LamenessDLCBackend` ile bağlanmıştır. Pose işi ayrı Python 3.12 ortamında çalışır;
tracklet kutuları doğrudan crop olarak kullanıldığı için ikinci kez YOLO tracking
çalıştırılmaz.

## v0.3'te önemli değişiklikler

- YOLO yalnızca **inek tespiti ve takibi** yapar; kimlik modeli değildir.
- Re-ID başlangıcı artık genel ImageNet ResNet-18 değil, OpenCows2020 üzerinde
  eğitilmiş ResNet-50 Softmax+Reciprocal Triplet Loss ağırlığıdır.
- Çiftliğe fine-tune sırasında classification + batch-hard metric loss kullanılır;
  çıktı 128 boyutlu normalize inek embedding'idir.
- Binlerce çift sırayla etiketlenmez. Bir `COW_xxxx` seçilir ve başka videolardan
  yalnızca en yakın 3–12 aday getirilir.
- Eski `different` kararları cannot-link kısıtıdır; çelişkili kimlikler silinmeden
  eğitimden karantinaya alınır.
- Label ve eski embedding dosyaları otomatik yedeklenir.

| Kimlik | Anlamı |
|---|---|
| `tracklet_id` | Tracker'ın tek video geçişine verdiği teknik kimlik |
| `COW_0001` | Aynı hayvan olduğu insan tarafından doğrulanan geçici kimlik |
| Gerçek çiftlik ID | RFID/küpe geldikten sonra geçici kimliğin yeni adı |

Trajectory doğrudan kimlik değildir. Tracker aynı geçişin karelerini toplar;
Re-ID modeli o karelerdeki gövde/benek imzasını kullanır.

## Kurulum ve güvenli güncelleme

Mac'te Finder ile eski klasöre **Replace** demek `runs/`, `.venv/` ve `weights/`
klasörlerini silebilir. ZIP'i ayrı açıp kaynak kodu birleştirin:

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
python -m pip install --upgrade pip
python -m pip install -e ".[deep,ui]"
```

Mevcut run için tam komut sırası `START_HERE_TR.md` içindedir.

## GitHub'dan sıfır kurulum

Bu repo Git LFS ile geliyor — büyük checkpoint dosyası (`weights/ablation/
20260907_stage_a_torchvision_finetune.pt`, ~98MB) normal git blob değil, LFS
pointer olarak tutuluyor. `git lfs` kurulu değilse önce onu kur (`brew install
git-lfs` / `apt install git-lfs`), sonra:

```bash
git clone <bu-repo-url> cow_reid_pipeline
cd cow_reid_pipeline
git lfs install
git lfs pull        # LFS pointer'ları gerçek checkpoint'e çevirir

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[deep,ui]"

# Repoda bulunmayan iki genel-amaçlı ağırlık:
cow-reid download-cattle-weights          # OpenCows2020, weights/opencows2020_softmaxrtl.pkl
# yolo11n-seg.pt ilk `cow-reid extract` çalıştığında ultralytics tarafından
# otomatik indirilir; istersen elle de repo köküne koyabilirsin.

# Hızlı doğrulama -- gerçek video gerekmez:
python -m pytest -q
```

`weights/README.md` hangi checkpoint'in repoda olduğunu, hangilerinin bilerek
dışarıda bırakıldığını (eski/sızıntılı checkpoint, kaybeden ablasyon
varyantları) ve neden olduğunu anlatıyor. Gerçek çiftlik videosu, DB dökümü ve
insan etiket verisi bu repoda **yok** — `git clone` sonrası kendi videolarınla
`cow-reid run` / `process-new` akışını çalıştırman gerekiyor; bkz. "Gallery ve
yeni gün kimliklendirmesi" bölümü.

## Etiket denetimi ve arayüz

```bash
cow-reid audit-labels \
  --run runs/all_videos_120s \
  --labels runs/all_videos_120s/labels.csv

cow-reid review --run runs/all_videos_120s
```

Üretilen `label_audit.csv`, `label_conflicts.csv` ve
`labels_training_safe.csv` dosyaları hiçbir kullanıcı etiketini silmez. Handoff
denetiminde 32 geçici kimliğin 28'i ve 69 tracklet doğrudan güvenli bulundu;
`COW_0009`, `COW_0019`, `COW_0025`, `COW_0028` karantinaya alındı.

Arayüzde ana akış:

1. `Video ile inek kontrolü`: örneğin `COW_0012` ara ve gerçek tarihe göre
   sıralanmış tüm video geçişlerini izle.
2. Debug klipte tracklet'tan iki saniye önce başlayan kutu, trajectory, kimlik,
   durum ve güven bilgisini gör. Kaynak video değiştirilmez; klip cache'lenir.
3. `Aynı inek`, `Farklı inek`, `Emin değilim`, `Tracklet hatalı` veya
   `Geçişi böl` kararı ver. Bölme ve tracker düzeltme kararları kuyruğa yazılır.
4. Torso/contact sheet'i yalnızca yardımcı yakın plan kanıtı olarak kullan.

`Eski çift kuyruğu` opsiyoneldir; tamamlanması gerekmez.

## Cattle-pretrained ağırlığı ve fine-tune

```bash
cow-reid download-cattle-weights
```

Bu komut yaklaşık 103 MB public ağırlığı
`weights/opencows2020_softmaxrtl.pkl` konumuna indirir ve SHA-256 doğrular.
Ağırlık ZIP'e gömülmez.

```bash
cow-reid --config configs/default.yaml train-reid \
  --run runs/all_videos_120s \
  --labels runs/all_videos_120s/labels.csv \
  --pretrained weights/opencows2020_softmaxrtl.pkl \
  --output weights/farm_metric_resnet50.pt \
  --device mps \
  --epochs 30
```

Audit eğitimden hemen önce otomatik tekrar çalışır. Çelişkili COW'lar checkpoint'e
girmez. Mac M4 için `mps`, NVIDIA için `cuda`, GPU yoksa `cpu` kullanılabilir.

## Yeni embedding, clustering ve değerlendirme

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
```

Eski `track_embeddings.npz`, yeniden embedding öncesinde
`embedding_backups/<timestamp>/` altına kopyalanır. Değerlendirme track-level
Top-1, Top-5 ve mAP üretir ve aynı session'ı aday yapmaz. Aynı etiketler eğitime
de girdiyse bu son doğruluk değildir; deployment öncesi görülmemiş başka bir gün
tamamen test olarak tutulmalıdır.

## Gallery ve yeni gün kimliklendirmesi (insansız, otomatik akış)

`runs/ablation_20260907/comparison_report.md`'deki temiz-split ablasyonunda
seçilen checkpoint (`weights/ablation/20260907_stage_a_torchvision_finetune.pt`
— mevcut torchvision stride mimarisi, tam fine-tune; retrieval top1 %92,9,
n=14) ile üretim galerisi kuruldu (`runs/production_gallery_20260907/`, 75
kimlik, 173 prototip — bkz. `label_audit_summary.json`). Yeni videoların
kimliklendirmesi bu galeriye karşı **insan tıklaması olmadan** çalışır:

```bash
# Yalnızca doğrulanmış (confirmed) kimliklerden galeri kur — bir defalık,
# checkpoint her değiştiğinde yeniden çalıştırılır.
cow-reid embed --run <run> --backend metric \
  --checkpoint weights/ablation/20260907_stage_a_torchvision_finetune.pt
cow-reid build-gallery --run <run> --labels <run>/labels.csv \
  --output <run>/cow_gallery.npz

# Yeni gün videoları için: extract + embed (aynı checkpoint) + identify —
# hiçbiri insan onayı beklemez.
cow-reid --config configs/default.yaml extract --manifest data/canonical_manifest_full.csv --output runs/new_day
cow-reid embed --run runs/new_day --backend metric \
  --checkpoint weights/ablation/20260907_stage_a_torchvision_finetune.pt
cow-reid identify --run runs/new_day \
  --gallery runs/production_gallery_20260907/cow_gallery.npz
```

Gallery ve query farklı checkpoint/ön-işleme/crop sürümüyle üretilmişse sistem
tahmin yapmaz, açık hatayla durur (`gallery.py::identify_run`'ın içerik
tabanlı sürüm sözleşmesi — dosya yolu değil SHA-256/preprocessing_profile/
crop_version/embedding_schema_version karşılaştırılır). Eşiğin altındaki
sorgu zorla kimliğe bağlanmak yerine `UNKNOWN` kalır; iki güçlü aday
birbirine çok yakınsa `AMBIGUOUS` kalır. **Otomatik tahminler hiçbir zaman
`labels.csv`'ye veya galeriye yazılmaz** — yalnızca `identity_predictions.csv`/
`identity_summary.json` içinde kalır; galeri/eğitim etiketleri yalnızca insan
onayından geçen tracklet'lerden güncellenir.

Önemli sınır: eşik (`matching.unknown_threshold`, halen 0,95) henüz ayrı bir
kalibrasyon verisiyle kalibre edilmedi — sabit, temkinli bir başlangıç
değeri. Gerçek veride birçok yüksek-benzerlikli doğru eşleşme (ör. 0,947,
0,923 benzerlik) bu yüzden `UNKNOWN` dönüyor; bu kasıtlı ve güvenli bir
tercih (yanlış kimlik atamaktansa "bilmiyorum" demek), ama otomatik kapsama
oranını düşürüyor. Kalibrasyon (KNOWN/AMBIGUOUS/UNKNOWN/INSUFFICIENT_EVIDENCE,
ayrı kalibrasyon verisiyle) sonraki aşamanın konusu — bkz.
`runs/ablation_20260907/comparison_report.md` ve plan dosyasındaki "Follow-up
roadmap".

## Yeni video işleme

Varsayılan v0.3 config cattle ağırlığını bekler; önce download komutu çalışmalıdır.

```bash
cow-reid --config configs/default.yaml run \
  --videos "./çiftlik kayıtları" \
  --output runs/new_day \
  --max-seconds 300
```

Fine-tune edilmiş checkpoint kullanılacaksa `embed --backend metric --checkpoint
weights/farm_metric_resnet50.pt` verin (`run` komutu tek seferde `--embedder metric`
kabul eder). `process-new` de artık aynı şekilde açık `--backend metric` bekler —
yalnızca `--checkpoint` vermek backend'i otomatik seçmez; uyuşmazlıkta pipeline
hiçbir aşamayı çalıştırmadan hemen açıklayıcı bir hata verir.

## Canlı entegrasyon

Eğitim offline yapılır. Canlı servis daha sonra RTSP/NVR frame'i alır, tracker ile
8–12 temiz torso crop toplar, tek track embedding üretir ve gallery sonucunu
`COW_xxxx` veya `UNKNOWN` olarak kaydeder. Tek frame'de anlık sabit kimlik yerine
önce `collecting…`, yeterli kanıttan sonra kimlik gösterilmelidir.

## Keypoint ve kamburluk geçmişi

Re-ID ile keypoint hattının ortak anahtarı `tracklet_id` değeridir. Harici model
şu minimum CSV'yi verdiğinde sonuç kimlik geçmişine bağlanabilir:

```csv
tracklet_id,frame_idx,timestamp_s,arch_score,model_name,model_version
vid_abc_t0007,1920,76.80,0.34,cow_keypoints,v1
```

```bash
cow-reid import-posture \
  --run runs/all_videos_120s \
  --scores /path/to/keypoint_scores.csv
```

Lameness adapter'ı mevcut tracklet trajectory'sini kullanarak her karede ineği
crop eder, 15 keypoint'i kaynak video koordinatlarına geri taşır, anatomik kalite
kapılarını uygular ve dorsal zincirden postür gözlemleri üretir:

```bash
/opt/homebrew/bin/python3.12 -m venv .venv-dlc
.venv-dlc/bin/python -m pip install deeplabcut 'ultralytics==8.3.253'
.venv-dlc/bin/python -m pip install -e . --no-deps

cow-reid pose-run \
  --run runs/all_videos_120s \
  --repo /Users/anil/Downloads/lameness-main \
  --checkpoint /Users/anil/Downloads/lameness-main/dlc_projects/cow-lameness-mehmet-2026-08-20/dlc-models-pytorch/iteration-0/cow-lamenessAug20-trainset90shuffle1/train/snapshot-best-130.pt \
  --python .venv-dlc/bin/python \
  --tracklet vid_ed1b0d909924_t0228 \
  --device cpu
```

Önemli: mevcut `lameness-main` kopyasındaki `snapshot-*.pt` dosyaları yalnızca
Git LFS pointer'ıdır. Gerçek `snapshot-best-130.pt` (pointer'a göre 118236749
bayt) aynı konuma indirilmeden inference başlamaz; CLI bunu erken ve açık bir
hata ile bildirir. Detaylı genel adapter sözleşmesi
`docs/KEYPOINT_INTEGRATION.md` içindedir.

## Testler

```bash
python -m pip install -e ".[dev]"
pytest
python scripts/smoke_test.py
python scripts/metric_smoke_test.py \
  --checkpoint weights/opencows2020_softmaxrtl.pkl
```

## Kaynaklar ve lisans

- OpenCows2020 paper: <https://arxiv.org/abs/2006.09205>
- Public code/weights: <https://github.com/CWOA/MetricLearningIdentification>
- Cows2021 tracklet self-supervision: <https://arxiv.org/abs/2105.01938>
- CowIDentifier: <https://github.com/Phoenix4582/CowIDentifier>

OpenCows2020 ağırlığı ZIP içinde dağıtılmaz. Ultralytics bileşenleri AGPL-3.0
veya Enterprise koşullarına tabidir. Ayrıntılar `THIRD_PARTY_NOTICES.md` içindedir.
