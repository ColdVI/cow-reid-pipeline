# Keypoint → Cow Identity → Longitudinal Posture Integration

## Amaç

Kimlik modeli ile keypoint modeli aynı işi yapmaz:

- Re-ID hattı bir geçişe `tracklet_id`, insan doğrulamasından sonra `cow_id`
  verir.
- Keypoint hattı aynı geçişin frame'lerinde anatomik noktaları üretir.
- Geometri/skorlama katmanı bu noktalardan frame-level `arch_score` üretir.
- Postür deposu skorları `tracklet_id` üzerinden doğru `cow_id` geçmişine ekler.

```text
source video + tracklet time range
  -> keypoint inference at original FPS
  -> per-frame keypoints and visibility
  -> quality/visibility gate
  -> per-frame arch score
  -> robust tracklet aggregation
  -> cow_id longitudinal history
```

## Minimum adapter çıktısı

Keypoint reposu doğrudan skor üretebiliyorsa minimum çıktı:

```csv
tracklet_id,frame_idx,timestamp_s,arch_score,model_name,model_version
vid_abc_t0007,1920,76.80,0.34,cow_keypoints,v1
```

`LamenessDLCBackend` koordinatları şu kanonik uzun biçimde yazar:

```csv
tracklet_id,frame_idx,timestamp_s,keypoint_name,x_px,y_px,x_normalized,y_normalized,visibility,model_name,model_version
vid_abc_t0007,1920,76.80,withers,521.2,214.9,0.43,0.12,0.96,lameness_custom_hrnet_w32,snapshot-best-130:sha12
```

Modelin 15 noktası doğrudan korunur. Dorsal skor `withers`,
`thoracic_spine`, `dorsal_apex`, `lumbar_spine`, `sacrum` zincirinden
hesaplanır.

## Tracklet klibinin hazırlanması

`tracklets/<video>/<tracklet>/metadata.json` şu bilgileri içerir:

- kaynak video yolu;
- geçiş başlangıç/bitiş saniyesi;
- örneklenen frame numaraları ve bounding box'lar;
- detector confidence ve trajectory.

Keypoint inference için yalnızca Re-ID'de saklanan 12 torso karesi kullanılmaz.
Kaynak videonun ilgili zaman aralığı orijinal FPS'te okunur; track bounding box
değerleri ara frame'lere interpolate edilir. Lameness reposunun YOLO tracker'ı
yeniden çalıştırılmaz; böylece Re-ID ile pose aynı tracklet sınırını kullanır ve
bacak/sırt hareketi kaybolmaz.

## Kalite kapıları

Bir frame aşağıdakilerden biri gerçekleşirse tracklet skoruna katılmamalıdır:

- gerekli sırt keypoint'larının görünürlüğü eşik altında;
- iki inek örtüşüyor;
- gövdenin kritik bölgesi kadraj dışında;
- tracker kimlik sıçraması şüphesi var;
- keypoint anatomik sırası bozuk.

Tracklet özeti tek kötü frame'in maksimumuyla değil, temiz frame'lerin robust
istatistikleriyle oluşturulur. Paket şu anda mean, median, p90, max, standard
deviation ve geçerli frame sayısını saklar.

## Kişisel baseline

İlk üç doğrulanmış ölçümden sonra geçmiş ölçümlerin expanding median değeri
kişisel baseline olarak kullanılır. Ölçülen değişim:

```text
delta = current_track_arch_score - previous_measurements_median
```

Mevcut ölçüm baseline hesabına katılmaz. Bu değer şimdilik araştırma sinyalidir;
alarm veya hastalık kararı için veteriner doğrulaması ve sahada kalibrasyon
gerekir.

## Uygulanan lameness-main adapter'ı

1. Ana uygulama Python ortamı `cow_reid.pose.worker` sürecini ayrı Python 3.12
   DeepLabCut ortamında başlatır.
2. Her orijinal video frame'i interpolate edilmiş tracklet kutusu etrafında %15
   payla kırpılır ve 448×448 RGB modele verilir.
3. Noktalar kaynak video koordinatlarına taşınır; lameness-main anatomik
   düzeltmesi ve görünürlük kapıları uygulanır.
4. `keypoints.csv`, `quality.json` ve temiz dorsal frameler için
   `posture_scores.csv` yazılır; skorlar `tracklet_id` üzerinden kimlik geçmişine
   aktarılır.
5. Model sürümü checkpoint adı ve SHA-256 özetiyle saklanır; farklı sürümler
   sessizce aynı longitudinal seri gibi karşılaştırılmaz.

Mevcut indirilen repo içindeki iki checkpoint 134 baytlık Git LFS pointer'ıdır.
Adapter bunu inference öncesinde algılar. Asıl yaklaşık 118 MB ağırlık dosyası
aynı konuma konulduğunda `cow-reid pose-run` ek kod değişmeden çalışır.
