# Ekli videolarda PoC doğrulaması

## Temiz split üzerinde kademeli ablasyon (2026-09-07)

Veri/ölçüm bütünlüğü turunun (aşağıda) ürettiği düzeltmelerle önce gerçek
kullanılabilir veri miktarı ölçüldü: `runs/review_all_20260903`'ün 121
doğrulanmış kimliğinden yalnızca **29'unun** gerçekten farklı, doğrulanmış
günlerde tracklet'i var (8'inde ≥3 gün). En iyi tek doğrulama günü olarak
her aday gün tek tek denenip **2026-08-25** seçildi (14 kullanılabilir
kimlik; naif "en son gün" politikası yalnızca 4 verirdi).

Bu sabit ayrımda 2×2 ablasyon çalıştırıldı (stride yerleşimi: mevcut
torchvision vs kaynak-uyumlu opencows; backbone: tam fine-tune vs donuk).
Checkpoint seçimi artık sınıflandırma accuracy'si değil, train-günü
prototiplerine karşı val-günü retrieval başarısı (top1/mAP) ile yapılıyor.

| deney | stride | donuk backbone | retrieval top1 | retrieval mAP |
|---|---|---|---:|---:|
| A (seçildi) | torchvision | hayır | **%92,9** (13/14) | %94,6 |
| B | opencows | hayır | %78,6 (11/14) | %88,1 |
| C | torchvision | evet | %85,7 (12/14) | %92,9 |
| D | opencows | evet | %78,6 (11/14) | %86,3 |

Mevcut torchvision stride yerleşimi hem tam fine-tune hem donuk-backbone
koşulunda kaynak-uyumlu opencows yerleşimini geçti — iki bağımsız kıyasta
tutarlı bir yön, fakat n=14 ile tek sorgu ~7 puan oynatıyor; ayrıntı için
`runs/ablation_20260907/comparison_report.md` (hiperparametrelerin mimariye
göre ayrı ayarlanmadığı uyarısı dahil). **n≈14-29, yönsel bulgu — üretim
güvenilirliği iddiası değil.**

Seçilen checkpoint (`weights/ablation/20260907_stage_a_torchvision_finetune.pt`)
ile tam doğrulanmış etiket kümesinden üretim galerisi kuruldu
(`runs/production_gallery_20260907/`, 75 kimlik, 173 prototip). Uçtan uca
`cow-reid identify` çalıştırıldı: `labels.csv` değişmediği doğrulandı
(md5 öncesi/sonrası aynı), 758 sorgudan 164'ü `KNOWN`, 16'sı `AMBIGUOUS`,
594'ü `UNKNOWN` — mekanizma insan tıklaması olmadan çalışıyor. Eşik (0,95)
henüz kalibre edilmedi; birçok yüksek-benzerlikli doğru eşleşme bu yüzden
`UNKNOWN` dönüyor (kasıtlı temkinli davranış, bkz. README).

`weights/farm_metric_resnet50.pt` (eski, sızıntılı split) hâlâ değiştirilmedi
ve yalnızca tarihsel referans olarak duruyor.

## Veri/ölçüm bütünlüğü temel turu (2026-09-07)

Harici inceleme raporunun (`review_20260906/INCELEME_TR.md`) Öncelik-1 bulgusu
doğrulandı ve düzeltildi:

- **Sızıntı doğrulandı**: `runs/all_videos_120s/training_split.csv`'de 12/14
  etiketli ineğin bütün `val` bölümü `vid_eef4a44f4b99`'dan geliyordu; bu video
  `vid_e2b59d770f3b`'nin (aynı ineklerin `train` kaynağı) 17 saniye kaydırılmış
  aynı kaydıydı. `weights/farm_metric_resnet50.pt`'nin raporlanan
  `best_val_tracklet_accuracy`'si bu nedenle gerçek görülmemiş-gün doğruluğu
  değil. Bkz. README'deki "Bilinen sızıntı" uyarısı.
- **Kök neden**: `cow_reid/overlap.py::assign_overlap_groups` eksik/`NaN`
  zaman damgalı kayıtları (OCR başarısızlığı) sessizce yanlış-güvenli tekil
  grup sayıyordu; `training.py::_tracklet_split` ise gerçek tarih veya
  örtüşme grubu değil, `session_id` metnini lexicographic sıralayıp
  sonuncusunu val seçiyordu (tek session varsa onu bile ortadan ikiye
  bölüyordu).
- **Düzeltme**: `assign_overlap_groups` artık doğrulanamayan zaman damgalarını
  açıkça `overlap_unverified=True` işaretliyor. Yeni `cow_reid/manifest.py`
  (`cow-reid build-manifest`) ham envanteri insan-onaylı düzeltme
  export'larıyla birleştirip örtüşme gruplarını doğru zaman damgalarıyla
  yeniden hesaplıyor — gerçek veride `vid_e2b59d770f3b`/`vid_eef4a44f4b99`
  artık aynı `overlap_group_id`'de. `training.py` artık
  `evaluation.held_out_date_split` tabanlı, örtüşme-grubu ve tarih-farkında
  bir ayrım kullanıyor; doğrulanamayan tarihli kayıtlar asla val/test'e
  girmiyor, tek fiziksel geçiş asla train ve val arasında bölünmüyor.
- **Kapsam dışı**: mevcut checkpoint bu turda yeniden eğitilmedi (yalnızca
  kod/ölçüm düzeltmesi); temiz ayrımla yeniden eğitim ayrı bir tur.

Aynı turda ayrıca düzeltildi: `tracklet_reviews.csv` (insan "bu iz bozuk"
kararları) daha önce hiçbir yerde tüketilmiyordu — artık `build_track_embeddings`,
`build_gallery`, `training._derive_tracklet_split` ve `audit.inspect_label_consistency`
bu işaretli tracklet'leri dışlıyor. Embedding/galeri uyumluluk kontrolü artık
checkpoint dosya yolu yerine içerik SHA-256'sı + ön işleme/crop/şema sürümüyle
yapılıyor (`gallery.py::identify_run`), böylece bir checkpoint dosyası yerinde
yeniden eğitilip üzerine yazıldığında eski/yeni embedding uzayları artık sessizce
uyumlu sayılmıyor. `process_new_videos` artık `--checkpoint` verilmesini backend
seçimi saymıyor (`--backend` açıkça gerekli) ve tüm tahminler `UNKNOWN`/`AMBIGUOUS`
olduğunda video durumunu `ID_ASSIGNED` yerine `ID_UNRESOLVED` yapıyor. Aynı
`overlap_group_id`'yi paylaşan yinelenen kayıtlar `process_new_videos` içinde artık
bağımsız olarak yeniden işlenmiyor (`DUPLICATE_LINKED`).

55 test geçti (49 eski + yeni ayrım/embedding-sözleşme/worker regresyonları,
bkz. `tests/test_split_integrity.py`, `tests/test_embedding_contract.py`,
`tests/test_worker.py`).

## v0.3.0 handoff denetimi ve cattle Re-ID geçişi

2 Eylül 2026 tarihinde alınan handoff salt kimlik verisi üzerinde denetlendi:

- 254 geçerli tracklet label satırı, 101 atanmış tracklet, 32 geçici kimlik
- 80 `same`, 169 `different` insan kararı
- 28 doğrudan çelişkisiz kimlik ve 69 eğitim-güvenli tracklet
- Karantinaya alınan gruplar: `COW_0009`, `COW_0019`, `COW_0025`, `COW_0028`
- 31 doğrudan cannot-link çelişkisi; hiçbir kullanıcı etiketi silinmedi

Eski ResNet-18 embedding'leri güvenli 69 tracklet üzerinde cross-session sanity
check'te Top-1 %98.55, Top-5 %100, mAP %98.79 verdi. Aynı etiketlerle eğitilip
aynı run üzerinde ölçüldüğü için bu bağımsız görülmemiş-gün doğruluğu değildir.

OpenCows2020 public checkpoint'i SHA-256 ile doğrulandı. ResNet-50 omurga + 128-D
projection katmanına 322 uyumlu tensor yüklendi, iki dataset classifier tensörü
atlandı. Bir epoch CE+reciprocal metric fine-tune, checkpoint reload, embedding
ve retrieval zinciri sentetik run'da tamamlandı. Saf regresyonlar 17/17 geçti.

## v0.2.1 kimlik birleştirme düzeltmesi

- Ayrı `COW_0022` ve `COW_0023` gruplarına atanmış bir aday çifti yeniden aynı
  olarak onaylama senaryosu regresyon testine eklendi.
- Varsayılan olarak küçük numaralı kimlik (`COW_0022`) korunur; kullanıcı
  arayüzden diğer ana kimliği de seçebilir.
- Birleştirme yalnızca ekrandaki iki tracklet'ı değil, iki kimlik grubunun bütün
  üyelerini taşır ve kararı `pair_reviews.csv` içine kaydeder.

## v0.2.0 kimlik çalışma masası

- Temel regresyonlar ile yeni kimlik/postür testleri: **12/12 geçti**.
- Streamlit headless uygulama testi dört sekmeyi de hatasız açtı.
- Etkileşimli çift onayı uçtan uca denendi: iki sentetik tracklet `COW_0001`
  olarak `labels.csv` dosyasına yazıldı ve galeride birlikte göründü.
- Kimlik prototipiyle arama, beklenen sentetik çapraz-video eşleşmesini ilk
  sıraya getirdi.
- Frame-level keypoint skorları tracklet seviyesinde toplandı, güncel Cow_ID ile
  birleştirildi ve video zamanı + tracklet başlangıcı doğru gözlem zamanını verdi.
- Eski tamamen boş `candidate_pairs.csv` davranışı regresyon testi altında.
- Crop → embedding → cluster → HTML sentetik smoke testi tamamlandı.

Kullanıcının son aday ekranlarında 0.985–0.989 benzerlikli birkaç çiftte yerel
benek geometrisi videolar arasında görsel olarak uyuştu. Bu, manuel bootstrap
akışı için olumlu kanıttır; henüz ölçülmüş bir Re-ID doğruluğu değildir.

Bu not, 1 Eylül 2026 tarihinde paylaşılan video grubuyla yapılan kısa teknik
doğrulamayı özetler. Ham videolar pakete dahil edilmemiştir.

## Envanter

- Fiziksel dosya: 22
- SHA-256 sonrası benzersiz video: 14
- Birebir kopya: 8
- Kamera üst yazısından tarih/sağım oturumu güvenle okunan benzersiz video: 12
- OCR'ın güvenle okuyamadığı gece/IR video: 2; bunlar yanlış tarih uydurmak yerine
  `unknown` bırakıldı ve manifestte elle düzeltilebilir.

## Uçtan uca kısa test

Aynı günün bir sabah ve bir öğleden sonra videosunun ilk 60 saniyesi işlendi:

- 2 video
- 19 ham tracker izi
- Hareket, yön, süre ve temiz-kare filtrelerinden geçen 10 tracklet
- Tracklet başına en fazla 12 doğal torso, maskeli torso ve tam-vücut görüntüsü
- 10 adet 512 boyutlu track embedding'i

ImageNet ön-eğitimli baseline'ın en yüksek oturumlar-arası aday benzerliği yaklaşık
`0.934` oldu. Kimliği doğrulayacak küpe/RFID/insan etiketi olmadığı için bu adaylar
doğru kabul edilmedi. Temkinli `0.95` eşik ve `0.02` marjla otomatik çok-oturumlu
eşleşme sayısı sıfırdır; aday sıraları inceleme için yine üretilmiştir.

Bu sonuç beklenen güvenli davranıştır: sistem yeterli kanıt yokken rastgele bir
`Cow_ID` vermek yerine tracklet ve aday listesi üretir. Gerçek doğruluk, ortak
ineklerin sabah/öğle geçişlerine Cow_ID etiketi bağlandıktan sonra ölçülmelidir.
