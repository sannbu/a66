# Uygulama görevi: `bg_sequence_v4.py`

Bu doküman, IR (termal) görüntü dizilerinden etiketli objeleri silip arka planı dolduran v4 scriptini **sıfırdan** yazacak kodlama ajanı içindir. Önceki sürümleri (v1, v2, v3) ve onların analizini görmeyeceksin; ihtiyacın olan her şey burada. Burada yazmayan bir tasarım kararı vermen gerekirse karar verme: Bölüm 15'teki kurala uy.

## 0. Görevin özeti ve çalışma kuralları

**Ne yazacaksın:**

| Dosya | İçerik |
|---|---|
| `bg_sequence_v4.py` | Ana script, tek dosya (kullanıcı tek dosya istiyor) |
| `v4_tools.py` | Sentetik düzenek üretici, değerlendirme, kalibrasyon, insan değerlendirme sayfası |
| `test_bg_sequence_v4.py` | pytest testleri (Bölüm 9) |
| `V4_KULLANIM.md` | Kısa kullanım kılavuzu (komutlar, çıktılar, sınırlar) |
| `V4_RAPOR.md` | Bitiş raporu (Bölüm 14'teki şablon) |

**Kurallar:**
- Kod tanımlayıcıları İngilizce. Yorumlar, docstring'ler, komut satırı yardım metinleri ve konsol çıktıları Türkçe.
- Bağımlılıklar: Python ≥ 3.9, `numpy`, `opencv-python` (contrib değil), `scipy`. Yalnız `--lama` için `torch`. Testler için `pytest`. Başka paket ekleme.
- Algoritmayı Bölüm 6'da yazdığı gibi uygula. Sabitleri Bölüm 6.1'deki adlarla, modül düzeyinde tanımla.
- Bölüm 7'deki dersler zorunlu kurallardır; her biri önceki sürümlerde ölçülmüş gerçek bir hatadan çıkmıştır.
- Bölüm 11'deki aşama sırasını izle. Her aşamanın testleri geçmeden sonrakine geçme.
- Ayar yalnız Bölüm 12'deki protokolle yapılır.
- Commit ve push yapma; kullanıcı ayrıca isterse yaparsın.

## 1. Problem ve hedef

- **Girdi:** Sıralı karelerden oluşan IR dizileri (video gibi) ve her karedeki objelerin sınır kutuları (bbox).
- **İş:** Etiketli objeleri sil, yerlerini doğal görünen arka planla doldur, her silme işlemine bir kalite kararı ver (`iyi`, `orta`, `kotu`).
- **Kısıtlar:**
  - Gerçek referans (objesiz kare, GT) yok. Değerlendirme referanssız yapılır.
  - Etiketler yalnız bbox; piksel maskesi yok. Kutu objeden büyüktür.
  - Kamera bazı dizilerde sabit, bazılarında hareketli.
  - Veri 8 bit ya da 16 bit (14 bit veri 16 bit kapsayıcıda). Bit derinliği ve delik dışındaki pikseller birebir korunur.
- **Temel fikir (korunacak):** Objenin arkasındaki arka plan başka karelerde görünür. O karelerdeki **gerçek pikseller** hizalanıp taşınır. Gerçek piksel, sıcaklık değerini ve sensör gürültüsünü korur; uydurma dolgudan değerlidir.
- **Tipik veri:** 640×512 ya da 640×480; objeler dron, kuş, insan, araç; boyut 8×5 pikselden 150×70 piksele; çoğu sıcak (parlak), bazıları soğuk; çevrelerinde ısı halesi olur; arka plan gökyüzü (bulutlu, zamanla değişir) ya da dokulu zemin.

## 2. Varsayımlar

Bunlar kullanıcı adına sabitlenmiş kararlardır. Yanlış olan varsa kullanıcı bu bölümü düzeltir; sen sorgulamadan uygula.

1. **Çıktının amacı:** Öncelik tek kare arka plan görüntüsüdür. Ardışık kareler arası tutarlılık ölçülür ve raporlanır; yalnız aşırı titremede karar `orta`ya düşer.
2. **Kamera hareketi:** Uzak sahne (gök, deniz, uzak arazi) ya da pan/tilt varsayılır. Ardışık kareler arası hareket benzerlik dönüşümüyle (öteleme + dönme + ölçek) modellenir. Yakın yapıların önünde ötelenen kamera (paralaks) kapsam dışıdır; kalite sinyalleri bunu yakalayıp `kotu` der.
3. **Etiketler:** Etiket kaydı olan her karede sahnedeki **tüm** objeler etiketlidir. Etiket kaydı olmayan kare "bilinmiyor" sayılır (Bölüm 3.4).
4. **Görünürlük payı:** Dizinin parlaklık aralığının %0,5'inden küçük hata gözle görünmez kabul edilir (`VISIBLE_FRACTION`). Testler ve kabul ölçümleri ise en katı durumda, `--visible-fraction 0` ile yapılır.
5. **Gürültü eklenmez.** Hiçbir koşulda sentetik gürültü üretilmez.
6. **Öğrenilmiş dolgu isteğe bağlıdır** (`--lama`). Hiç açılmayan piksel için kullanılır; yoksa o obje `kotu` olur.
7. **TIFF üst verisi korunmaz;** piksel değerleri ve veri tipi birebir korunur.
8. Buhar gibi yarı saydam ön plan katmanları kapsam dışıdır.

## 3. Girdi biçimleri

### 3.1 Kareler
- Klasördeki `.tif` / `.tiff` dosyaları. Sıra: dosya adı gövdesindeki **son tam sayıya** göre artan.
- **Dizilere ayırma:** Gövdenin son 6 karakteri rakamsa önek = gövde eksi o 6 karakter; değilse önek = sondaki rakamlar atılmış gövde. Aynı önekli kareler bir dizidir; diziler birbirinden bağımsız işlenir.
- **Okuma:** `np.fromfile(path, np.uint8)` + `cv2.imdecode(..., cv2.IMREAD_UNCHANGED)`. `cv2.imread` kullanma: Windows'ta ASCII dışı yollarda (ör. `Masaüstü`) çalışmaz.
- **Paketli TIFF denetimi:** Klasik TIFF başlığından `BitsPerSample` (etiket 258) okunur. 8'in katı değilse (paketli 12/14 bit) `IOError` ver: OpenCV bunları hata vermeden yanlış değerlerle okur. BigTIFF'te bu denetim atlanır.
- **Kanallar:** Tek kanal beklenir. 3 ya da 4 kanallı karede geometri ve kararlar gri düzlemde (ilk 3 kanalın float32 ortalaması) hesaplanır, dolgu ilk 3 kanala ayrı ayrı uygulanır, 4. kanal (alfa) değiştirilmez.
- **Yazma:** `cv2.imencode(uzantı, image)`; kodlanan veri geri çözülüp `dtype` ve `np.array_equal` ile doğrulanır, sonra `tofile`. Doğrulama tutmazsa `IOError`.

### 3.2 Proje düzeni (`--project KÖK`)
```
KÖK/2_Etiketlenecek_Veri/tif/*.tif|*.tiff
KÖK/3_Etiketlenmis_Veri_Ciktisi/coco-images.json
KÖK/3_Etiketlenmis_Veri_Ciktisi/coco-annotations.json
```

### 3.3 Etiket kaynakları
- **COCO (iki dosya):** `coco-images.json` → `{"images": [{"id": 7, "file_name": "gorsel/frame_000007.png", ...}]}`. Kare ile eşleşme **uzantısız dosya adıyla** yapılır (`frame_000007`); klasör ve uzantı farklı olabilir. `coco-annotations.json` → `{"annotations": [{"image_id": 7, "bbox": [x, y, w, h], ...}]}`; `bbox` piksel, sol üst köşe + genişlik/yükseklik. Kategori yok sayılır.
- **YOLO (`--labels KLASÖR`):** Kare başına `<gövde>.txt`; satır `sınıf cx cy w h`, değerler 0–1 normalize.
- **CSV (`--labels DOSYA.csv`):** Başlık `frame,x,y,w,h`; `frame` kare numarası (dosya adındaki son tam sayı), `x,y` sol üst köşe, piksel.
- Kutu görüntüye kırpılır; kırpıldıktan sonra genişlik ya da yükseklik 1 pikselden küçükse atılır.

### 3.4 Bilinen, boş ve bilinmeyen kare
- **Bilinen:** Etiket kaydı olan kare (kutu listesi boş olabilir).
- **Bilinmeyen:** COCO'da `images` listesinde olmayan kare; YOLO'da `.txt` dosyası olmayan kare. Bilinmeyen kare diziden **çıkarılır**: işlenmez, yazılmaz, kaynak olarak kullanılmaz. Sayısı `ozet.json`'a yazılır ve konsolda uyarı verilir.
- **Boş (bilinen, kutusuz):** COCO'da kayıtlı ama anotasyonsuz kare; YOLO'da boş `.txt`. Dosya bayt bayt `iyi/` klasörüne kopyalanır.
- CSV boş kareyi ifade edemez: CSV'de geçmeyen kare **boş** sayılır.
- `--missing-labels {auto,unknown,empty}`: `auto` yukarıdaki davranış; diğer ikisi kayıtsız kareyi zorla bilinmeyen ya da boş sayar.
- Hiçbir kare eşleşmezse `ValueError`.
- Zaman indisi `t`, dizinin bilinen karelerindeki sıra numarasıdır; "kare uzaklığı" bu indisle ölçülür.

## 4. Çıktılar (`--out KÖK`)

| Yol | İçerik |
|---|---|
| `iyi/`, `orta/`, `kotu/` | `bg_<özgün dosya adı>`: temizlenmiş kare, özgün biçim ve veri tipinde. Karenin klasörü en kötü objesinin kategorisidir |
| `silinmemis/` | Özgün karelerin kopyası (`--no-originals` ile kapanır) |
| `guven/<gövde>.png` | 8 bit güven haritası, yalnız objesi olan kareler için: 255 dokunulmamış piksel; delik pikseli 200 (`iyi`), 128 (`orta`), 0 (`kotu`). Dikdörtgen değil, gerçek delik maskesi |
| `onizleme/<gövde>.png` | `--preview-every N` ile: solda özgün kare (delik sınırı kategori renginde, üstünde `mod ê neden`), sağda sonuç. 8 bit; ölçek özgün gri karenin %0,5–%99,8 yüzdelikleri. Renkler BGR: iyi (0,200,0), orta (0,200,255), kotu (0,0,255) |
| `rapor.csv` | Obje başına bir satır (aşağıda) |
| `hizalama.csv` | Kare çifti başına: `sequence, frame, dx, dy, rot_deg, scale, corner_px, n_points, n_inliers, reliable` |
| `ozet.json` | `surum: "v4"`, kullanılan argümanlar ve sabitler, dizi başına `{ad, kare, bilinmeyen_kare, kamera, hareket_p95_px, sigma_t, sigma_eff, aralik}`, kategoriye göre kare ve obje sayıları, `reason` ve `mode` dağılımı, `sure_sn`, `saniye_obje` |

**`rapor.csv` sütunları (bu sırayla):**

`frame, sequence, object, label_x0, label_y0, label_x1, label_y1, region, hole_x0, hole_y0, hole_x1, hole_y1, hole_pixels, rect_pixels, mode, n_sources, gap_near, gap_far, one_sided, unrevealed, a1, a2, a2_coverage, offset, leftover, seam, texture, flicker, e_hat, category, reason, refined, chain_reliable, sigma_t, sigma_eff`

- `object`: karedeki kutu sırası (0'dan). `region`: karedeki bölge numarası; aynı bölgedeki objeler aynı kalite değerlerini taşır.
- `hole_*`: deliğin sınırlayan kutusu. `hole_pixels` = `|H|`, `rect_pixels` = `|R|`.
- `mode`: `kopya`, `poisson`, `lama`, `uzamsal` (Bölüm 6.9).
- `n_sources`: delikte kullanılan kaynak kare sayısı. `gap_near` / `gap_far`: yakın / uzak kaynakların hedefe kare uzaklığının piksel ağırlıklı ortalaması.
- `unrevealed`: delikte hiçbir kaynakta açılmayan piksel oranı.
- `refined`: kullanılan kaynaklardan en az biri yerel hizalandıysa 1. `chain_reliable`: kamera sabitse ya da hedef ile kullanılan kaynaklar arasındaki tüm kare çiftleri güvenilirse 1.
- Kalite sütunları (`a1` … `e_hat`), `category` ve `reason`: Bölüm 6.10–6.11.
- Ölçülemeyen değer boş bırakılır (NaN yazma).

## 5. Komut satırı

| Seçenek | Varsayılan | Anlam |
|---|---|---|
| `--project` | – | Bölüm 3.2 kökü |
| `--frames` | `<project>/2_Etiketlenecek_Veri/tif` | Kare klasörü |
| `--images-json`, `--annotations-json` | proje altındakiler | COCO dosyaları |
| `--labels` | – | COCO yerine YOLO klasörü ya da CSV dosyası |
| `--missing-labels` | `auto` | Bölüm 3.4 |
| `--out` | zorunlu | Çıktı kökü |
| `--camera` | `auto` | `auto`, `static`, `moving` (Bölüm 6.5) |
| `--search` | `1000` | Kaynağın hedefe en çok uzaklığı (kare) |
| `--margin` | `3` | Obje bir pikselden çıktıktan sonra beklenecek kare (ısıl iz) |
| `--box-pad` | `auto` | Kutu payı (piksel); `auto` Bölüm 6.3'teki formül |
| `--thresholds` | – | `T_IYI,T_KOTU` (`e_hat` biriminde). Kodda varsayılanı `None`: verilmişse o kullanılır; verilmemişse kalibrasyon dosyası; o da yoksa sabitler (1.4, 2.5) |
| `--calibration` | – | Kalibrasyon JSON'u (Bölüm 8.3): `KAPPA`, `T_IYI`, `T_KOTU` |
| `--visible-fraction` | `0.005` | Görünürlük payı |
| `--lama` | – | TorchScript inpainting modeli (Bölüm 6.9.4) |
| `--lama-device` | `cuda` | `cuda` yoksa kendiliğinden `cpu` |
| `--every` | `1` | Yalnız her N. kareyi işle; kaynak ve hizalama için tüm kareler kullanılır |
| `--workers` | `4` | Süreç sayısı |
| `--preview-every` | `0` | Her N karede bir önizleme |
| `--no-originals` | kapalı | `silinmemis/` yazma |

`--project` ya da (`--frames` ve bir etiket kaynağı) verilmemişse hata ver.

## 6. Algoritma

### 6.1 Terimler ve sabitler

- **Kutu dikdörtgeni `R_obj`:** etiket kutusu + pay `p`.
- **Bölge dikdörtgenleri `R`:** bir karedeki `R_obj`'lerin birleşiminin 8-bağlantılı bileşenleri. Her bileşen bir **bölge**dir.
- **Delik `H`:** gerçekten değiştirilecek pikseller; `H ⊆ R`.
- **Halka:** bölge dikdörtgeni `R`'nin dışında, `R`'den 3 ile `3 + RING_WIDTH` piksel uzaktaki pikseller (tam tanım 6.7). Hizalama, `a1`, `texture` ve `flicker` bu halkayı kullanır. Yalnız `leftover` kendi halkalarını deliğe (`H`) göre kurar.
- **Baskın kaynak:** delikte (inceltmeden önce `R`'de) en çok pikseli veren kaynak; eşitlikte hedefe en yakın olan.
- **Kaynak:** piksel taşınan başka bir kare.
- **`sigma_t`:** sensörün zamansal gürültü standart sapması. **`sigma_eff`** = `max(sigma_t, VISIBLE_FRACTION × aralık)`.
- Koordinat: `(x, y)`, piksel merkezleri tam sayı. Dikdörtgenler yarı açık `[x0, y0, x1, y1)`; alt sınırlar `floor`, üst sınırlar `ceil`.

| Sabit | Değer | Kullanım |
|---|---|---|
| `BOX_PAD_FRACTION`, `BOX_PAD_MIN`, `BOX_PAD_MAX` | 0.25, 4, 10 | Kutu payı |
| `MAX_REGION_FRACTION` | 0.20 | Bundan büyük bölge doldurulmaz |
| `MAX_SOURCES_PER_SIDE` | 6 | Geçmiş ve gelecek için ayrı ayrı |
| `MIN_GAIN_FRACTION`, `MIN_GAIN_PIXELS` | 0.04, 16 | Kaynak kabul eşiği |
| `RING_INNER`, `RING_WIDTH_MIN`, `RING_WIDTH_MAX`, `RING_MIN_PIXELS` | 3, 4, 12, 30 | Halka |
| `WINSOR` | 20 | Artıkların kırpılacağı kat (`sqrt(2)·sigma` biriminde) |
| `COPY_MAX_OFFSET`, `COPY_MAX_A1` | 0.5, 1.25 | Düz kopya koşulu |
| `REFINE_SKIP_A1`, `REFINE_MIN_GAIN` | 1.2, 0.10 | Yerel hizalama koşulu |
| `IDENTITY_SNAP` | 0.25 px | Bundan küçük dönüşüm birim sayılır |
| `STATIC_PAIR_P95`, `STATIC_LONG_GAP`, `STATIC_LONG_P95` | 0.3 px, 50 kare, 1.0 px | Sabit kamera testi |
| `FEATURE_CLEARANCE` | 12 px | Özellik noktalarının objelerden uzaklığı |
| `CORE_Z`, `GROW_Z`, `CORE_MIN_AREA`, `HOLE_DILATE`, `HOLE_FULL_FRACTION` | 4.0, 2.0, 4, 2, 0.85 | Delik inceltme |
| `UNREVEALED_MAX` | 0.10 | Açılmayan piksel oranı sınırı |
| `KAPPA`, `T_IYI`, `T_KOTU` | 0.8, 1.4, 2.5 | Tahmini hata ve eşikler |
| `LEFT_Z`, `LEFT_ORTA`, `LEFT_KOTU` | 4.0, 0.75, 1.5 | Kalıntı |
| `SEAM_MIN_PAIRS`, `SEAM_ORTA`, `SEAM_KOTU` | 6, 2.0, 3.0 | Dikiş |
| `TEX_ORTA`, `TEX_KOTU` | 1.6, 2.5 | Doku |
| `FLICKER_ORTA` | 2.5 | Titreme |
| `VISIBLE_FRACTION` | 0.005 | Görünürlük payı |
| `NOISE_PAIRS`, `NOISE_BLOCK`, `NOISE_QUANTILE` | 30, 32, 0.20 | Gürültü ölçümü |
| `CACHE_BYTES` | 300 MB | Süreç başına kare önbelleği (LRU) |

### 6.2 Dizi başına akış

```
1. kareleri ve etiketleri yükle, bilinmeyenleri çıkar            (Bölüm 3)
2. global hizalama zinciri ve kamera sınıfı                      (6.5)
3. doluluk dikdörtgenleri                                        (6.3)
4. sigma_t, aralık, sigma_eff                                    (6.4)
5. her işlenecek kare için, her bölge için:
     kaynak planı (6.6) -> yerel hizalama (6.7) -> delik inceltme (6.8)
     -> dolgu (6.9) -> kalite sinyalleri (6.10) -> karar (6.11)
6. kareyi, güven haritasını, rapor satırlarını yaz               (6.12)
```

### 6.3 Kutu payı ve doluluk

- Pay: `p = clamp(round(BOX_PAD_FRACTION × max(w, h)), BOX_PAD_MIN, BOX_PAD_MAX)`; `--box-pad N` verilirse `p = N`.
- `R_obj` = etiket kutusu her yönden `p` piksel büyütülmüş, tam sayıya yuvarlanmış, görüntüye kırpılmış.
- **Doluluk listesi `occ[t]`** (kare `t`'nin kendi koordinatında dikdörtgenler): `t` karesinin `R_obj`'leri **ve** `k = 1..margin` için `t−k` karelerinin `R_obj`'lerinin `t` koordinatına taşınmış halleri (dört köşe `M[t−k→t]` ile dönüştürülür, eksen hizalı sınırlayan dikdörtgen alınır). Bu, ısıl izi sahne koordinatında bekletir.
- Bir kaynak pikseli **geçerlidir** ⇔ görüntü içindedir ve `occ[t']` dikdörtgenlerinin hiçbirinde değildir.
- Bölge alanı kare alanının `MAX_REGION_FRACTION` katını aşıyorsa doldurma: kategori `kotu`, `reason = cok_buyuk`, pikseller değişmez.
- Etiket kutusunu farklı ölçeklerle (1.0, 1.1, … 1.4) deneyip puanlama **yoktur**. Yerini sabit pay `p` ve delik inceltme (6.8) alır.

### 6.4 Gürültü ve görünürlük ölçüsü

- `NOISE_PAIRS` kadar eşit aralıklı ardışık kare çifti `(t−1, t)` seç.
- Her çiftte `d = I_t − I_{t−1}`'in `t` koordinatına hizalanmış hali (6.5'teki `M`, en yakın piksel). Maske: iki karede de geçerli, hiçbir doluluk dikdörtgeninde değil.
- Kareyi `NOISE_BLOCK × NOISE_BLOCK` bloklara böl; en az yarısı maskeli bloklarda `s_blok = 1.4826 × MAD(d) / sqrt(2)`. Çiftin değeri blok değerlerinin `NOISE_QUANTILE` yüzdeliği (en düz bloklar: doku ve hizalama artığı gürültüye karışmasın).
- `sigma_t` = çift değerlerinin medyanı. Çift yoksa yedek: tek karede `1.4826 × median(|yatay komşu farkı|) / sqrt(2)`.
- `aralık` = örneklenen karelerde `(p99 − p1)` medyanı. `sigma_eff = max(sigma_t, VISIBLE_FRACTION × aralık)`.

### 6.5 Global hizalama zinciri ve kamera sınıfı

**Çift dönüşümü `P[t]`** (kare `t` koordinatı → kare `t−1` koordinatı, 3×3):

```
lo, hi = I_{t-1} gri karesinin p1, p99'u
to8(x) = clip((x - lo) / (hi - lo) * 255, 0, 255).astype(uint8); GaussianBlur(sigma = 1.5)
maske  = occ[t-1] dikdörtgenlerinden FEATURE_CLEARANCE ve kenardan 8 piksel uzak pikseller
pts    = cv2.goodFeaturesToTrack(prev8, maxCorners=800, qualityLevel=0.01, minDistance=8, blockSize=7, mask=maske)
ileri  = cv2.calcOpticalFlowPyrLK(prev8, cur8, pts, None, winSize=(21,21), maxLevel=4,
                                  criteria=(COUNT|EPS, 30, 0.01))
geri   = aynı çağrı cur8 -> prev8
tut    = iki durum da başarılı, |geri - pts| <= 1.0 piksel, ileri noktası occ[t]'den FEATURE_CLEARANCE uzak
en az 30 nokta varsa:
    cv2.setRNGSeed(0)
    A, inliers = cv2.estimateAffinePartial2D(ileri[tut], pts[tut], method=cv2.RANSAC,
                                             ransacReprojThreshold=1.0, maxIters=2000,
                                             confidence=0.995, refineIters=10)
    güvenilir = inlier sayısı >= max(20, 0.4 * nokta sayısı)
güvenilir değilse: P[t] = P[t-1] (o güvenilirse), yoksa birim; reliable = 0
```

- Zincir: `C[0] = I`, `C[t] = C[t−1] · P[t]` (kare `t` → kare 0). `M[a→b] = inv(C[b]) · C[a]`. float64.
- **Kamera sınıfı (`--camera auto`):**
  - `corner_px[t]` = `P[t]` altında dört görüntü köşesinin en büyük yer değiştirmesi.
  - Ek olarak eşit aralıklı 20 çift `(t, t + STATIC_LONG_GAP)` aynı yöntemle **doğrudan** hizalanır (yavaş kayma zincir gürültüsüne karışmasın).
  - Güvenilir çift oranı < 0.5 ⇒ `gozlenemiyor`: sabit gibi davran, konsolda ve `ozet.json`'da uyar.
  - Değilse `p95(corner_px) < STATIC_PAIR_P95` **ve** uzun aralıklı çiftlerin `p95`'i `< STATIC_LONG_P95` ⇒ `sabit`; aksi halde `hareketli`.
- `sabit` (ya da `--camera static`): tüm `P[t]` birim alınır. `hareketli` (ya da `--camera moving`): ölçülen `P[t]` kullanılır.
- **Piksel taşıma kuralı:** Hedef piksel koordinatları `M[t→t']` ile kaynak koordinatına çevrilir, `floor(x + 0.5)` ile **en yakın piksele** yuvarlanır ve kaynak **tam kareden** dizinlenerek okunur. Kare dışına düşen piksel geçersizdir. `cv2.remap`/`warpAffine` ile kenar doldurma kullanma. Bölge köşelerini `IDENTITY_SNAP`'ten az oynatan dönüşüm birim sayılır (birebir kopya).

### 6.6 Bölge için kaynak planı

Plan dikdörtgen bölge `R` üzerinde yapılır (delik inceltmeden önce).

```
kalan_gecmis = kalan_gelecek = R'nin tüm pikselleri
min_kazanc = max(MIN_GAIN_PIXELS, MIN_GAIN_FRACTION * |R|)
for d in 1..search:
    for (yön, t') in ((geçmiş, t - d), (gelecek, t + d)):     # eşitlikte önce geçmiş
        t' dizi dışındaysa ya da o yön tamamlandıysa atla
        gecerli = R piksellerinin M_zincir[t -> t'] altında geçerli olanları        (6.3)
        kazanc  = gecerli & kalan_yön
        kabul   = |kazanc| > 0 ve ( |kazanc| == |kalan_yön|                         # yönü tamamlıyor
                                    ya da ( |kazanc| >= min_kazanc ve seçili_yön < MAX_SOURCES_PER_SIDE - 1 ) )
        kabul ise: t' seçililere eklenir, kalan_yön -= kazanc
    iki yön de tamamlandıysa dur
```

- Geçerlilik analitik hesaplanır: nokta dönüşümü + dikdörtgen testi; görüntü çarpıtılmaz.
- Çıktı: geçmiş ve gelecek için seçili kaynak kareleri (yakından uzağa).

### 6.7 Yerel hizalama (her seçili kaynak için)

Amaç: zincirdeki birikmiş kaymayı ve arka planın kendi hareketini (sürüklenen bulut) objenin çevresinde düzeltmek. Ölçüt halka artığıdır.

- **Halka:** `RING_WIDTH = clamp(round(0.5 × min(R yüksekliği, R genişliği)), RING_WIDTH_MIN, RING_WIDTH_MAX)`; `R`'den uzaklığı `[RING_INNER, RING_INNER + RING_WIDTH]` olan, görüntü içindeki, hedef karede başka objenin `R_obj`'sinde olmayan ve kaynakta geçerli pikseller. `RING_MIN_PIXELS`'ten azsa bu kaynağın halka ölçümleri "yok" sayılır.
- **Halka artığı** (dönüşüm `M` için): `d = I_t(halka) − S_{t'}(M · halka)`, `o = median(d)`,
  `artik(M) = sqrt(mean(clip(d − o, ±WINSOR·sqrt(2)·sigma_eff)²))`, `a1_ham = artik / (sqrt(2)·sigma_t)`.
  Medyan değil **karekök-ortalama** kullan: seyrek ama büyük yapısal hatalar (kaymış bir çizgi) görülmeli. Kırpma düzeyi her yerde aynıdır: `±WINSOR·sqrt(2)·sigma_eff`.
- **Kural:**
  1. `a1_ham(M_zincir) ≤ REFINE_SKIP_A1` ise `M = M_zincir`, `refined = 0`.
  2. Değilse yerel dönüşüm ara:
     ```
     pencere = R, her yönden clamp(1.5 * max(Rw, Rh), 40, 120) piksel büyütülmüş, kareye kırpılmış
     hedef8  = I_t penceresi; kaynak8 = S_{t'}'nin M_zincir ile pencereye taşınmış hali (en yakın piksel)
              ikisi de hedef penceresinin p1-p99'u ile 8 bite, GaussianBlur(sigma = 1.5)
     pts     = goodFeaturesToTrack(hedef8, maxCorners=200, qualityLevel=0.01, minDistance=5, blockSize=7,
                                   mask = iki karenin dolulukları ve geçersiz pikselleri FEATURE_CLEARANCE uzak)
     LK (winSize=(21,21), maxLevel=3), ileri-geri hata <= 0.7 piksel
     en az 12 nokta: setRNGSeed(0); A = estimateAffinePartial2D(pts, ileri, RANSAC, 1.0)   # hedef -> ön-hizalı kaynak
     kabul: inlier >= max(10, 0.5 * nokta), |öteleme| <= 20 px, ölçek [0.95, 1.05], |dönme| <= 3 derece
     noktalar pencere koordinatından tam kare koordinatına çevrilir, A 3x3 yapılır
     M_aday = M_zincir · A
     ```
  3. `artik(M_aday) ≤ (1 − REFINE_MIN_GAIN) × artik(M_zincir)` ise `M = M_aday`, `refined = 1`; değilse `M = M_zincir`.
- Yerel hizalama sabit kamerada da çalışır (bulut kayması).
- Son `M` ile her kaynağın `R` üzerindeki geçerliliği yeniden hesaplanır. Her kaynak için `o_k` ve `a1_ham_k` saklanır.
- **Piksel ataması:** her `R` pikseli için **yakın kaynak** = o pikselde geçerli seçili kaynaklardan `|t' − t|` en küçük olan (eşitlikte geçmiş). **Uzak kaynak** = yakın kaynağın zamansal olarak karşı tarafındaki, o pikselde geçerli en yakın seçili kaynak (yoksa yok). Hiç geçerli kaynağı olmayan piksel **açılmamış**tır.

### 6.8 Delik inceltme (kutudan obje maskesine)

Kutu objeden büyüktür. Kutunun köşelerindeki, bu karede zaten görünen gerçek arka plan korunmalıdır; düz kenarlı dikdörtgen dikiş de en kolay fark edilen biçimdir.

```
B   = R üzerinde: yakın kaynağın pikseli + o kaynağın o_k'si          # geçici arka plan tahmini
D   = I_t - B                                                        # açılmamış pikselde D tanımsız
s'  = max(sqrt(2) * sigma_t, baskın kaynağın artik değeri)           # baskın = R'de en çok pikseli olan kaynak
çekirdek = |D| > CORE_Z * s', 8-bağlantılı bileşenlerden alanı >= CORE_MIN_AREA olanlar
büyüme   = |D| > GROW_Z * s' piksellerinden çekirdeğe 8-bağlantılı olanlar (histerezis)
H = çekirdek ∪ büyüme ∪ (R'nin açılmamış pikselleri)
H = 3x3 kapama; kapalı boşlukları doldur; 3x3 ile HOLE_DILATE kez genişlet; R'ye kırp
çekirdek boşsa, ya da |H| > HOLE_FULL_FRACTION * |R| ise:  H = R
```

- `R`'de hiç zamansal kaynak yoksa `H = R`.
- `R \ H` pikselleri özgün değerleriyle kalır.

### 6.9 Dolgu

Yakın/uzak atamaları `H`'ye kısıtlanır. `unrevealed` = `H`'deki açılmamış piksel oranı.

**6.9.1 Mod seçimi**
- Zamansal kaynak hiç yok ya da `unrevealed > UNREVEALED_MAX`: `--lama` varsa `mode = lama`, yoksa `mode = uzamsal`.
- Değilse, `H`'de kullanılan **tüm** kaynaklar için `|o_k| ≤ COPY_MAX_OFFSET × sigma_t` ve `a1_ham_k ≤ COPY_MAX_A1` ve `unrevealed == 0` ise `mode = kopya`.
- Değilse `mode = poisson`.

**6.9.2 `kopya`:** Her delik pikseli yakın kaynağının piksel değerini alır, hiçbir değişiklik yapılmadan (dönüşüm birimse kaynak kareyle birebir aynı).

**6.9.3 Poisson (gradyan alanında birleştirme)** — `poisson`, `lama`, `uzamsal` modlarında:
- Bilinmeyenler `H` pikselleri. 4-komşuluk. Her `p ∈ H` için: `Σ_q (x_p − x_q) = Σ_q g_pq`, `q` görüntü içindeki 4-komşular.
- `q ∉ H` ise `x_q = I_t(q)` bilinen sınır değeridir (Dirichlet) ve sağ tarafa geçer. Görüntü dışındaki komşu denklemde yer almaz (Neumann).
- **Yönlendirme gradyanı `g_pq`:**
  - `k = p`'nin yakın kaynağı. Kaynak `k` hem `p`'de hem `q`'da geçerliyse `g1 = S_k(p) − S_k(q)`.
  - `q ∈ H` ve `q`'nun yakın kaynağı `m`, hem `p`'de hem `q`'da geçerliyse `g2 = S_m(p) − S_m(q)`.
  - İkisi de varsa `g = (g1 + g2) / 2`; biri varsa o; hiçbiri yoksa `mode == lama` ise `g = L(p) − L(q)`, değilse `g = 0`. Lama yalnız `lama` modunda çalıştırılır; `poisson` modunda açılmamış piksellerin bağlantıları `g = 0` alır.
  - **İki uçta da geçerlilik şarttır.** Tek uçta geçerli kaynaktan gradyan alma.
- Matris deliğin biçimine bağlıdır: `scipy.sparse` CSC kur, `scipy.sparse.linalg.factorized` ile bir kez çözümle, tüm kanallarda yeniden kullan.
- `uzamsal`: tüm `g = 0` (harmonik dolgu). `lama`: zamansal kaynağı geçerli olan bağlantılar yukarıdaki gibi, kalanlar `L`'den.

**6.9.4 Lama arayüzü (isteğe bağlı)**
- TorchScript model: `out = model(image, mask)`; `image` float32 `[1, 3, h, w]`, 0–1; `mask` float32 `[1, 1, h, w]`, 1 = delik; `h`, `w` 8'in katı; `out` `[1, 3, h, w]`. Kanalların ortalaması alınır, `[0, 1]`'e kırpılır.
- Kırpıntı: `H`'nin sınırlayan kutusu, her yönden `max(32, en uzun kenar)` piksel büyütülmüş, kareye kırpılmış.
- Normalizasyon: kırpıntının delik dışı piksellerinin `p0.5` ve `p99.5`'i ile 0–1; çıktı aynı ölçekle geri çevrilir.
- En uzun kenar > 512 ise `INTER_AREA` ile 512'ye küçült, çıktıyı `INTER_CUBIC` ile geri büyüt. 8'in katına yansıtmalı dolguyla tamamla, sonra kırp.
- Gri düzlemde bir kez çalıştırılır; çok kanallı karede her kanalın Poisson çözümünde aynı `L` gradyanları kullanılır.
- Model süreç başına bir kez, ilk kullanımda yüklenir.

**6.9.5 Yazma:** Tam sayı tiplerinde `rint`, sonra `[alt, üst]`'e kırp: `alt/üst` = hedef kırpıntısı ile kullanılan kaynak piksellerinin en küçük/en büyük değeri (veri tipi sınırları içinde). Poisson taşması böylece engellenir.

### 6.10 Kalite sinyalleri

Hepsi **yazılacak son tam sayı pikseller** üzerinde hesaplanır. Fonksiyon saf olmalı: `assess_region(final, target, hole, sources, ...)`; testler bozulmuş `final` verebilmeli.

`e_hat`, Bölüm 8.2'deki gerçek hata `e_true` ile aynı birimde olacak biçimde tasarlandı: gürültüden başka fark yoksa ikisi de ≈ 1'dir, halkadaki hata delikteki hataya eşitse `e_hat ≈ e_true` olur. Sentetik düzenekte ikisi arasında kaba bir uyum görmüyorsan uygulamada hata vardır.

| Sinyal | Tanım | Kusursuz durumda |
|---|---|---|
| `a1` | `max_k a1_k`; `a1_k = artik_k / (sqrt(2)·sigma_eff)` (6.7'deki `artik`, son dönüşümle). `H` piksellerinin en az %10'unu veren kaynaklar ve baskın kaynak üzerinden. "Aynı kaynaklar deliğin hemen yanında ne kadar yanılıyor" | ≈ 1 (ya da < 1) |
| `a2` | Uzak kaynağı olan delik piksellerinde `d = final(p) − (S_uzak(p) + o_uzak)`; `a2 = sqrt(mean(clip(d, ±WINSOR·sqrt(2)·sigma_eff)²)) / (sqrt(2)·sigma_eff)`. `d`'den medyan **çıkarılmaz**: dolgunun tümden kayması (ders 24) burada görünmelidir. "Son dolgu, zamanın öbür yanındaki bağımsız görüntüyle ne kadar uyuşuyor" | ≈ 1 |
| `a2_coverage` | Uzak kaynağı olan delik pikseli oranı. `< 0.5` ise `a2` yok ve `one_sided = 1` | – |
| `offset` | `max_k |o_k| / sigma_t` | ≈ 0 |
| `leftover` | Aşağıda | 0 |
| `seam` | Aşağıda | ≈ 1 |
| `texture` | `hp = X − GaussianBlur(X, sigma=1.5)`; `r = MAD(hp[H]) / MAD(hp[halka])`; `texture = max(r, 1/r)`. Yalnız `kopya` ve `poisson` modlarında karara girer | ≈ 1 |
| `flicker` | Aşağıda | ≈ 1 |
| `e_hat` | `max(a1, KAPPA·a2)`; biri yoksa öbürü; ikisi de yoksa yok. "Tahmini hata, gürültü katı olarak" | ≈ 1 |

**`leftover` (deliğin dışında kalan obje/hale):** baskın kaynak `k` ile.
```
d = 1, 2, 3 için: halka_d = delikten uzaklığı d'ye yuvarlanan, kaynakta geçerli, başka objede olmayan pikseller
   v_d  = median(I_t - S_k - o_k) halka_d üzerinde;   n_d = piksel sayısı
uzak = d = 4 .. 3 + RING_WIDTH için aynı v değerleri;  med_uzak = median(uzak);  yay_uzak = 1.4826 * MAD(uzak)
işaret = D'nin çekirdek üzerindeki toplamının işareti (6.8); çekirdek yoksa mutlak değer kullan
fazla_d = işaret * (v_d - med_uzak)
se_d    = 1.2533 * sqrt(2) * sigma_t / sqrt(n_d)
anlamlı ⇔ n_d >= 6 ve fazla_d > max(LEFT_Z * se_d, 3 * yay_uzak)
leftover = anlamlı olanların max(fazla_d) / sigma_eff;  hiçbiri anlamlı değilse 0;  kaynak yoksa "yok"
```
`leftover > LEFT_ORTA` ve `H ≠ R` ise **bir kez** yeniden dene: `H`'yi 2 piksel genişlet (`R`'ye kırp), dolguyu ve sinyalleri yeniden hesapla; `leftover` küçüldüyse yeni sonucu kullan.

**`seam` (kenar başına en kötü basamak):**
```
her (p ∈ H, q ∉ H, q p'nin 4-komşusu, q2 = q + (q - p) görüntü içinde, q ve q2 başka objede değil) için:
    iç = |final(p) - I_t(q)|;   dış = |I_t(q) - I_t(q2)|;   kenar = (q - p) yönü (üst, alt, sol, sağ)
kenar başına en az SEAM_MIN_PAIRS çift varsa:  oran = mean(iç) / max(mean(dış), 1.128 * sigma_t)
seam = oranların en büyüğü
```
Çevrenin ortalamasını **alma**: tek kenardaki güçlü dikiş diğer üç kenarla seyrelmemeli.

**`flicker`:** `--every 1` iken, bir önceki karenin sonucu `Out_{t−1}` ile.
```
bölge = H_t ∩ (H_{t-1}'in t koordinatına taşınmışı), en az 30 piksel
iç  = sqrt(mean(clip(Out_t - Out_{t-1}'in taşınmışı)²)) bölgede          # kırpma: ±WINSOR·sqrt(2)·sigma_eff
dış = sqrt(mean(clip(I_t - I_{t-1}'in taşınmışı)²)) halkada, iki deliğin de dışında
flicker = iç / dış
```
Önceki karede obje yoksa ya da ortak bölge 30 pikselden küçükse `flicker` yok.
Her iş parçasının ilk karesi için bir önceki kare de (yazılmadan) işlenir ki sonuç süreç sayısından bağımsız olsun. `--every > 1` iken `flicker` yok.

### 6.11 Karar kuralları

İlk tutan kural kategoriyi ve `reason` değerini belirler.

| Sıra | Koşul | Kategori | `reason` |
|---|---|---|---|
| 1 | Bölge çok büyük (6.3) | kotu | `cok_buyuk` |
| 2 | `mode == uzamsal` | kotu | `kaynak_yok` |
| 3 | `leftover > LEFT_KOTU` | kotu | `kalinti` |
| 4 | `seam > SEAM_KOTU` | kotu | `dikis` |
| 5 | mod `kopya`/`poisson` ve `texture > TEX_KOTU` | kotu | `doku` |
| 6 | `e_hat > T_KOTU` | kotu | `hata` |
| 7 | `mode == lama` | orta | `ogrenilmis` |
| 8 | `one_sided == 1` | orta | `tek_yon` |
| 9 | `a1` yok (halka yetersiz) | orta | `halka_yok` |
| 10 | `leftover > LEFT_ORTA` | orta | `kalinti` |
| 11 | `seam > SEAM_ORTA` | orta | `dikis` |
| 12 | mod `kopya`/`poisson` ve `texture > TEX_ORTA` | orta | `doku` |
| 13 | `flicker > FLICKER_ORTA` | orta | `titreme` |
| 14 | `e_hat > T_IYI` | orta | `hata` |
| 15 | aksi halde | iyi | `tamam` |

- Karşılaştırmalar tam yazıldığı gibi: `e_hat == T_IYI` → `iyi`.
- Sınırlamalar kategoriyi **doğrudan** belirler; sayısal bir tavanla eşik karşılaştırması yapma.

### 6.12 Kare düzeyi

- Karenin kategorisi = objelerinin en kötüsü. Objesi olmayan bilinen kare `iyi`'ye bayt bayt kopyalanır.
- Güven haritası Bölüm 4'teki gibi, delik maskeleriyle.
- Deliklerin dışındaki her piksel girdiyle birebir aynı olmalıdır.

### 6.13 Paralellik ve belirlenimcilik

- `ProcessPoolExecutor`. Hizalama çiftleri paralel hesaplanabilir. Kareler **ardışık** parçalara bölünür.
- Rastgelelik yok. Tek rastgele kaynak OpenCV RANSAC'ıdır; her çağrıdan önce `cv2.setRNGSeed(0)`.
- Aynı girdi, `--workers` ve parça sayısından bağımsız olarak **bayt bayt aynı** çıktıyı ve aynı `rapor.csv`'yi üretmelidir.
- Fonksiyonlara durum, nesne özniteliği üzerinden gizlice taşınmaz; parametre olarak geçilir.
- Kare önbelleği süreç başına LRU, `CACHE_BYTES` ile sınırlı.
- Sonucu bayt düzeyinde değiştirmeyen hız iyileştirmeleri serbesttir.

## 7. Önceki sürümlerden dersler (zorunlu kurallar)

Her madde ölçülmüş bir hatadır. Parantez içi, eski sürümde ne olduğudur.

**Puanlama**
1. **Hatayı hizalanmamış kare farkına bölme.** (Eskiden "gürültü tabanı" komşu kareyle hizalamasız farktı; kamera hızlıyken taban büyüdü, bozuk dolgu "iyi" çıktı: hızlı salınım testinde 9 "iyi"nin 8'i bozuktu.) Yalnız `sigma_t` / `sigma_eff` kullan.
2. **Karar, objenin kendi yerindeki kanıta ve son piksellere dayanmalı.** (Eskiden hata terimi başka yerlerdeki sınama bölgelerinden geliyordu; şeritlerle bozuk bir dolgu 0,95 puan aldı.) `a1` deliğin hemen yanında, `a2` deliğin içinde ve son pikselde ölçülür.
3. **Yöntemler arasında piksel ortalama hatasıyla seçim yapma.** (7 aday, en çok 4 sınama bölgesinin medyanıyla seçiliyordu. Piksel ortalaması bulanık dolguyu kayırdı: uzamsal dolgu yapılı sahnelerin %52'sinde bulanık leke bıraktı. Kazananın hatası aynı zamanda puan olduğu için tahmin iyimserdi: 1,1–1,5 tahmin, 1,9–2,2 gerçek.) v4'te tek zamansal yöntem vardır; seçim yoktur.
4. **Tavan değeri eşikle çakıştırma.** (Bir dolgu türünün puanı 0,8 ile sınırlıydı, "iyi" eşiği de ≥ 0,8 idi; "en fazla orta" denen dolgu "iyi" çıktı.) Sınırlamalar kategoriyi doğrudan belirler (6.11).
5. **Üretimin kendiliğinden tutturduğu ölçüte güvenme.** (Poisson birleştirme dikiş ölçütünü, eklenen gürültü doku ölçütünü kendiliğinden tutturuyordu.) Bu yüzden dikiş kenar başına ölçülür ve asıl karar `a1`/`a2`'dedir.
6. **Çevre ortalaması dikişi seyreltir.** (Tek kenarda 4 katlık basamak ortalamada 1,75 yaptı ve geçti.)
7. **Seyrek büyük hataları kaçıran dayanıklı istatistik kullanma.** (Medyan tabanlı ölçü "düz" derken standart sapma 75 idi.) Artıklar karekök-ortalama ile ölçülür.
8. **Kalıntı denetimi maskeyi üreten mantıktan bağımsız olmalı.** (Hale ölçümü ile kalıntı denetimi aynı "dört kenar aynı işaretli" testini kullanıyordu; yapılı arka planda ikisi birden kör oldu. Maskenin dışında gürültünün 5 katı hale kaldı ve fark edilmedi.) v4'te maske tek kare farkından, kalıntı ise halka medyanlarının zamansal kaynağa göre fazlasından ölçülür.
9. **Ölçütler son piksellerde hesaplanır.** (Bir düzeltme adımı doku ölçüldükten sonra uygulanıyordu; çıktısı hiç denetlenmedi.)
10. **Kırpılıp en küçüğü alınan terimlerden puan yapma.** (Çok sayıda tam 0 ve tam 1 üretti; sıralama bilgisi kayboldu.) `e_hat` süreklidir ve fiziksel birimdedir.
11. **Ayar ve raporu aynı veride yapma.** (Eşikler tek bir sentetik dizide ayarlanıp aynı dizide raporlandı.) Bölüm 8.3 ve 12.
12. **Koşullu oranları iki yönde, açık yaz.** (Eski belgede "gerçekte kötülerin %74'ü kotu çıktı" yazıldı; doğrusu %67 idi. %74, "kotu denenlerin gerçekten kötü olma oranı"ydı.) Bölüm 8.2.
13. **Kalıntı objenin işaretini taşır.** (İşaretsiz ölçüm, bulut içindeki karanlık bir boşluğu sıcak objenin halesi sandı ve iyi dolguyu `kotu` yaptı.) `leftover` işaretlidir.
14. **Görünürlük payını yerel kontrasta bağlama.** (Eskiden yerel parlaklık aralığının %1,5'iydi; aynı hata bir yerde geçip başka yerde kalıyordu.) Pay dizi düzeyindeki aralıktan hesaplanır ve tek bir bildirilmiş varsayımdır.
15. **Zamansal tutarlılığı ölç.** (Eskiden ne üretimde ne puanda vardı.) `flicker`.
16. **Tek kötü obje yüzünden kareyi kullanılmaz bırakma.** Karenin klasörü en kötü objeye göredir, ama güven haritası obje başına kararı piksel düzeyinde taşır.

**Hizalama ve piksel taşıma**
17. **Tüm doluluk ve kaynak mantığı hizalanmış koordinatta çalışır.** (Eskiden ham piksel koordinatındaydı; kamera hareketinde kaynak yanlış yerden kopyalandı.)
18. **Tek öteleme vektörü yetmez.** (±1° dönmede hata 2,3 kata çıktı.) En az benzerlik dönüşümü.
19. **Kenarı yansıtarak doldurma; kırpıntıdan örnekleme.** (Kayma kırpıntıdan büyük olunca yansıtılan içerik geçerli sayıldı: dikey şeritli dolgu.) Tam kareden oku, dışarısı geçersiz.
20. **Hareket tahmininde obje yakınını kullanma.** (Yoğun akış hareketli objenin hareketini çevresine yaydı: gerçek kayma 0,3 pikselken 4–9 piksel ölçüldü.) `FEATURE_CLEARANCE`.
21. **Piksel taşırken en yakın pikseli kullan, çift doğrusal enterpolasyon kullanma.** (Çift doğrusal örnekleme gürültüyü yumuşattı; yama çevresinden pürüzsüz göründü.)
22. **Hizalamayı yalnız işe yarıyorsa uygula.** Ölçüt halka artığıdır (6.7).

**Maske ve dolgu**
23. **Kutunun tamamını silme.** (Dikdörtgen maske kutu köşelerindeki gerçek arka planı attı, düz kenarlı dikiş bıraktı. Doluluk kutusu etiketin 1,3 katı + 6 pikseldi; yavaş objede arka planın açılması gereksiz uzadı.) 6.3 ve 6.8.
24. **Poisson sınırı temiz olmalı.** (Maskenin dışında kalan hale sınır koşuluna girdi, tüm dolguyu +12…+32 birim kaydırdı.) Delik genişletmesi ve kalıntı yeniden denemesi bunun içindir.
25. **Gradyan için kaynağın iki uçta da geçerli olması şart.** (Tek uçta geçerli kaynaktan alınan gradyan çift kutuplu leke bıraktı.)
26. **Sentetik gürültü ekleme.** (Eklenen gürültünün tohumu yalnız maske boyutuna bağlıydı: her karede birebir aynı desen. Ayrıca beyaz gürültü gerçek IR gürültüsüne benzemez.)
27. **Kaynak piksellerini ortalama.** (İki karenin ağırlıklı ortalaması gürültüyü azalttı; telafi için sentetik gürültü eklemek gerekti.) v4'te her piksel tek bir kaynaktan gelir.
28. **Hızlı yürüyüş (Telea) inpainting kullanma.** (Büyük alanda X biçimli iz bıraktı.) Uzamsal yedek harmonik dolgudur ve her zaman `kotu`dur.
29. **Yöntemi kareden kareye değiştirme.** (Aynı obje için 30 karede 4 farklı yöntem seçildi.) Tek yöntem.
30. **Poisson çıktısını kaynak aralığına kırp** (taşma).

**Veri ve mühendislik**
31. **Etiketsiz kareyi temiz sayma.** (Etiketi olmayan kare doğrudan "iyi"ye kopyalanıyor ve serbest kaynak sayılıyordu.) Bölüm 3.4.
32. **Paketli 14 bit TIFF'i reddet; `cv2.imread` kullanma; yazdığını doğrula.** Bölüm 3.1.
33. **Tek kod tabanı.** (Üç sürüm, birbirinden yamayla üretilmiş üç ayrı dosyaydı.) Seçenekler bayraktır.
34. **Testler ve ölçüm düzeneği teslimatın parçasıdır.** (Eski testler ve düzenek silindi; sonuçlar yeniden üretilemedi.)
35. **Öğrenilmiş bileşenleri piksel hatasıyla kabul etme.** (Sahneye özel küçük bir düzeltici ağ, kabul ölçütü piksel hatası olduğu için gürültüyü silerek "iyileşti"; gerçek kazanç 1,56 → 1,54, süre 3 sn → 246 sn.) v4'te böyle bir bileşen yok.
36. **Lama çıktısı gürültüsüzdür ve doğrulanamaz;** en fazla `orta` olur. Eldeki model yalnız sokak ve yaya sahneleriyle eğitildi; gökyüzünde güvenme.

## 8. Araçlar (`v4_tools.py`)

### 8.1 `sentez`: gerçek arka planı bilinen sentetik dizi

```
python v4_tools.py sentez --out KLASÖR --scene {gok,zemin,karma} --camera {sabit,salinim,pan,donme,zoom,hizli}
                          --frames 600 --seed 1 [--bulut-evrim 520]
```

Çıktı: Bölüm 3.2 düzeninde proje (`frame_%06d.tiff`, uint16, 640×512, COCO dosyaları), `gt/frame_%06d.tiff` ve `truth.json`.

- **Tuval:** kareden her yönden 140 piksel büyük. `fbm(biçim, tohum, taban=(gh, gw), oktav, kazanç)` = oktav `o` için `(gh·2^o, gw·2^o)` boyutlu standart normal ızgaranın bikübik büyütülmüşlerinin `kazanç^o` ağırlıklı ortalaması.
- **Gökyüzü:** taban `5300 + 700·y/yükseklik`. `f = cos(φ)·A + sin(φ)·B`, `φ = t / bulut_evrim`; `A`, `B` = `fbm(taban=(4,6), oktav=6, kazanç=0.55)`, tohumlar `seed+1`, `seed+2`. `detay = cos(1.7φ)·D1 + sin(1.7φ)·D2`, `D` = `fbm(taban=(12,18), oktav=4, kazanç=0.55)`. Alanlar rüzgârla `(0.16, 0.035)·t` piksel kaydırılır. `örtü = smoothstep(clip((f − 0.02) / 0.42, 0, 1))`. Gök = taban + `örtü·(650 + 260·detay)`.
- **Zemin (durağan):** `fbm(taban=(8,10), oktav=7, kazanç=0.6)` %1–%99 yüzdelikleriyle `6600 … 9200` aralığına; üstüne 14 rastgele dikdörtgen (kenar 20–120 piksel, ±150…500 birim) ve 8 çizgi (kalınlık 2–4 piksel, ±400 birim).
- **Sahne:** `gok` tümü gök; `zemin` tümü zemin; `karma` tuvalde `y < 140 + 330` gök, altı zemin.
- **Kamera** (tuval → kare, kare merkezi çevresinde, `INTER_LINEAR`): `sabit` yok; `salinim` `dx = 12·sin(2πt/60)`, `dy = 0.3·dx`; `pan` `dx = 0.27·t`; `donme` `θ = 1°·sin(2πt/80)`; `zoom` ölçek `1 + 0.02·sin(2πt/100)`; `hizli` `dx = 38·sin(2πt/40)`, `dy = 0.3·dx`.
- **Objeler** (sahne koordinatında hareket eder, merkezleri kamera dönüşümüyle kareye taşınır, en yakın piksele yerleştirilir): katman = `genlik · max(çekirdek, 0.35 · GaussianBlur(çekirdek, sigma=2.5))`.

  | Ad | Boyut (g×y) | Genlik | Hız (piksel/kare) | Yer |
  |---|---|---|---|---|
  | `hizli_kus` | 8×5, elips | 800 | 3,2; yatay gidip gelir, `y = 110 + 20·sin(t/25)` | gök |
  | `orta_drone` | 18×10 | 2500 | 1,0; gidip gelir | gök |
  | `yavas_drone` | 22×12 | 2500 | 0,23 | gök |
  | `cok_yavas_drone` | 24×13 | 2500 | 0,10 | gök |
  | `sabit_drone` | 20×12 | 2500 | 0 | gök |
  | `yavas_insan` | 12×30, dikey elips + baş | 1500 | 0,30 | zemin |

  Dron çekirdeği: gövde elipsi (değer 1) + dört köşede rotor elipsi (değer 0,8). `gok` sahnesinde insan yok; `zemin` sahnesinde tüm objeler zeminde.
- **Sensör:** kare başına zamansal gürültü `N(0, 5.5)`; sabit desen = sütun başına `N(0, 2.0)` + piksel başına `N(0, 1.0)`, **sensör koordinatında** ve dizi boyunca aynı (kamera dönüşümünden sonra eklenir).
- `girdi = round(clip(arka plan + objeler + gürültü + sabit desen))`; `gt = round(clip(arka plan + gürültü + sabit desen))` (**aynı** gürültü gerçekleşmesi).
- **Etiket:** çekirdeğin (> 0,5) sıkı kutusu; her kenar boyutun `U(0, 0.15)` katı kadar bollaştırılır, sonra `{−1, 0, 1}` tam sayı oynatma eklenir.
- `truth.json`: kare başına obje listesi `{name, core_box, label_box}`, ayrıca `sigma = 5.5`.
- Üretici fonksiyon, kamera hareketini parametre olarak da alabilmelidir (`dx(t)`, `dy(t)`, `θ(t)`, `ölçek(t)`) ve küçük boyut/kısa dizi üretebilmelidir; testler bunu kullanır.

### 8.2 `degerlendir`

```
python v4_tools.py degerlendir --bench SENTETİK_KLASÖR --run ÇIKTI_KLASÖRÜ
```

- **Gerçek hata:** `E` = (sonuç ≠ girdi olan pikseller) ∪ (`|girdi − gt| > 0.5` olan pikseller), objenin etiket kutusunun 12 piksel büyütülmüşü içinde. `e_true = mean|sonuç − gt|` (`E` üzerinde) `/ (1.128 × sigma_eff_gercek)`; `sigma_eff_gercek = max(5.5, vf × aralık)`, `vf` koşunun `ozet.json`'undaki görünürlük payı, `aralık` örneklenen `gt` karelerinde `(p99 − p1)` medyanı. `vf = 0` iken birim: 1,0 ≈ başka kareden kusursuz kopya.
- **Sınıflar:** gerçek iyi `e_true ≤ 1.6`; gerçek kötü `e_true > 3.0`; arası sınırda.
- **Rapor:** obje adına göre `e_true` medyanı ve %90'ı; 3×3 karışıklık tablosu (karar × gerçek); iki yönde koşullu oranlar, **açık adlarla**: `P(gerçek kötü | karar iyi)`, `P(gerçek iyi değil | karar iyi)`, `P(karar kotu | gerçek kötü)`, `P(karar iyi | gerçek kötü)`, `P(karar iyi | gerçek iyi)`; `e_hat` ile `e_true` arasında Spearman sıra korelasyonu; `reason` ve `mode` dağılımı. Sonuç ayrıca `degerlendirme.json`'a yazılır.

### 8.3 `kalibre`

```
python v4_tools.py kalibre --pairs BENCH1:RUN1 BENCH2:RUN2 ... --out kalibrasyon.json
```

- Izgara: `KAPPA ∈ {0.5, 0.6, …, 1.0}`, `T_IYI ∈ [1.1, 2.0]` adım 0.05, `T_KOTU ∈ [T_IYI + 0.3, 4.0]` adım 0.1. Kararlar `rapor.csv`'deki sinyallerden yeniden hesaplanır (6.11).
- Kısıt: `P(gerçek kötü | karar iyi) ≤ 0.03` ve `P(karar kotu | gerçek kötü) ≥ 0.75`. Amaç: kısıtı sağlayanlar içinde `P(karar iyi | gerçek iyi)` en büyük. Kısıt sağlanamıyorsa ikinci kısıt 0.05'lik adımlarla gevşetilir ve bu rapora yazılır.
- Çıktı: `{"kappa", "t_iyi", "t_kotu", "olusturma": {çiftler, tarih, kısıtlar, ulaşılan oranlar}}`.
- **Kalibrasyon tohumları 1 ve 2'dir; rapor ve kabul ölçütleri tohum 3 ve 4'te ölçülür.**

### 8.4 `sayfa` ve `uyum` (insan değerlendirmesi)

- `sayfa --run ÇIKTI --frames KARE_KLASÖRÜ --count 150 --out KLASÖR`: `(category × mode)` hücrelerinden eşit sayıda, sabit sırayla (rastgelelik yok: `rapor.csv` sırasında eşit aralıklı) obje seçer. Her biri için `NNN.png`: solda özgün, sağda sonuç; bölge ortada, çevresinde 30 piksel bağlam, 4 kat büyütülmüş (en yakın piksel), **kategori gösterilmeden**. `anahtar.csv` (`id, frame, object, category, e_hat, mode`) ve doldurulacak `puanlar.csv` (`id, insan`).
- `uyum --sheet KLASÖR`: doldurulmuş `puanlar.csv` ile 3×3 tablo, Bölüm 8.2'deki iki yönlü koşullu oranlar ve Cohen kappa.

## 9. Testler (`test_bg_sequence_v4.py`)

Testler küçük sentetik dizileri kendi üretir (gerekirse `v4_tools.sentez` fonksiyonlarını küçük boyutla çağırır); dış veri gerektirmez. Aksi yazmıyorsa `--visible-fraction 0` ile koşar.

1. **Giriş/çıkış:** 8 ve 16 bit gidiş-dönüş birebir; paketli (14 bit) TIFF `IOError`; ASCII dışı yol (`ğüşıöç`) çalışır.
2. **Dokunulmazlık:** delik dışındaki her piksel girdiyle aynı; güven haritası tam o piksellerde 255.
3. **Etiket kaynakları:** aynı kutular COCO, YOLO ve CSV ile verildiğinde çıktılar bayt bayt aynı.
4. **Bilinmeyen/boş kare:** bilinmeyen kare yazılmaz ve hiçbir satırda kaynak olmaz; boş kare `iyi/`'de bayt bayt aynı.
5. **Sabit sahne, hızlı obje:** `mode == kopya`; delik pikselleri kaynak karenin pikselleriyle birebir aynı; `a1` ve `a2` 0,85–1,15 arasında; karar `iyi`.
6. **Belirlenimcilik:** `--workers 1` ile `--workers 4` ve iki ardışık koşu bayt bayt aynı çıktı ve aynı `rapor.csv`.
7. **`--every 3`:** işlenen karelerin çıktısı tam koşudakiyle aynı (`flicker` hariç).
8. **Kenar:** görüntü kenarına taşan kutu hata vermez, kurallar korunur.
9. **Kıpırdamayan obje:** lama yokken `uzamsal` / `kotu` / `kaynak_yok`; sahte bir TorchScript modelle (girdiyi geri veren küçük bir modül) `lama` / `orta`.
10. **Çok kanal:** 3 kanalda kanallar tutarlı; RGBA'da alfa değişmez.
11. **Kamera hareketi:** kare başına 2 piksel tam sayı pan'da hızlı obje için `e_true ≤ 1.3`; ±1° dönmede `e_true > 3` olan hiçbir obje `iyi` değil.
12. **Karar sınırları:** `e_hat == T_IYI` → `iyi`; `mode == lama` ya da `one_sided` hiçbir zaman `iyi` değil.
13. **Kusur ekleme (kalite sinyallerinin duyarlılığı),** `assess_region` üzerinden, dokulu sabit sahnede kabul edilmiş bir dolguyla:

    | Eklenen kusur | Beklenen |
    |---|---|
    | Kaynak hizasını 1, 2, 3 piksel kaydır | `a1` kesin artan; 2 pikselden itibaren karar `iyi` değil |
    | Son dolguyu yatay aynala | `a2` yükselir, karar `iyi` değil |
    | Son dolguyu `GaussianBlur(sigma=1.5)` ile yumuşat | `texture` yükselir, karar `iyi` değil |
    | Son dolguyu harmonik dolguyla değiştir | karar `iyi` değil |
    | `kopya` modunda dolguya `3·sigma_t` sabit ekle | `seam` yükselir, karar `iyi` değil |
    | Hedef karede deliğin 1–3 piksel dışındaki piksellere `0.5, 1, 2, 4 × sigma_t` ekle (dışarıda kalmış hale) | `leftover` azalmayan; `2·sigma_t`'den itibaren karar `iyi` değil |
    | Kaynak içeriğini başka bir yerin pikselleriyle değiştir | `a1` ya da `a2` yükselir, karar `iyi` değil |

## 10. Kabul ölçütleri

Hepsi **tohum 3 ve 4**'te, 600 karelik dizilerde, `--visible-fraction 0` ile ölçülür. Kalibrasyon (yapılırsa) tohum 1 ve 2'de. Bir oranın paydası 20'den küçükse o ölçüt "ölçülemedi" yazılır, geçti sayılmaz.

**Koşullar:** A = `zemin/sabit`; B = `zemin/{salinim, pan, donme, zoom, hizli}`; C = `gok/sabit` ve `karma/sabit`; D = `gok/salinim` (yalnız raporlanır).

**Zorunlu (kararın dürüstlüğü) — A, B'nin her biri ve C için ayrı ayrı:**

| Ölçüt | Sınır | Eski sürümde ölçülen |
|---|---|---|
| Q1 `P(gerçek kötü | karar iyi)` | ≤ 0,03 | 0,06 (bulutlu sahne); hızlı kamerada 9'da 8 |
| Q2 `P(gerçek iyi değil | karar iyi)` | ≤ 0,15 | 0,29 |
| Q3 `P(karar kotu | gerçek kötü)` | ≥ 0,75 | 0,67 |
| Q4 `P(karar iyi | gerçek kötü)` | ≤ 0,07 | 0,14 |
| Q5 Spearman(`e_hat`, `e_true`), C'de | ≥ 0,60 | 0,44 (mutlak değer) |
| Q6 `sabit_drone` (lama yok) | %100 `kotu` | – |

**Hedef (dolgu kalitesi):**

| Ölçüt | Sınır |
|---|---|
| F1 A'da `hizli_kus`, `orta_drone`: `e_true` medyanı | ≤ 1,15; en az %95'i `iyi` |
| F2 A'da `yavas_drone`, `yavas_insan`: `e_true` medyanı | ≤ 1,2; en az %50'si `iyi` |
| F3 B'de (`salinim`, `pan`, `donme`, `zoom`) `hizli_kus` + `orta_drone`: `e_true` medyanı | ≤ 1,8 |
| F4 B'de `hizli`: aynı objeler | ≤ 2,2 |
| F5 C'de obje başına `e_true` medyanı | yalnız raporlanır (eski sürüm: tümü ≈ 1,5; yavaş dronlar 3–4) |
| Hız | ≤ 1,0 saniye / obje / süreç (640×512) |

**Mühendislik (zorunlu):** Bölüm 9'daki tüm testler geçer.

- Eski sürümün sayıları yaklaşık kıyastır: orada gerçek hata tanımı biraz farklıydı.
- Değişen bulut önündeki çok yavaş obje bu yaklaşımın bilinen sınırıdır. Orada dolgunun iyi olması beklenmez; kararın bunu **doğru söylemesi** beklenir.

## 11. Aşamalar

1. **İskelet:** okuma/yazma, etiket yükleyiciler, dizi ayırma, bilinmeyen/boş kare, çıktı klasörleri. Testler 1, 3, 4.
2. **Sentetik düzenek ve değerlendirme:** `sentez`, `degerlendir`. Önce bunlar: sonraki her aşama bunlarla ölçülür.
3. **Sabit kamera çekirdeği:** doluluk, kaynak planı (dönüşümler birim), delik = `R`, `kopya` ve Poisson, yazma. Testler 2, 5, 8, 10.
4. **Kalite sinyalleri ve karar:** 6.10, 6.11, `rapor.csv`, güven haritası, önizleme. Testler 12, 13.
5. **Delik inceltme:** 6.8 ve kalıntı yeniden denemesi.
6. **Hizalama:** 6.4, 6.5, 6.7; `hizalama.csv`. Test 11.
7. **Yedekler:** `uzamsal`, lama arayüzü. Test 9.
8. **Paralellik, `--every`, `flicker`.** Testler 6, 7.
9. **Ölçüm:** tohum 1–2'de koş, gerekirse `kalibre`; tohum 3–4'te kabul ölçütleri.
10. **`sayfa`, `uyum`, `V4_KULLANIM.md`, `V4_RAPOR.md`.**

Her aşamanın sonunda A koşulunda (`zemin/sabit`, tohum 1) kısa bir koşu yap ve `degerlendir` çıktısına bak.

## 12. Ayar protokolü

- **Kalibrasyonla değişebilenler:** `KAPPA`, `T_IYI`, `T_KOTU` — yalnız `kalibre` aracıyla, yalnız tohum 1–2 verisinde.
- **Elle değişebilenler** (yalnız tohum 1–2'de bir zorunlu ölçüt ya da test tutmuyorsa; her değişiklik eski değer, yeni değer, neden ve önce/sonra ölçümle rapora yazılır): `LEFT_ORTA`, `LEFT_KOTU`, `SEAM_ORTA`, `SEAM_KOTU`, `TEX_ORTA`, `TEX_KOTU`, `FLICKER_ORTA`, `CORE_Z`, `GROW_Z`, `HOLE_DILATE`, `REFINE_SKIP_A1`, `REFINE_MIN_GAIN`, `COPY_MAX_OFFSET`, `COPY_MAX_A1`, `FEATURE_CLEARANCE` ve 6.5/6.7'deki OpenCV çağrı parametreleri.
- **Değişmeyenler:** algoritmanın adımları, sinyallerin tanımı, karar kurallarının sırası, Bölüm 7'deki kurallar, `VISIBLE_FRACTION` (kullanıcı kararı).
- Tohum 3–4 sonuçlarına bakarak **hiçbir şey** ayarlanmaz. Bakıp ayarladıysan bunu rapora yaz ve yeni tohumlarla (5, 6) yeniden ölç.

## 13. Kapsam dışı (yapma)

- Sentetik gürültü, kaynak ortalaması, yöntem seçimi için sınama bölgeleri, sahneye özel öğrenen düzeltici.
- Obje maskesine dayalı kaynak geçerliliği (v4'te geçerlilik kutu dikdörtgeniyledir).
- Sabit desen gürültüsünün ayrıştırılması, paralaks/yerel akış, derin video inpainting, dedektörle denetim.
- Aynı pikselin art arda karelerde aynı kaynaktan gelmesinden doğan "donuk" gürültünün giderilmesi (`flicker` < 0,3 olarak yalnız raporlanır).
- TIFF üst verisinin korunması.

Bunlar bilinen sınırlar olarak `V4_KULLANIM.md`'ye yazılır.

## 14. Bitiş raporu şablonu (`V4_RAPOR.md`)

1. **Durum:** tamamlanan aşamalar; geçen/kalan test sayısı.
2. **Kabul tablosu:** Q1–Q6 ve F1–F5 için koşul başına ölçülen değer, sınır, geçti/kaldı. Oranların pay ve paydası sayı olarak da yazılır.
3. **Karışıklık tabloları** (A, B'nin her modu, C, D) ve Bölüm 8.2'deki koşullu oranlar, açık adlarıyla.
4. **Obje başına `e_true`** medyan ve %90, koşul başına.
5. **Kalibrasyon:** kullanıldıysa değerler ve hangi veride yapıldığı.
6. **Değiştirilen sabitler** (Bölüm 12 biçiminde).
7. **Hız:** saniye/obje, donanım.
8. **Sapmalar ve açık sorular:** dokümandan ayrıldığın ya da karar veremediğin her nokta.
9. **Doğrulanmayanlar:** çalıştırmadığın testler, gerçek veride denenmeyenler.

Başarısız ölçütü gizleme; ölçtüğün sayıyı yaz.

## 15. Emin olamadığın durumda

- Bu dokümanda karşılığı olmayan bir tasarım kararı gerekirse: en basit, en tutucu seçeneği (kararı `orta`/`kotu` yönüne çeken) uygula, kodda `# KARAR:` yorumuyla işaretle ve raporun 8. maddesine yaz.
- Bir kural diğeriyle çelişiyor görünüyorsa Bölüm 7 önceliklidir.
- Bir zorunlu ölçüt, izinli ayardan sonra da tutmuyorsa algoritmayı değiştirme: dur, sinyal başına tanı (hangi `reason`, hangi obje sınıfı, hangi koşul) ile raporla.
- Gerçek kullanıcı verisi ya da lama modeli elinde yoksa bunları isteme; sentetik düzenekle tamamla ve "gerçek veride denenmedi" diye yaz.
