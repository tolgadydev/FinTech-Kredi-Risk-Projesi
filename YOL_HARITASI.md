# Kredi Risk Projesi: Yol Haritası

Bu dosya projeyi baştan sona yapman için hazırlandı. Her aşamada neden o işi yaptığın, ne yapacağın ve işin ne zaman bitmiş sayılacağı yazıyor.

Yanında iki dosya daha var:

- **Tasarım:** `docs/superpowers/specs/2026-11-10-alternatif-veri-kredi-risk-design.md`. Neyi neden seçtiğimizi anlatır.
- **Teknik plan:** `docs/superpowers/plans/2026-11-10-alternatif-veri-kredi-risk.md`. Dosya adlarını, fonksiyon adlarını ve hazır test örneklerini içerir. Aşamalardaki "Görev N" ifadesi bu dosyayı gösterir.

## Çalışma düzeni

Her aşamada aynı döngüyü izle:

1. Aşamayı oku, neden yapıldığını anla.
2. Teknik plandaki ilgili görevin testini yaz.
3. Testi çalıştır ve başarısız olduğunu gör.
4. Kodu yaz, test geçene kadar düzelt.
5. Git ile kaydet (commit).

Önce test yazmak, kodun gerçekten istediğin işi yaptığını kanıtlar.

Kaydetmek için:

```powershell
git add .
git commit -m "Yaptığın işi kısaca yaz"
git push
```

Teknik plandaki `Co-Authored-By` satırını kendi commit'lerine ekleme, o satır yalnızca Claude'un yaptığı commit'ler içindi.

**Paket kurulumu:** Her aşamaya başlarken o aşamanın paketlerini kur. Bu listenin tamamının Python 3.13 ile birlikte kurulabildiği 2026 sonunda kontrol edildi.

| Aşama | Komut |
|---|---|
| 2-3 | Hazır: `pandas`, `pandera`, `pyarrow`, `pytest`, `pytest-mock`, `ruff`, `pyyaml` |
| 4 | `pip install scikit-learn lightgbm catboost optbinning` |
| 5 | `pip install imbalanced-learn optuna` |
| 6 | `pip install mlflow` |
| 7 | `pip install fairlearn evidently` |
| 8 | `pip install fastapi uvicorn httpx shap` |
| 9-11 | `pip install django djangorestframework djangorestframework-simplejwt drf-spectacular pytest-django dj-database-url celery redis "psycopg[binary]"` |
| 12 | `pip install faker sentence-transformers` |
| 13 | `pip install streamlit` |

**Bana sorarken** şunları yaz: hangi aşamadasın, ne yapmaya çalıştın, hata mesajının tamamı ne. Sadece anlamadığın bir şey de sorabilirsin. Örnek: "Aşama 6'da kalibrasyonun ne işe yaradığını anlamadım."

## Süre tahmini

Haftada 10-15 saat çalışırsan kabaca 12-16 hafta sürer. Bu yalnızca kaba bir tahmindir.

| Bölüm | Aşamalar | Kaba süre |
|---|---|---|
| Hazırlık ve veri | 0-3 | 2-3 hafta |
| Model | 4-7 | 4-5 hafta |
| Servisler | 8-11 | 3-4 hafta |
| Demo ve bitiriş | 12-14 | 3-4 hafta |

Süren kısalırsa önce Aşama 12'yi (sahte veri ve Model B) kısalt. Modeli ve API'yi kısaltma.

---

## Aşama 0: Hazırlık

**Neden:** Kod yazmadan önce araçların ve verinin hazır olması gerekir.

- [x] VS Code'u kur, Python eklentisini ekle.
- [x] Docker Desktop'ın açıldığını kontrol et.
- [x] Bu klasörde sanal ortamı etkinleştir:

```powershell
.\.venv\Scripts\Activate.ps1
```

- [x] Kaggle'da `home-credit-default-risk` yarışma sayfasını aç, kuralları kabul et, tüm veriyi indir.
- [x] Dosyaları `data/raw/` klasörüne çıkar. `application_train.csv` doğrudan bu klasörde olmalı.
- [x] `data/` klasörünü asla GitHub'a yükleme. Kaggle verisi paylaşılamaz, `.gitignore` bunu engeller.

**Bitti sayılır:** `python --version` çalışıyor ve `data/raw/application_train.csv` var.

**Not:** Bilgisayarında Python 3.13 var. Bir paket kurulurken hata alırsan Python 3.12 kurup sanal ortamı onunla yeniden oluşturmak gerekebilir.

## Aşama 1: Veriyi tanı

**Neden:** Tanımadığın veriyle model kurarsan hatalarını fark edemezsin. Bu aşamada sunumda kullanacağın grafikler de çıkar.

- [x] `pip install jupyter matplotlib` ile araçları kur.
- [x] `notebooks/01_veri_tanima.ipynb` dosyasını oluştur.
- [x] Şu sorulara cevap bul:
  - Kaç başvuru var, krediyi ödemeyenlerin oranı ne? (yaklaşık %8 bekliyoruz)
  - Hangi kolonlarda çok eksik değer var?
  - `DAYS_EMPLOYED` dağılımında garip bir değer görüyor musun? (`365243` çıkmalı)
  - Her tablo diğerine hangi kolonla bağlanıyor?
  - `bureau` tablosunda hiç kaydı olmayan kaç kişi var?
- [x] Bulduklarını not al ve 3-4 grafik kaydet.

**Bitti sayılır:** Veriyi bir arkadaşına iki dakikada anlatabiliyorsun.

## Aşama 2: Veri kontrolü (Görev 1)

**Neden:** Bozuk bir dosya sessizce modele girerse sonuçların yanlış olur. Bu kod, veriyi kullanmadan önce kontrol eder.

- [xd] Teknik plandaki Görev 1'i aç. Git deposu, `.venv`, `.gitignore`, `pyproject.toml` ve `.env.example` hazır, onları atla.
- [x] `packages/credit_features` paketini, `ml/ingest.py` dosyasını ve `tests/conftest.py` dosyasını oluştur.
- [x] Paketi kur: `pip install -e packages/credit_features`
- [x] Testleri yaz, başarısız olduğunu gör, sonra kodu yaz.
- [x] CSV dosyalarını Parquet formatına çevirmek için proje klasöründe şunu çalıştır: `python -c "from pathlib import Path; from ml.ingest import convert_raw; convert_raw(Path('data/raw'), Path('data/parquet'))"`
- [x] Parquet dosyalarının `data/parquet/` altında oluştuğunu gör.

**Bitti sayılır:** `python -m pytest tests/features/test_schemas.py -q` geçiyor ve 8 Parquet dosyası oluştu.

## Aşama 3: Programın kullanacağı bilgiler (Görev 2-3)

**Neden:** Model ham tablolardan öğrenemez, kişi başına tek satır özet ister. Projenin asıl değeri buradaki özetlerden çıkar.

Üç bilgi grubu hazırlayacaksın:

| Grup | İçerik | Nerede kullanılır |
|---|---|---|
| G1 | Başvuru formu: gelir, kredi tutarı, oranlar | Ana model |
| G2 | Ödeme alışkanlığı: gecikme, eksik ödeme | Ana model |
| G3 | Kredi skorları ve banka geçmişi | Yalnızca karşılaştırma |

- [x] `build_g1` fonksiyonunu yaz (Görev 2).
- [x] `build_g2` ve `build_g3` fonksiyonlarını yaz (Görev 3).
- [x] Şu üç kurala dikkat et:
  - Gelir 0 ise oran sonsuz değil, eksik değer olmalı.
  - `DAYS_EMPLOYED = 365243` eksik değerdir.
  - Cinsiyet, yaş, medeni durum, çocuk sayısı ve aile büyüklüğü modele girmez.

**Bitti sayılır:** `python -m pytest tests/features -q` geçiyor.

## Aşama 4: Sınav ayırma ve ilk deney (Görev 4)

**Neden:** Projenin ana sorusu burada cevaplanır: Banka bilgisi olmadan da iyi tahmin yapılabiliyor mu?

- [ ] Verinin %20'sini ayır ve **son aşamaya kadar kullanma**. Bu senin sınav kâğıdın.
- [ ] Puanları hesaplayan `evaluate` fonksiyonunu yaz.
- [ ] Basit bir temel model kur (lojistik regresyon). Diğer modelleri bununla kıyaslayacaksın.
- [ ] Üç deneyi çalıştır: G1, G1+G2, G1+G2+G3.
- [ ] Sonuçları bir de banka kaydı olmayan kişiler için ayrı hesapla.
- [ ] G1+G2+G3 modeli için `results/kaggle_submission.csv` üret ve Kaggle'a kendin "late submission" olarak yükle. Bu yalnızca ek bir kontroldür.

**Puanların anlamı:**

| Puan | Ne söyler |
|---|---|
| ROC-AUC / Gini | Riskli kişiyi risksizden ayırma gücü. Yüksek olması iyidir. |
| KS | İki grubun en çok ayrıştığı nokta. Bankalarda yaygındır. |
| PR-AUC | Az sayıdaki batan müşteriyi yakalama başarısı. |
| Brier | Verilen yüzdelerin gerçeğe yakınlığı. Düşük olması iyidir. |

"Accuracy" kullanma. Herkese "öder" diyen program bile %92 alır.

**Bitti sayılır:** `results/feature_comparison.json` oluştu ve "Ödeme bilgisi modeli ne kadar iyileştirdi?" sorusuna sayıyla cevap verebiliyorsun.

**Beklenti:** Kredi skorları olmadan puan biraz düşer. Bu normaldir ve projenin bulgusudur.

## Aşama 5: Az sayıda batan müşteri sorunu (Görev 5)

**Neden:** Batan müşteri az olduğu için model onları görmezden gelebilir.

- [ ] LightGBM ve CatBoost için üç yöntemi dene: hiçbir şey yapma, batanlara daha çok ağırlık ver, SMOTE ile yapay örnek üret.
- [ ] SMOTE'u **yalnızca eğitim verisine** uygula. Teste karışırsa sonuçlar sahte yükselir.
- [ ] En iyi ayarları Optuna ile ara.
- [ ] En iyi yöntemi seç ve nedenini not al.

SMOTE yavaş çalışırsa üç yöntemi de aynı küçük veri parçasıyla karşılaştır.

**Bitti sayılır:** `results/imbalance_comparison.json` içinde kazanan yöntem yazıyor.

## Aşama 6: Yüzdeleri düzeltme ve son sınav (Görev 6)

**Neden:** Program "%20 ihtimalle ödemez" diyorsa, gerçekten de o tür kişilerin yaklaşık %20'si ödememeli. Buna kalibrasyon denir.

- [ ] Kalibrasyonu yap ve önceki/sonraki Brier puanını karşılaştır.
- [ ] Final modeli eğit, `models/model_a/` altına kaydet.
- [ ] MLflow'a `data_scope=real` etiketiyle kaydet.
- [ ] **Şimdi** ayırdığın %20'lik sınavı yap. Bir kez yap, sonra modeli değiştirip tekrar deneme.
- [ ] Banka kaydı olmayan kişilerdeki kazancı güven aralığıyla hesapla.

**Bitti sayılır:** `results/final_evaluation.json` hazır. Sonuç beklediğin gibi çıkmasa bile raporda dürüstçe yaz.

## Aşama 7: Karar kuralı ve adillik (Görev 7)

**Neden:** Bir yüzde tek başına karar değildir. Ayrıca sistemin bazı gruplara haksızlık edip etmediğini bilmen gerekir.

- [ ] Karar kuralını yaz: %12,5 altı onay, %25 ve üstü ret, arası manuel inceleme.
- [ ] Bu sınırları değiştirince sonucun nasıl değiştiğini tabloyla göster.
- [ ] Fairlearn ile cinsiyet ve yaş gruplarında onay ve hata oranlarını karşılaştır.
- [ ] Evidently ile tek seferlik veri kayması raporu üret.

**Bitti sayılır:** Sunumda "Sistem adil mi?" sorusuna sayılarla cevap verebiliyorsun.

## Aşama 8: Skor üreten servis (Görev 8)

**Neden:** Model şu an yalnızca bilgisayarındaki bir dosya. FastAPI onu başka programların çağırabileceği bir servise çevirir.

- [ ] `POST /score` ve `GET /health` uç noktalarını yaz.
- [ ] SHAP ile "neden" bilgisini ekle, her başvuruda üç neden dönsün.
- [ ] `config/reason_codes.yaml` içine Türkçe neden metinlerini yaz.
- [ ] Eğitimdeki ve servisteki özelliklerin aynı çıktığını test et.

Çalıştırmak için `services/scoring` klasöründe:

```powershell
uvicorn app.main:app --reload
```

Sonra tarayıcıda `http://127.0.0.1:8000/docs` adresini aç.

**Bitti sayılır:** `/docs` sayfasından bir başvuru gönderip yüzde ve üç neden alabiliyorsun.

## Aşama 9: Django temeli (Görev 9)

**Neden:** Kullanıcı girişi ve kalıcı kayıtlar burada tutulur.

- [ ] Django projesini `services/api` altında `core` adıyla kur.
- [ ] `Application`, `BehaviourFeatures`, `Decision`, `Batch`, `BatchItem` modellerini yaz.
- [ ] JWT ile giriş ekle.
- [ ] Karar kayıtlarının değiştirilemez olduğunu test et.

`pyproject.toml` içine pytest için `DJANGO_SETTINGS_MODULE = "core.settings"` satırını bu aşamada ekle.

**Bitti sayılır:** Token alabiliyorsun ve bir kararı değiştirmeye çalışınca hata alıyorsun.

## Aşama 10: Başvuru API'si (Görev 10)

**Neden:** Sistemin ana akışı burada birleşir: başvuru gelir, skor alınır, karar kaydedilir.

- [ ] `POST /api/v1/applications` uç noktasını yaz.
- [ ] Skor servisini çağıran istemciyi yaz.
- [ ] Şu durumları test et:
  - Skor servisi kapalı: `503` döner, başvuru `pending` kalır.
  - Karar verilmiş başvuru tekrar gönderilir: `409` döner.
  - Bilinmeyen kişi: skorlanır ve işaretlenir.

**Bitti sayılır:** Django üzerinden bir başvuru gönderip kaydedilmiş karar görebiliyorsun.

## Aşama 11: Toplu işlemler (Görev 11)

**Neden:** Çok sayıda başvuruyu tek tek beklemeden işlemek ve özetleri veritabanına yüklemek için Celery kullanılır.

- [ ] Özet bilgileri veritabanına yükleyen görevi yaz.
- [ ] Toplu skorlama görevini yaz.
- [ ] İş ortasında skor servisi kapanırsa işlenenlerin korunduğunu test et.

**Bitti sayılır:** Toplu iş gönderip sonucunu `GET /api/v1/batches/{id}` ile görebiliyorsun.

## Aşama 12: Sahte veri ve Model B (Görev 12)

**Neden:** Elimizde gerçek e-ticaret ve Türkçe metin verisi yok. Sahte veri, bu hattın çalıştığını göstermek içindir.

- [ ] Faker ile Türkçe sipariş, yorum ve fatura verisi üret.
- [ ] Yorumları sayısal özelliklere çevir.
- [ ] Sahte kolonların adını `syn_` ile başlat.
- [ ] Model B'yi `data_scope=synthetic_demo` etiketiyle eğit.

**Önemli:** Sahte veriye sinyali sen koyuyorsun. Model B'nin puanını asla başarı olarak gösterme ve sunumda sahte olduğunu açıkça söyle.

**Bitti sayılır:** Model B çalışıyor ve Model A'da hiçbir `syn_` kolonu yok.

## Aşama 13: Demo ve tek komutla çalıştırma (Görev 13)

**Neden:** Jüri veya işveren kodunu değil, çalışan sistemi görür.

- [ ] Streamlit ekranını yaz: kişi seç, formu düzenle, sonucu gör.
- [ ] Ekranda yüzde, karar, üç neden ve model etiketi görünsün.
- [ ] `docker-compose.yml` dosyasını yaz. FastAPI dışarıya port açmamalı.
- [ ] GitHub Actions ile testlerin otomatik çalışmasını sağla.
- [ ] `.env.example` dosyasını `.env` adıyla kopyala ve örnek şifreleri değiştir.
- [ ] Dockerfile'larda Python 3.13 kullan. `models/`, `config/` ve `data/` klasörlerini Compose'da ilgili servislere bağla (volume), çünkü `models/` ve `data/` GitHub'da yok.
- [ ] Sistemi çalıştır:

```powershell
docker compose up --build -d
```

- [ ] Sistem açılınca veritabanını hazırla: `migrate`, ardından gerçek ve sahte veri için `load_features`, son olarak `seed_demo`. Bunu yapmazsan herkes "bilinmeyen kişi" görünür.

**Bitti sayılır:** `http://localhost:8501` açılıyor ve bir başvuru baştan sona skorlanıyor.

## Aşama 14: Rapor ve sunum

**Neden:** Projenin değeri, sonuçları anlatabildiğin kadar görünür olur.

Rapor bölümleri:

- [ ] Problem: Banka geçmişi olmayan insanlar neden krediye erişemiyor?
- [ ] Veri: Home Credit neden seçildi, ödeme verisi neden fatura yerine vekil kullanıldı?
- [ ] Yöntem: Bilgi grupları, doğrulama, dengesizlik, kalibrasyon.
- [ ] Sonuçlar: Üç deneyin tablosu ve banka kaydı olmayanlardaki sonuç.
- [ ] Açıklanabilirlik ve adillik.
- [ ] Mimari: Servisler ve veri akışı.
- [ ] Hukuk ve etik: KVKK m. 11, sosyal medya verisinin neden kullanılmadığı.
- [ ] Sınırlılıklar: Tarih bilgisi yok, sahte veri, reddedilenlerin sonucu bilinmiyor.

Sunum için en güçlü sıra:

1. Sorun
2. Canlı demo
3. Ana deney sonucu
4. Bir kararın nedenlerini gösterme
5. Adillik ve sınırlılıklar

**Bitti sayılır:** README'yi okuyan biri projeyi kurup çalıştırabiliyor.

---

## Kısa sözlük

| Terim | Sade anlamı |
|---|---|
| Model | Geçmiş örneklerden öğrenip tahmin yapan program |
| Özellik (feature) | Modele verilen tek bir bilgi, örneğin gecikme oranı |
| Eğitim | Modelin geçmiş verilerden öğrenmesi |
| Çapraz doğrulama | Veriyi 5 parçaya bölüp sırayla test ederek daha güvenilir puan almak |
| Sızıntı (leakage) | Modelin görmemesi gereken bilgiyi yanlışlıkla görmesi |
| SMOTE | Az olan gruba benzer yapay örnekler üretme |
| Kalibrasyon | Verilen yüzdeleri gerçeğe yaklaştırma |
| SHAP | Kararı hangi bilginin ne kadar etkilediğini gösterme |
| API | Programların birbirinden iş istediği kapı |
| JWT | Giriş yaptığını kanıtlayan dijital bilet |
| Docker | Programı her bilgisayarda aynı çalışan kutuya koyma |
| Commit | Kodun o anki halini Git'e kaydetme |
