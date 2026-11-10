# Alternatif Veriyle Kredi Risk Analizi: Tasarım

Tarih: 2026-11-10
Proje türü: Bitirme / portföy

## 1. Amaç

Bankacılık geçmişi sınırlı kişiler için kredi riskini alternatif verilerle tahmin eden, kararlarını açıklayabilen ve API üzerinden kullanılabilen uçtan uca bir sistem geliştirmek.

Projenin ana araştırma sorusu: **Geleneksel kredi verisi olmadan, başvuru bilgisi ve ödeme davranışıyla ne kadar iyi tahmin yapılabilir?**

## 2. Kapsam

Kapsam içinde:

- Home Credit Default Risk verisiyle model eğitimi ve değerlendirme
- Türkçe sentetik e-ticaret, yorum ve fatura verisi üretimi ve işlenmesi
- Django ana API, FastAPI skorlama servisi, Celery toplu işleri
- SHAP ile karar nedenleri, Fairlearn ile adillik ölçümü
- Streamlit demo ekranı

Kapsam dışında:

- Sosyal medya verisi. Raporda KVKK, platform erişimi ve ayrımcılık riski gerekçesiyle "gelecek çalışma" olarak anlatılır.
- Gerçek kişisel veri, gerçek kurum entegrasyonu
- Reject inference, Kubernetes, Kafka, Prometheus, Grafana, Sentry
- Canlı drift izleme. Evidently ile tek seferlik rapor üretilir.

## 3. Teknoloji Seçimleri

| Alan | Seçim |
|---|---|
| Dil | Python |
| Ana API | Django, Django REST Framework, drf-spectacular, djangorestframework-simplejwt |
| Skorlama | FastAPI |
| Veritabanı | PostgreSQL |
| Asenkron işler | Celery, Redis |
| Veri saklama ve doğrulama | Yerel Parquet, Pandera, Pydantic |
| Sentetik veri | Faker (`tr_TR`), sabit seed |
| Metin | sentence-transformers, `paraphrase-multilingual-MiniLM-L12-v2`, PCA |
| Modeller | Lojistik regresyon + WoE (optbinning), LightGBM, CatBoost |
| Model geliştirme | scikit-learn, imbalanced-learn, Optuna, MLflow |
| Açıklanabilirlik ve adillik | SHAP (TreeSHAP), Fairlearn |
| Veri kayması raporu | Evidently |
| Demo | Streamlit |
| Dağıtım ve kalite | Docker Compose, pytest, ruff, GitHub Actions |

## 4. Mimari

### Servisler

| Birim | Görevi | Bağımlılığı |
|---|---|---|
| `services/api` (Django) | Kimlik doğrulama, başvurular, karar kayıtları, toplu iş başlatma | PostgreSQL, Redis, skorlama servisi |
| `services/api` Celery işçisi | Veri yükleme ve toplu skorlama | PostgreSQL, Redis, skorlama servisi |
| `services/scoring` (FastAPI) | Özellik üretimi, tahmin, SHAP. Durum tutmaz. | `packages/features`, model dosyaları |
| `packages/features` | Eğitim ve skorlamanın ortak özellik kodu | Yok |
| `ml/` | Sentetik veri üretimi, eğitim, deney ve rapor komutları | `packages/features`, MLflow |
| `demo/` (Streamlit) | Sunum ekranı. Yalnızca Django API'yi çağırır. | Django API |

Django, ML kütüphanelerine bağımlı değildir. Home Credit verisi Kaggle kurallarına tabidir ve depoya eklenmez, `data/` klasörü `.gitignore` içinde yer alır.

### Modeller

| Model | Veri | Kullanım |
|---|---|---|
| Model A | G1 + G2 | Varsayılan skorlama modeli. Raporlanan tüm başarı sonuçları bu model ve kıyas deneylerinden gelir. |
| Model B | G1 + G2 + sentetik özellikler | Yalnızca veri hattının çalıştığını gösterir. Metrikleri raporlanmaz ve kıyas tablolarına girmez. |

İki model de MLflow'da `data_scope` etiketiyle (`real` veya `synthetic_demo`) kayıtlıdır. Demo ekranı kullanılan modeli ve sentetik veri içerip içermediğini gösterir.

### Çevrimdışı veri akışı

1. Home Credit CSV dosyaları ve sentetik üretici çıktıları Pandera ile doğrulanır.
2. Doğrulanan veri Parquet olarak yazılır.
3. `packages/features` özellikleri üretir. Metinler MiniLM ve PCA ile sayısal özelliklere dönüşür.
4. Geçmişe dayalı özet özellikler (G2 ve sentetik özellikler) PostgreSQL'e `SK_ID_CURR` anahtarıyla yüklenir.
5. Eğitim komutları modelleri üretir ve MLflow'a kaydeder.

### Çevrimiçi skorlama akışı

1. İstemci, `SK_ID_CURR` referansı ve başvuru formu alanlarıyla `POST /api/v1/applications` çağırır.
2. Django JWT ve girdiyi doğrular, başvuruyu kaydeder, kişinin özet özelliklerini PostgreSQL'den okur.
3. Django, form alanlarını ve özet özellikleri FastAPI `POST /score` uç noktasına gönderir.
4. FastAPI G1 özelliklerini üretir, olasılığı ve neden kodlarını döndürür.
5. Django karar kaydını yazar, olasılık, bant ve nedenleri yanıtlar.

Diğer uç noktalar: `GET /api/v1/applications/{id}`, `POST /api/v1/applications/{id}/score` (bekleyen başvuruyu yeniden skorlar), `POST /api/v1/batches` (toplu skorlama).

İstekteki `model` alanı `A` veya `B` olabilir, varsayılanı `A`'dır.

### Karar kaydı

Model sürümü, kullanılan özelliklerin anlık görüntüsü, olasılık, bant ve neden kodları saklanır. API'de güncelleme veya silme yolu yoktur, Django modeli de kayıt oluştuktan sonra güncelleme ve silmeyi reddeder.

### Hata durumları

- Skorlama servisine ulaşılamazsa Django `503` döner, başvuru `pending` kalır ve varsayılan skor üretilmez.
- Şemaya uymayan girdi `422` alır. Eksik değerlere izin verilir.
- `SK_ID_CURR` için özet özellik yoksa G2 alanları eksik değer olarak gönderilir ve kayıtta işaretlenir.

## 5. Veri ve Özellikler

| Grup | Kaynak | Kullanım |
|---|---|---|
| G1: Başvuru | `application_train/test` içindeki gelir, tutar, oranlar, istihdam süresi | Model A |
| G2: Ödeme davranışı | `installments_payments`, `POS_CASH_balance`, `credit_card_balance`, `previous_application` özetleri | Model A |
| G3: Geleneksel kredi | `bureau`, `bureau_balance`, `EXT_SOURCE_1/2/3` | Yalnızca kıyas |

Kurallar:

- G2, fatura ödeme alışkanlığının vekili olarak kullanılır. Asıl kayıtların kredi taksiti olduğu raporda açıkça belirtilir.
- `DAYS_EMPLOYED = 365243` eksik değer sayılır.
- Demografik alanlar (`CODE_GENDER`, `DAYS_BIRTH`, `NAME_FAMILY_STATUS`, `CNT_CHILDREN`, `CNT_FAM_MEMBERS`) özellik olarak kullanılmaz. Cinsiyet ve yaş grubu yalnızca adillik ölçümünde kullanılır.
- Sentetik veri her `SK_ID_CURR` için sabit seed ile üretilir. Sinyal üretici tarafından konduğu için üzerinden başarı iddia edilmez ve sunumda sentetik olduğu belirtilir.

## 6. Model Geliştirme

### Doğrulama

- Başta `application_train` verisinin %20'si stratified test seti olarak ayrılır ve yalnızca son değerlendirmede bir kez kullanılır.
- Kalan %80 üzerinde 5 katlı stratified çapraz doğrulama yapılır.
- Kaggle geç gönderimi yalnızca G1+G2+G3 kıyas modeli için yapılır.
- Veride gerçek başvuru tarihi olmadığı için zamana göre ayrım yapılmaz. Bu sınırlılık raporda belirtilir.

### Deneyler

1. **Özellik karşılaştırması:** G1, G1+G2 ve G1+G2+G3. Sonuçlar tüm veri ve `bureau` kaydı olmayan grup için ayrı raporlanır.
2. **Dengesizlik karşılaştırması:** LightGBM ve CatBoost için düzeltme yok, sınıf ağırlığı ve SMOTENC kolları. SMOTENC yalnızca eğitim katlarına uygulanır ve eksik değerleri eğitim katından öğrenilen değerlerle doldurur. Süre sorun olursa üç kol da aynı stratified örneklem üzerinde karşılaştırılır.
3. **Temel model:** WoE'li lojistik regresyon tüm karşılaştırmalarda kıyas noktasıdır.

Seçim, çapraz doğrulamadaki PR-AUC ve kalibrasyon sonrası Brier skoruyla yapılır. Sonuçta tek model kullanılır, stacking yapılmaz.

### Kalibrasyon ve metrikler

Isotonic kalibrasyon out-of-fold tahminlerle öğrenilir. Raporlanan metrikler ROC-AUC/Gini, KS, PR-AUC, Brier skoru ve kalibrasyon eğrisidir. Accuracy başarı ölçütü olarak kullanılmaz.

### Karar kuralı

Eşikler `config/decision.yaml` dosyasından okunur. Başlangıç varsayımları faiz marjı `m = 0,15` ve temerrütte kayıp `LGD = 0,45` değerleridir.

- Ret eşiği: `T_ret = m / (m + LGD) = 0,25`
- Onay eşiği: `T_onay = T_ret / 2 = 0,125`

| Kalibre olasılık | Bant |
|---|---|
| `PD < T_onay` | Onay |
| `T_onay <= PD < T_ret` | Manuel inceleme |
| `PD >= T_ret` | Ret |

Bu değerler gerçek kurum verisi değil varsayımdır. Raporda farklı `m` ve `LGD` değerleri için duyarlılık analizi yer alır.

### Açıklanabilirlik ve adillik

- SHAP değerleri kalibrasyon öncesi model çıktısından hesaplanır.
- Riski en çok artıran üç özellik, eşleme tablosuyla Türkçe neden metnine çevrilir.
- Fairlearn ile cinsiyet ve yaş grubuna göre onay oranı, gerçek pozitif ve yanlış pozitif oranı farkları seçilen eşiklerde ölçülür.

## 7. Başarı Ölçütleri

1. G1+G2 boosting modeli, çapraz doğrulamada lojistik regresyonu ROC-AUC açısından en az 0,01 geçer.
2. Test setinde, `bureau` kaydı olmayan grupta G1+G2 modeli G1 modelinden iyidir. ROC-AUC farkının bootstrap %95 güven aralığı sıfırı içermez.
3. Olasılıklar için kalibrasyon sonrası Brier skoru, kalibrasyon öncesinden kötü değildir.
4. Demo ekranında bir başvuru uçtan uca skorlanır ve olasılık, bant, en az üç neden gösterilir.
5. `docker compose up` ile tüm servisler ayağa kalkar ve CI testleri geçer.

1 ve 2 numaralı ölçütler tutmazsa sonuç gizlenmez, olumsuz bulgu olarak raporlanır.

## 8. Test Stratejisi

- **Sızıntı testi:** SMOTENC ve doldurma adımlarının yalnızca eğitim katında öğrenildiği doğrulanır.
- **Özellik eşitliği testi:** Aynı girdi için eğitim ve skorlama akışlarının özellikleri eşleşir.
- **Sözleşme testleri:** Django istemcisi ile FastAPI şeması uyumludur.
- **API testleri:** Yetkisiz erişim, `422`, skorlama servisi kapalıyken `503` ve `pending`, karar kaydının değiştirilemezliği.
- **Sentetik veri testi:** Aynı seed aynı veriyi üretir, Model A'nın özellik listesinde sentetik kolon bulunmaz.

## 9. Güvenlik ve Uyum

- Gerçek kişisel veri kullanılmaz.
- Sırlar `.env` dosyasında tutulur ve depoya eklenmez, örnek değerler `.env.example` içindedir.
- FastAPI yalnızca Compose iç ağından erişilebilir, dışarıya Django açılır.
- KVKK m. 11 kapsamındaki otomatik karara itiraz hakkı, manuel inceleme bandı ve neden kodlarıyla ilişkilendirilerek raporda tartışılır.

## 10. Geliştirme Sırası

1. Veri indirme, doğrulama ve özellik paketi
2. Temel model ve özellik karşılaştırma deneyi
3. Dengesizlik deneyi, kalibrasyon, karar kuralı, SHAP, adillik
4. FastAPI skorlama servisi
5. Django API, karar kayıtları, Celery toplu işleri
6. Sentetik veri, metin hattı ve Model B
7. Streamlit demo, Docker Compose, CI ve rapor çıktıları
