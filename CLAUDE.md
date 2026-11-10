# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Çalışma şekli

Bu bir bitirme/portföy projesi ve kodu kullanıcı kendisi yazıyor. Claude, sorulduğunda açıklar ve yönlendirir, istenmeden projeyi kodlamaya devam etmez. Açıklamalar sade Türkçeyle ve önce "neden" anlatılarak yapılır.

Kaynaklar, öncelik sırasıyla:

- `docs/superpowers/specs/2026-11-10-alternatif-veri-kredi-risk-design.md`: Bağlayıcı tasarım.
- `docs/superpowers/plans/2026-11-10-alternatif-veri-kredi-risk.md`: Görev bazında dosya adları, fonksiyon imzaları ve testler.
- `YOL_HARITASI.md`: Kullanıcının izlediği aşamalar. Kutular kullanıcının ilerlemesini gösterir, "Aşama N" ile plandaki "Görev N" aynı numara değildir.

Çalışmalar doğrudan `main` dalında yapılır.

## Komutlar

Windows PowerShell, proje kökünde, `.venv` etkin (`.\.venv\Scripts\Activate.ps1`):

```powershell
pip install -e packages/credit_features
python -m pytest
python -m pytest tests/features/test_schemas.py::test_convert_raw_writes_parquet -q
python -m ruff check .
python -c "from pathlib import Path; from ml.ingest import convert_raw; print(convert_raw(Path('data/raw'), Path('data/parquet')))"
```

Son komut `data/raw/` altındaki Kaggle CSV'lerini doğrulayıp `data/parquet/` altına yazar. Notebook'lar `notebooks/` içinde çalışır ve veriye `../data/raw/` yoluyla erişir.

## Mimari

Şu an var olanlar:

- `packages/credit_features`: Eğitim ve skorlamanın ortak özellik paketi. `schemas.py` tablo başına Pandera şemalarını tutar. Özellik kodu (`application.py`, `behaviour.py`, `bureau.py`, `frame.py`) burada yazılacak.
- `ml/`: Çevrimdışı işler. `ingest.py` CSV'den Parquet'e dönüşümü yapar.
- `tests/`: `--import-mode=importlib` kullanılır, bu sayede `tests/ml` kök `ml` paketini gölgelemez. Testler gerçek veri olmadan `conftest.py` içindeki küçük tablolarla çalışır.

Planlanan, henüz olmayan parçalar: `services/scoring` (durumsuz FastAPI, tahmin ve SHAP), `services/api` (Django, JWT, başvurular, değiştirilemez karar kayıtları, Celery), `packages/credit_decision` (eşik kuralı), `demo/` (Streamlit). Akış: Streamlit → Django → FastAPI → Django karar kaydı. Özet özellikler önceden hesaplanıp PostgreSQL'e yüklenir.

Veri bağlantıları: `SK_ID_CURR` kişi anahtarıdır. `bureau_balance` tablosunda bu kolon yoktur ve kişiye `SK_ID_BUREAU` üzerinden `bureau` ile bağlanır. `installments_payments`, `POS_CASH_balance`, `credit_card_balance` ve `previous_application` kişi başına çok satırlıdır ve özetlenerek kullanılır.

## Değişmez kurallar

- Model A yalnızca G1 (başvuru) ve G2 (ödeme davranışı) kullanır. G3 (`bureau`, `bureau_balance`, `EXT_SOURCE_*`) yalnızca kıyas içindir.
- Sentetik kolonlar `syn_` önekiyle başlar ve yalnızca Model B'ye girer. Model B (`data_scope=synthetic_demo`) için metrik raporlanmaz.
- `CODE_GENDER`, `DAYS_BIRTH`, `NAME_FAMILY_STATUS`, `CNT_CHILDREN`, `CNT_FAM_MEMBERS` özellik olamaz. Cinsiyet ve yaş grubu yalnızca adillik ölçümünde kullanılır.
- `DAYS_EMPLOYED = 365243` eksik değerdir. Sıfır paydalı oranlar sonsuz değil eksik değer olur.
- %20'lik test seti yalnızca son değerlendirmede bir kez kullanılır. SMOTE ve doldurma adımları yalnızca eğitim katlarında öğrenilir. Rastgelelik `seed = 42` ile sabitlenir.
- Django servisine ML kütüphanesi eklenmez. FastAPI dışarıya port açmaz.
- Kaggle verisi (`data/`) depoya eklenmez ve not defteri çıktılarında ham satır bulunmamalıdır.

## Plandan bilinen farklar

- Plan Python 3.12 yazıyor, proje Python 3.13 kullanıyor.
- Plandaki `Co-Authored-By` satırı kullanıcının kendi commit'leri için geçerli değildir.
- `DJANGO_SETTINGS_MODULE = "core.settings"` pytest ayarına Django kurulurken (Aşama 9) eklenir, modül var olmadan eklenirse testler bozulabilir. Django projesinin adı `core` olur, çünkü kökteki `config/` YAML ayarlarına ayrılmıştır.

## Bilinen tuzaklar

- Pandera `pandera.pandas` üzerinden içe aktarılır. Tür dönüştürme hatası `SchemaError` yerine `SchemaErrors` fırlatabilir.
- `HomeCredit_columns_description.csv` `encoding="latin-1"` ile okunur.
