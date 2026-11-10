# Alternatif Veriyle Kredi Risk Analizi Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Home Credit verisiyle kredi riskini tahmin eden, kararlarını açıklayan, API ve demo ekranıyla çalışan uçtan uca sistemi kurmak.

**Architecture:** Ortak `credit_features` paketi eğitim ve skorlamada aynı özellikleri üretir. Durumsuz FastAPI servisi tahmin ve SHAP nedenlerini döndürür, Django kimlik doğrulama, başvuru, karar kaydı ve toplu işleri yönetir. Model A gerçek veriyle, Model B yalnızca demo için sentetik veriyle eğitilir.

**Tech Stack:** Python 3.12, Django + DRF, FastAPI, PostgreSQL, Celery + Redis, LightGBM, CatBoost, optbinning, imbalanced-learn, Optuna, SHAP, Fairlearn, MLflow, Evidently, sentence-transformers, Streamlit, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-11-10-alternatif-veri-kredi-risk-design.md`

## Global Constraints

- Rastgelelik kullanan her adımda `seed = 42`.
- Test seti: `application_train` verisinin %20'si, yalnızca son değerlendirmede bir kez kullanılır.
- Çapraz doğrulama: kalan %80 üzerinde 5 katlı stratified.
- Demografik alanlar özellik olamaz: `CODE_GENDER`, `DAYS_BIRTH`, `NAME_FAMILY_STATUS`, `CNT_CHILDREN`, `CNT_FAM_MEMBERS`.
- `DAYS_EMPLOYED = 365243` eksik değerdir.
- Model A yalnızca G1 + G2 kullanır. G3 yalnızca kıyas içindir. Sentetik kolonlar `syn_` önekiyle başlar ve yalnızca Model B'ye girer.
- MLflow etiketi `data_scope`: Model A `real`, Model B `synthetic_demo`. `synthetic_demo` için metrik raporlanmaz.
- Karar varsayımları: `m = 0.15`, `LGD = 0.45`, `T_ret = m / (m + LGD)`, `T_onay = T_ret / 2`.
- Django servisine ML kütüphanesi eklenmez (pandas, numpy, lightgbm, catboost, shap, sklearn).
- FastAPI dışarıya port açmaz.
- Stacking yok. Accuracy başarı ölçütü değildir.
- `data/`, `models/`, `mlruns/`, `results/raw/` ve `.env` depoya eklenmez.
- Testler gerçek Home Credit verisi olmadan çalışır, küçük sahte tablolar kullanır.
- Geliştirme ortamı: kökte tek `.venv`, Docker imajlarında servis başına ayrı `requirements.txt`.
- Her görev yalnızca kendi Files listesindeki dosyaları commit eder. Commit mesajları şu satırla biter: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`

## Review Focus

- Geliri veya kredi tutarı 0 olan başvuru: oranlar sonsuz değil eksik değer olur ve skorlama 200 döner (Görev 2, Görev 8).
- Riski artıran özelliği üçten az olan düşük riskli başvuru: yine üç neden döner, her biri yönünü belirtir (Görev 8).
- Kararı verilmiş başvuruyu yeniden skorlama: `409` döner, ikinci karar kaydı oluşmaz (Görev 10).
- Toplu iş sırasında skorlama servisinin kapanması: işlenenler korunur, kalanlar `pending` olur, iş `partial` biter (Görev 11).
- Model dosyası eksik veya özellik listesi uyumsuz: servis sessizce skor üretmez, `/health` 503 döner (Görev 8).

---

## Dosya Yapısı

```text
packages/credit_features/src/credit_features/   schemas, application, behaviour, bureau, synthetic, frame
packages/credit_decision/src/credit_decision/   eşik ve bant kuralı (yalnızca standart kütüphane + PyYAML)
ml/                                             split, metrics, cv, baseline, imbalance, tune, calibrate, train_final, final_eval, stats, fairness
ml/experiments/  ml/reports/  ml/synthetic/     deneyler, raporlar, sentetik üretici
services/scoring/app/                           main, schemas, bundle, reasons
services/api/core/  services/api/credit/        Django ayarları ve uygulaması
demo/                                           Streamlit ekranı
config/                                         decision.yaml, reason_codes.yaml
tests/                                          features, ml, scoring, api, integration
```

### Task 1: Proje iskeleti ve veri doğrulama

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `tests/conftest.py`, `packages/credit_features/pyproject.toml`, `packages/credit_features/src/credit_features/__init__.py`, `packages/credit_features/src/credit_features/schemas.py`, `ml/ingest.py`
- Test: `tests/features/test_schemas.py`

**Interfaces:**
- Produces: `credit_features.schemas.SCHEMAS: dict[str, pandera.DataFrameSchema]` (anahtarlar Home Credit tablo adları), `ml.ingest.convert_raw(raw_dir: Path, out_dir: Path) -> list[Path]`

- [ ] **Step 1:** `git init -b main`, `.venv` oluştur, ruff ayarlarını ve `.gitignore` kurallarını yaz. Pytest ayarları: `addopts = "--import-mode=importlib"`, `pythonpath = [".", "services/api", "services/scoring"]`, `DJANGO_SETTINGS_MODULE = "core.settings"`. Paketi `pip install -e packages/credit_features` ile kur.
- [ ] **Step 2: Failing test**

```python
def test_application_rejects_duplicate_ids(app_df):
    bad = pd.concat([app_df, app_df.head(1)])
    with pytest.raises(pandera.errors.SchemaError):
        SCHEMAS["application_train"].validate(bad)

def test_convert_raw_writes_parquet(tmp_path, raw_csv_dir):
    paths = convert_raw(raw_csv_dir, tmp_path)
    assert all(p.suffix == ".parquet" for p in paths) and paths
```

- [ ] **Step 3:** `python -m pytest tests/features/test_schemas.py -v` çalıştır. Beklenen: FAIL.
- [ ] **Step 4:** Şemaları ve `convert_raw` fonksiyonunu yaz. Şemalar yalnızca kullanılan kolonları doğrular, `SK_ID_CURR` eksik olamaz, `application_*` içinde tekildir, diğer kolonlarda eksik değere izin verilir.
- [ ] **Step 5:** Testi çalıştır. Beklenen: PASS. **Commit:** `chore: scaffold project and validate raw data`

### Task 2: Başvuru özellikleri (G1) ve kolon hizalama

**Files:**
- Create: `packages/credit_features/src/credit_features/application.py`, `packages/credit_features/src/credit_features/frame.py`
- Test: `tests/features/test_application.py`

**Interfaces:**
- Produces: `RAW_G1_COLUMNS: list[str]`, `DEMOGRAPHIC_COLUMNS: list[str]`, `build_g1(df: pd.DataFrame) -> pd.DataFrame`, `frame.FeatureSpec(numeric: list[str], categorical: dict[str, list[str]])`, `frame.fit_spec(df) -> FeatureSpec`, `frame.align_frame(df, spec: FeatureSpec) -> pd.DataFrame`

Ham kolonlar: `AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE, DAYS_EMPLOYED, NAME_CONTRACT_TYPE, NAME_INCOME_TYPE, OCCUPATION_TYPE, ORGANIZATION_TYPE, FLAG_OWN_CAR, FLAG_OWN_REALTY, REGION_RATING_CLIENT`. Türetilenler: `credit_income_ratio = AMT_CREDIT / AMT_INCOME_TOTAL`, `annuity_income_ratio = AMT_ANNUITY / AMT_INCOME_TOTAL`, `credit_goods_ratio = AMT_CREDIT / AMT_GOODS_PRICE`, `employed_years = -DAYS_EMPLOYED / 365.25`.

- [ ] **Step 1: Failing test**

```python
def test_employed_sentinel_is_missing(app_df):
    app_df.loc[0, "DAYS_EMPLOYED"] = 365243
    assert np.isnan(build_g1(app_df).loc[0, "employed_years"])

def test_zero_income_gives_nan_not_inf(app_df):
    app_df.loc[0, "AMT_INCOME_TOTAL"] = 0
    out = build_g1(app_df)
    assert np.isnan(out.loc[0, "credit_income_ratio"])
    assert not np.isinf(out.select_dtypes("number")).any().any()

def test_no_demographics(app_df):
    assert not set(build_g1(app_df).columns) & set(DEMOGRAPHIC_COLUMNS)

def test_align_maps_unseen_category_to_missing(app_df):
    g1 = build_g1(app_df); spec = fit_spec(g1)
    g1.loc[0, "OCCUPATION_TYPE"] = "YENI_MESLEK"
    assert align_frame(g1, spec).loc[0, "OCCUPATION_TYPE"] == "MISSING"
```

- [ ] **Step 2:** Testi çalıştır. Beklenen: FAIL.
- [ ] **Step 3:** Fonksiyonları yaz. `align_frame` eksik kolonu NaN ile ekler, kolon sırasını `spec` ile eşitler, kategorik eksikleri ve görülmeyen değerleri `"MISSING"` yapar ve `category` tipine çevirir.
- [ ] **Step 4:** Testi çalıştır. Beklenen: PASS. **Commit:** `feat: add application features and frame alignment`

### Task 3: Ödeme davranışı (G2) ve geleneksel kredi (G3) özellikleri

**Files:**
- Create: `packages/credit_features/src/credit_features/behaviour.py`, `packages/credit_features/src/credit_features/bureau.py`
- Test: `tests/features/test_behaviour.py`, `tests/features/test_bureau.py`

**Interfaces:**
- Produces: `build_g2(installments, pos, credit_card, previous) -> pd.DataFrame`, `build_g3(application, bureau, bureau_balance) -> pd.DataFrame`, ikisi de `SK_ID_CURR` indeksli.

G2 kolonları: `g2_inst_late_days_mean`, `g2_inst_late_share`, `g2_inst_late_share_last12`, `g2_inst_underpay_ratio_mean`, `g2_pos_dpd_max`, `g2_cc_utilization_mean`, `g2_prev_count`, `g2_prev_refused_share`. Gecikme: `max(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT, 0)`. Son 12 ay: `DAYS_INSTALMENT >= -365`. Eksik ödeme: `max(AMT_INSTALMENT - AMT_PAYMENT, 0) / AMT_INSTALMENT`. Kullanım: `AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL`.

G3 kolonları: `g3_bureau_count`, `g3_bureau_active_share`, `g3_bureau_overdue_max`, `g3_bb_dpd_share`, `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`.

- [ ] **Step 1: Failing test**

```python
def test_late_payment_stats(tables):
    g2 = build_g2(*tables)  # müşteri 1: bir taksit 10 gün geç, bir taksit zamanında
    assert g2.loc[1, "g2_inst_late_share"] == pytest.approx(0.5)
    assert g2.loc[1, "g2_inst_late_days_mean"] == pytest.approx(5.0)

def test_underpayment_ignores_zero_instalment(tables):
    assert np.isfinite(build_g2(*tables).loc[2, "g2_inst_underpay_ratio_mean"])

def test_refused_share(tables):
    assert build_g2(*tables).loc[1, "g2_prev_refused_share"] == pytest.approx(1 / 3)

def test_customer_with_only_pos_row_is_present(tables):
    assert np.isnan(build_g2(*tables).loc[3, "g2_inst_late_share"])

def test_g3_keeps_external_scores(app_df, bureau, bureau_balance):
    assert "EXT_SOURCE_2" in build_g3(app_df, bureau, bureau_balance).columns
```

- [ ] **Step 2:** Testi çalıştır. Beklenen: FAIL.
- [ ] **Step 3:** Fonksiyonları yaz. Toplama işlemleri `TARGET` kullanmaz, `AMT_INSTALMENT = 0` olan satırlarda oran NaN olur.
- [ ] **Step 4:** Testi çalıştır. Beklenen: PASS. **Commit:** `feat: add behaviour and bureau features`

### Task 4: Test ayrımı, metrikler ve özellik karşılaştırma deneyi

**Files:**
- Create: `ml/split.py`, `ml/metrics.py`, `ml/dataset.py`, `ml/cv.py`, `ml/baseline.py`, `ml/models.py`, `ml/experiments/feature_comparison.py`
- Test: `tests/ml/test_split.py`, `tests/ml/test_metrics.py`, `tests/ml/test_cv.py`

**Interfaces:**
- Produces: `make_holdout(df, seed=42) -> tuple[np.ndarray, np.ndarray]` (geliştirme ve test `SK_ID_CURR` dizileri, `data/splits/holdout.parquet` içine yazılır), `evaluate(y, p) -> dict` (`roc_auc, gini, ks, pr_auc, brier`), `build_dataset(group: Literal["G1","G1G2","G1G2G3"], data_dir) -> tuple[pd.DataFrame, pd.Series, FeatureSpec]` (`SK_ID_CURR` indeksli), `thin_file_mask(index, bureau) -> pd.Series`, `cross_val_oof(make_model, X, y, n_splits=5, seed=42) -> CVResult(oof: np.ndarray, fold_metrics: list[dict])`, `make_baseline() -> Pipeline`, `make_model(name: Literal["lightgbm","catboost"], arm: Literal["none","class_weight","smotenc"], spec: FeatureSpec, params: dict | None = None) -> estimator` (bu görevde yalnızca `arm="none"` desteklenir, diğerleri `NotImplementedError` verir)

- [ ] **Step 1: Failing test**

```python
def test_holdout_stratified_and_deterministic(labelled):
    dev, test = make_holdout(labelled, seed=42)
    assert not set(dev) & set(test)
    assert len(test) / len(labelled) == pytest.approx(0.2, abs=0.01)
    rate = labelled.set_index("SK_ID_CURR").loc[test, "TARGET"].mean()
    assert rate == pytest.approx(labelled["TARGET"].mean(), abs=0.01)
    assert (make_holdout(labelled, seed=42)[1] == test).all()

def test_metrics_perfect():
    m = evaluate(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9]))
    assert m["roc_auc"] == 1 and m["gini"] == 1 and m["ks"] == 1

def test_oof_covers_every_row(xy):
    r = cross_val_oof(make_baseline, *xy, n_splits=5, seed=42)
    assert len(r.oof) == len(xy[1]) and not np.isnan(r.oof).any()
    assert len(r.fold_metrics) == 5
```

- [ ] **Step 2:** Testi çalıştır. Beklenen: FAIL.
- [ ] **Step 3:** Kodu yaz. `ks = max(tpr - fpr)`, `gini = 2 * auc - 1`. `make_baseline` optbinning WoE dönüşümü ve lojistik regresyondan oluşur.
- [ ] **Step 4:** `feature_comparison.py` komutunu yaz. G1, G1G2 ve G1G2G3 gruplarını LightGBM ve CatBoost ile, ayrıca WoE'li lojistik regresyonla çalıştırır. Tüm geliştirme verisi ve `bureau` kaydı olmayan grup için metrikleri `results/feature_comparison.json` dosyasına yazar. G1G2 için en iyi boosting ROC-AUC değeri lojistik regresyonu en az 0,01 geçiyorsa `criterion_1_passed: true` yazar. G1G2G3 için `results/kaggle_submission.csv` üretir, dosyayı kullanıcı Kaggle'a elle yükler.
- [ ] **Step 5:** Testi çalıştır. Beklenen: PASS. **Commit:** `feat: add holdout split metrics and feature comparison`
- [ ] **Step 6:** Veri `data/raw/` altındaysa `python -m ml.experiments.feature_comparison` çalıştır. Beklenen: JSON dosyası tüm gruplar için `gini`, `ks`, `pr_auc`, `brier` içerir.

### Task 5: Dengesizlik deneyi ve hiperparametre araması

**Files:**
- Create: `ml/imbalance.py`, `ml/tune.py`, `ml/experiments/imbalance_comparison.py`
- Modify: `ml/models.py`
- Test: `tests/ml/test_imbalance.py`, `tests/ml/test_tune.py`

**Interfaces:**
- Consumes: `build_dataset("G1G2", ...)`, `cross_val_oof`, `evaluate`, `make_model`
- Produces: `make_model` için `class_weight` ve `smotenc` kolları, `tune(name, arm, X, y, spec, n_trials=30, seed=42) -> dict`, `select_best(results: list[dict], tol=0.002) -> dict`

`smotenc` kolu tek `imblearn.pipeline.Pipeline` içindedir. Adımlar: `prep` (`ColumnTransformer`: `num` için `SimpleImputer(strategy="median")`, `cat` için `OrdinalEncoder`), `smote` (`SMOTENC(random_state=42)`), `model`.

`select_best`, en yüksek `pr_auc` değerine `tol` kadar yakın sonuçlar arasından en düşük `brier_calibrated` değerine sahip olanı seçer.

- [ ] **Step 1: Failing test**

```python
def test_imputer_learns_only_from_fit_data(xy_cat, spec):
    X, y = xy_cat; train, val = X.iloc[:80], X.iloc[80:].copy()
    val["AMT_CREDIT"] = 1e12
    pipe = make_model("lightgbm", "smotenc", spec).fit(train, y.iloc[:80])
    imputer = pipe.named_steps["prep"].named_transformers_["num"]
    assert imputer.statistics_[spec.numeric.index("AMT_CREDIT")] < 1e12
    assert len(pipe.predict_proba(val)) == len(val)

def test_smote_runs_once_on_fit_never_on_predict(xy_cat, spec, mocker):
    X, y = xy_cat
    spy = mocker.spy(SMOTENC, "fit_resample")
    pipe = make_model("lightgbm", "smotenc", spec).fit(X, y)
    pipe.predict_proba(X)
    assert spy.call_count == 1

def test_select_best_breaks_near_tie_with_brier():
    a = {"id": "a", "pr_auc": 0.300, "brier_calibrated": 0.070}
    b = {"id": "b", "pr_auc": 0.299, "brier_calibrated": 0.060}
    c = {"id": "c", "pr_auc": 0.250, "brier_calibrated": 0.010}
    assert select_best([a, b, c], tol=0.002)["id"] == "b"

def test_tune_is_reproducible(xy_cat, spec):
    a = tune("lightgbm", "none", *xy_cat, spec, n_trials=2, seed=42)
    assert a == tune("lightgbm", "none", *xy_cat, spec, n_trials=2, seed=42)
```

- [ ] **Step 2:** Testi çalıştır. Beklenen: FAIL.
- [ ] **Step 3:** Kodu yaz. `tune`, çapraz doğrulamadaki PR-AUC ortalamasını en büyütür.
- [ ] **Step 4:** Testi çalıştır. Beklenen: PASS.
- [ ] **Step 5:** `imbalance_comparison.py` komutunu yaz. 2 model × 3 kol için OOF metriklerini ve OOF tahminler üzerinde 5 katlı isotonic ile hesaplanan `brier_calibrated` değerini `results/imbalance_comparison.json` dosyasına yazar, `select_best` sonucunu `winner` alanına koyar, `--sample-frac` verilirse üç kol da aynı stratified örneklemi kullanır. **Commit:** `feat: add imbalance experiment and tuning`

### Task 6: Kalibrasyon, final eğitim ve tek seferlik test

**Files:**
- Create: `ml/calibrate.py`, `ml/train_final.py`, `ml/final_eval.py`, `ml/stats.py`
- Test: `tests/ml/test_calibrate.py`, `tests/ml/test_final_eval.py`, `tests/ml/test_stats.py`

**Interfaces:**
- Consumes: `make_model`, `cross_val_oof`, `build_dataset`, `evaluate`
- Produces: Bundle klasörü `models/model_a/` içinde `model.joblib`, `calibrator.joblib`, `features.json` (FeatureSpec), `metadata.json` (`model_version, data_scope, algorithm`). `fit_calibrator(oof, y) -> IsotonicRegression`, `train_bundle(model_key: Literal["A","B"], out_dir: Path) -> Path`, `final_evaluate(bundle_dir, data_dir, marker: Path) -> dict`, `bootstrap_auc_diff(y, p_a, p_b, n=2000, seed=42) -> tuple[float, float]`

- [ ] **Step 1: Failing test**

```python
def test_calibrator_outputs_probabilities():
    cal = fit_calibrator(np.linspace(0, 1, 100), np.r_[np.zeros(80), np.ones(20)])
    p = cal.predict(np.array([-1.0, 0.5, 2.0]))
    assert ((p >= 0) & (p <= 1)).all()

def test_holdout_is_single_use(bundle_real, data_dir, tmp_path):
    marker = tmp_path / "holdout_used.json"
    final_evaluate(bundle_real, data_dir, marker)
    with pytest.raises(RuntimeError, match="holdout"):
        final_evaluate(bundle_real, data_dir, marker)

def test_synthetic_bundle_cannot_report(bundle_synth, data_dir, tmp_path):
    with pytest.raises(ValueError, match="synthetic_demo"):
        final_evaluate(bundle_synth, data_dir, tmp_path / "m.json")

def test_bootstrap_detects_better_model(y_strong_weak):
    y, strong, weak = y_strong_weak
    low, high = bootstrap_auc_diff(y, strong, weak, n=500, seed=42)
    assert low > 0 and high > low
```

- [ ] **Step 2:** Testi çalıştır. Beklenen: FAIL.
- [ ] **Step 3:** Kodu yaz. `train_bundle("A", ...)` `results/imbalance_comparison.json` içindeki `winner` algoritma ve koluyla çapraz doğrulama yapar, kalibratörü OOF tahminlerle öğrenir, modeli geliştirme verisinin tamamıyla yeniden eğitir, MLflow'a `data_scope` etiketiyle kaydeder. `final_evaluate` sonuçları `results/final_evaluation.json` dosyasına yazar. Spec başarı ölçütleri 2 ve 3'ü de hesaplar: `bureau` kaydı olmayan grupta G1G2 ve G1 farkının güven aralığı, kalibrasyon öncesi ve sonrası Brier skoru. Kıyas için G1 modeli aynı algoritma ve kolla geliştirme verisinde eğitilir.
- [ ] **Step 4:** Testi çalıştır. Beklenen: PASS. **Commit:** `feat: add calibration final training and single-use holdout`

### Task 7: Karar kuralı, adillik ve raporlar

**Files:**
- Create: `packages/credit_decision/pyproject.toml`, `packages/credit_decision/src/credit_decision/__init__.py`, `config/decision.yaml`, `ml/fairness.py`, `ml/reports/sensitivity.py`, `ml/reports/drift.py`
- Test: `tests/ml/test_decision.py`, `tests/ml/test_fairness.py`

**Interfaces:**
- Produces: `Thresholds(approve: float, reject: float)`, `load_thresholds(path: Path) -> Thresholds`, `thresholds_from(m: float, lgd: float) -> Thresholds`, `assign_band(pd: float, t: Thresholds) -> Literal["approve","manual_review","reject"]`, `fairness_report(y, bands, sensitive: pd.Series) -> pd.DataFrame`

- [ ] **Step 1: Failing test**

```python
def test_thresholds_from_spec():
    t = thresholds_from(0.15, 0.45)
    assert t.reject == pytest.approx(0.25) and t.approve == pytest.approx(0.125)

def test_band_boundaries():
    t = thresholds_from(0.15, 0.45)
    assert assign_band(t.approve - 1e-6, t) == "approve"
    assert assign_band(t.approve, t) == "manual_review"
    assert assign_band(t.reject, t) == "reject"

def test_fairness_has_group_rates():
    r = fairness_report(np.array([0, 1, 0, 1]),
                        np.array(["approve", "reject", "reject", "approve"]),
                        pd.Series(["F", "F", "M", "M"]))
    assert set(r.index) == {"F", "M"}
    assert {"approval_rate", "tpr", "fpr"} <= set(r.columns)
```

- [ ] **Step 2:** Testi çalıştır. Beklenen: FAIL.
- [ ] **Step 3:** Kodu yaz. Adillikte pozitif sınıf `TARGET = 1`, modelin pozitif tahmini `reject` bandıdır, yaş grupları `<30`, `30-50`, `50+` şeklindedir. `sensitivity.py`, `m ∈ {0.10, 0.15, 0.20}` ve `LGD ∈ {0.35, 0.45, 0.60}` için bant oranlarını yazar. `drift.py`, geliştirme verisi ile `application_test` arasında Evidently HTML raporu üretir ve Model B bundle'ını reddeder.
- [ ] **Step 4:** Testi çalıştır. Beklenen: PASS. **Commit:** `feat: add decision bands fairness and reports`

### Task 8: FastAPI skorlama servisi

**Files:**
- Create: `services/scoring/app/main.py`, `services/scoring/app/schemas.py`, `services/scoring/app/bundle.py`, `services/scoring/app/reasons.py`, `services/scoring/requirements.txt`, `config/reason_codes.yaml`
- Test: `tests/scoring/test_api.py`, `tests/scoring/test_reasons.py`, `tests/integration/test_feature_parity.py`

**Interfaces:**
- Consumes: Bundle dosyaları, `build_g1`, `align_frame`
- Produces:
  - `ScoreRequest(model: Literal["A","B"] = "A", application: ApplicationIn, behaviour: dict[str, float | None] = {})`. `ApplicationIn` alanları `RAW_G1_COLUMNS` ile aynıdır, hepsi opsiyoneldir, `extra="forbid"` kullanılır.
  - `ScoreResponse(probability: float, model_version: str, data_scope: str, features: dict[str, float | str | None], reasons: list[Reason])`, `Reason(feature: str, text: str, contribution: float, increases_risk: bool)`
  - `POST /score`, `GET /health`, `load_bundle(path: Path) -> Bundle`, `top_reasons(contributions: dict[str, float], texts: dict[str, str], k=3) -> list[Reason]`, `request_to_frame(req: ScoreRequest, spec: FeatureSpec) -> pd.DataFrame`

- [ ] **Step 1: Failing test.** Küçük bir LightGBM bundle'ı `conftest.py` içinde üretilir.

```python
def test_score_returns_probability_and_reasons(client):
    r = client.post("/score", json=VALID)
    assert r.status_code == 200 and 0 <= r.json()["probability"] <= 1
    assert len(r.json()["reasons"]) == 3

def test_zero_income_still_scores(client):
    r = client.post("/score", json=with_field(VALID, "AMT_INCOME_TOTAL", 0))
    assert r.status_code == 200

def test_missing_values_allowed(client):
    assert client.post("/score", json={"application": {}}).status_code == 200

def test_wrong_type_is_422(client):
    assert client.post("/score", json=with_field(VALID, "AMT_CREDIT", "abc")).status_code == 422

def test_demographic_field_is_422(client):
    assert client.post("/score", json=with_field(VALID, "CODE_GENDER", "F")).status_code == 422

def test_always_three_reasons_even_low_risk():
    out = top_reasons({"a": -3.0, "b": -2.0, "c": -1.0, "d": -4.0}, {}, k=3)
    assert [r.feature for r in out] == ["c", "b", "a"]
    assert not any(r.increases_risk for r in out)

def test_mismatched_bundle_makes_health_503(client_bad_bundle):
    assert client_bad_bundle.get("/health").status_code == 503
    assert client_bad_bundle.post("/score", json=VALID).status_code == 503

def test_training_and_serving_features_match(app_row, spec):
    train = align_frame(build_g1(app_row), spec)
    serve = request_to_frame(ScoreRequest(application=app_row.iloc[0].to_dict()), spec)
    pd.testing.assert_frame_equal(train.reset_index(drop=True), serve, check_dtype=False)
```

- [ ] **Step 2:** Testi çalıştır. Beklenen: FAIL.
- [ ] **Step 3:** Kodu yaz. Bundle başlangıçta yüklenir ve `features.json` ile modelin özellik sayısı karşılaştırılır. SHAP, `shap.TreeExplainer` ile kalibrasyon öncesi çıktıdan hesaplanır. `top_reasons` en büyük işaretli katkıdan küçüğe sıralar. Eşleme tablosunda olmayan özellik için genel metin kullanılır.
- [ ] **Step 4:** `reason_codes.yaml` içine G1 ve G2 özelliklerinin Türkçe metinlerini yaz. Her Model A özelliğinin metni olduğunu doğrulayan test ekle.
- [ ] **Step 5:** Testi çalıştır. Beklenen: PASS. **Commit:** `feat: add scoring service with shap reasons`

### Task 9: Django modelleri, JWT ve değiştirilemez karar kaydı

**Files:**
- Create: `services/api/manage.py`, `services/api/core/settings.py`, `services/api/core/urls.py`, `services/api/core/celery.py`, `services/api/credit/models.py`, `services/api/credit/migrations/`, `services/api/requirements.txt`
- Test: `tests/api/test_models.py`

**Interfaces:**
- Produces: `Application(sk_id_curr, form: JSON, model_key, status: pending|scored, created_at)`, `BehaviourFeatures(sk_id_curr, scope: real|synthetic, features: JSON)` (`sk_id_curr, scope` tekil), `Decision(application: OneToOne, model_version, data_scope, features: JSON, probability, band, reasons: JSON, behaviour_missing: bool, created_at)`, `Batch(status: queued|running|done|partial, model_key, created_at)`, `BatchItem(batch, application, error)`. `POST /api/v1/token/`, `POST /api/v1/token/refresh/`

- [ ] **Step 1: Failing test**

```python
def test_decision_cannot_be_updated(decision):
    decision.band = "approve"
    with pytest.raises(ImmutableRecordError):
        decision.save()

def test_decision_cannot_be_deleted(decision):
    with pytest.raises(ImmutableRecordError):
        decision.delete()

def test_queryset_update_blocked(decision):
    with pytest.raises(ImmutableRecordError):
        Decision.objects.filter(pk=decision.pk).update(band="approve")

def test_token_issued_for_valid_user(client, user):
    r = client.post("/api/v1/token/", {"username": user.username, "password": "test-pass"})
    assert r.status_code == 200 and "access" in r.json()
```

- [ ] **Step 2:** Testi çalıştır. Beklenen: FAIL.
- [ ] **Step 3:** Kodu yaz. Veritabanı `DATABASE_URL` ortam değişkeninden okunur, yerel testlerde bu değişken yoksa SQLite kullanılır, CI ve Compose PostgreSQL kullanır. Admin'de `Decision` salt okunurdur.
- [ ] **Step 4:** Testi çalıştır. Beklenen: PASS. **Commit:** `feat: add django models jwt and immutable decisions`

### Task 10: Başvuru uç noktaları ve skorlama istemcisi

**Files:**
- Create: `services/api/credit/scoring_client.py`, `services/api/credit/services.py`, `services/api/credit/serializers.py`, `services/api/credit/views.py`, `services/api/credit/urls.py`
- Test: `tests/api/test_applications.py`, `tests/integration/test_contract.py`

**Interfaces:**
- Consumes: Görev 7 `assign_band`, Görev 8 `ScoreRequest` ve `ScoreResponse` (yalnızca sözleşme testinde), Görev 9 modelleri
- Produces: `build_score_payload(app: Application, behaviour: dict | None) -> dict`, `score(payload: dict, timeout=5.0) -> dict` (hata ve bozuk yanıtta `ScoringUnavailable` fırlatır), `score_application(app: Application) -> Decision` (kararı varsa `AlreadyDecided` fırlatır), `POST /api/v1/applications`, `GET /api/v1/applications/{id}`, `POST /api/v1/applications/{id}/score`

İstek gövdesi: `{"sk_id_curr": int, "application": {...}, "model": "A"}`. Yanıt alanları: `id, status, probability, band, reasons, model_version, data_scope, behaviour_missing`. Model B seçilirse davranış özellikleri `real` ve `synthetic` kayıtlarının birleşimidir. Eşikler `DECISION_CONFIG` yolundaki dosyadan okunur. Test yardımcısı `fake_scoring` olasılık olarak `0.20` döndürür.

- [ ] **Step 1: Failing test**

```python
def test_api_requires_token(client):
    assert client.get("/api/v1/applications/1").status_code == 401

def test_create_scores_and_writes_decision(auth, fake_scoring):
    r = auth.post("/api/v1/applications", VALID, format="json")
    assert r.status_code == 201 and r.json()["band"] == "manual_review"
    assert Decision.objects.count() == 1

def test_scoring_down_gives_503_and_pending(auth, scoring_down):
    assert auth.post("/api/v1/applications", VALID, format="json").status_code == 503
    assert Application.objects.get().status == "pending"
    assert Decision.objects.count() == 0

def test_malformed_scoring_response_is_503(auth, scoring_garbage):
    assert auth.post("/api/v1/applications", VALID, format="json").status_code == 503
    assert Decision.objects.count() == 0

def test_pending_can_be_retried(auth, pending_app, fake_scoring):
    assert auth.post(f"/api/v1/applications/{pending_app.id}/score").status_code == 200

def test_rescoring_decided_is_409(auth, decided_app, fake_scoring):
    assert auth.post(f"/api/v1/applications/{decided_app.id}/score").status_code == 409
    assert Decision.objects.filter(application=decided_app).count() == 1

def test_unknown_person_scored_and_flagged(auth, fake_scoring):
    r = auth.post("/api/v1/applications", {**VALID, "sk_id_curr": 999999}, format="json")
    assert r.status_code == 201 and r.json()["behaviour_missing"] is True

def test_contract(app_obj):
    ScoreRequest.model_validate(build_score_payload(app_obj, {"g2_prev_count": 2.0}))
```

- [ ] **Step 2:** Testi çalıştır. Beklenen: FAIL.
- [ ] **Step 3:** Kodu yaz. Başvuru önce kaydedilir, skor çağrısı sonra yapılır, böylece hata durumunda `pending` kalır. `Decision` oluşturma transaction içindedir. `drf-spectacular` ile `/api/v1/schema/` eklenir.
- [ ] **Step 4:** Testi çalıştır. Beklenen: PASS. **Commit:** `feat: add application endpoints and scoring client`

### Task 11: Celery ile özellik yükleme ve toplu skorlama

**Files:**
- Create: `services/api/credit/tasks.py`, `services/api/credit/management/commands/load_features.py`, `ml/export_features.py`
- Modify: `services/api/credit/views.py`, `services/api/credit/urls.py`
- Test: `tests/api/test_batches.py`, `tests/ml/test_export.py`

**Interfaces:**
- Consumes: `score_application`, `BehaviourFeatures`, `build_g2`
- Produces: `export_features(data_dir, out: Path, scope: str) -> int` (JSONL, NaN yerine `null`), `load_features_task(path: str, scope: str) -> int`, `score_batch_task(batch_id: int) -> str`, `POST /api/v1/batches` (`202`), `GET /api/v1/batches/{id}`

`GET /batches/{id}` spec'in saydığı uç noktalara ek olarak, toplu işin sonucunu görebilmek için eklenir.

- [ ] **Step 1: Failing test**

```python
def test_load_features_is_idempotent(jsonl_path):
    load_features_task(str(jsonl_path), "real")
    load_features_task(str(jsonl_path), "real")
    assert BehaviourFeatures.objects.count() == 2

def test_batch_returns_202(auth, fake_scoring, celery_eager):
    r = auth.post("/api/v1/batches", {"applications": [VALID, VALID2], "model": "A"}, format="json")
    assert r.status_code == 202

def test_batch_partial_when_scoring_dies_midway(batch, scoring_fails_after_one):
    assert score_batch_task(batch.id) == "partial"
    statuses = sorted(i.application.status for i in batch.batchitem_set.all())
    assert statuses == ["pending", "scored"]

def test_export_writes_null_for_nan(tmp_path, small_data_dir):
    export_features(small_data_dir, tmp_path / "f.jsonl", "real")
    assert "NaN" not in (tmp_path / "f.jsonl").read_text(encoding="utf-8")
```

- [ ] **Step 2:** Testi çalıştır. Beklenen: FAIL.
- [ ] **Step 3:** Kodu yaz. Görevler `sk_id_curr, scope` üzerinden upsert yapar. `load_features` komutu `load_features_task.delay` çağırır.
- [ ] **Step 4:** Testi çalıştır. Beklenen: PASS. **Commit:** `feat: add feature loading and batch scoring`

### Task 12: Sentetik veri, metin hattı ve Model B

**Files:**
- Create: `ml/synthetic/generate.py`, `packages/credit_features/src/credit_features/synthetic.py`, `ml/synthetic/text.py`
- Modify: `ml/train_final.py`, `ml/dataset.py` (`"G1G2SYN"` grubu), `ml/export_features.py` (`synthetic` kapsamı)
- Test: `tests/ml/test_synthetic.py`

**Interfaces:**
- Consumes: `train_bundle`, `export_features`
- Produces: `generate(ids, target: pd.Series | None, out_dir: Path, seed=42) -> None` (`orders`, `reviews`, `invoices` Parquet dosyaları), `build_synthetic(orders, invoices) -> pd.DataFrame` (`syn_order_count`, `syn_order_amount_mean`, `syn_return_share`, `syn_invoice_late_share`), `embed_reviews(reviews, fit_ids, n_components=10) -> pd.DataFrame` (`syn_text_pc_0..9`)

- [ ] **Step 1: Failing test**

```python
def test_same_seed_same_data(tmp_path, ids):
    generate(ids, None, tmp_path / "a", seed=42); generate(ids, None, tmp_path / "b", seed=42)
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "a/orders.parquet"),
                                  pd.read_parquet(tmp_path / "b/orders.parquet"))

def test_reviews_are_turkish(tmp_path, ids):
    generate(ids, None, tmp_path, seed=42)
    text = " ".join(pd.read_parquet(tmp_path / "reviews.parquet")["text"])
    assert any(ch in text for ch in "çğıöşü")

def test_features_are_prefixed(orders, invoices):
    assert all(c.startswith("syn_") for c in build_synthetic(orders, invoices).columns)

def test_model_a_has_no_synthetic_columns(bundle_real):
    spec = json.loads((bundle_real / "features.json").read_text(encoding="utf-8"))
    assert not [c for c in spec["numeric"] + list(spec["categorical"]) if c.startswith("syn_")]

def test_model_b_tagged(bundle_synth):
    meta = json.loads((bundle_synth / "metadata.json").read_text(encoding="utf-8"))
    assert meta["data_scope"] == "synthetic_demo"
```

- [ ] **Step 2:** Testi çalıştır. Beklenen: FAIL.
- [ ] **Step 3:** Kodu yaz. `target` verilmişse geç fatura oranı `TARGET = 1` kişilerde biraz yükseltilir. PCA yalnızca `fit_ids` ile öğrenilir. Metin testi model indirmemek için sahte gömme fonksiyonu kullanır.
- [ ] **Step 4:** Testi çalıştır. Beklenen: PASS. **Commit:** `feat: add synthetic data text features and model b`

### Task 13: Streamlit demo, Docker Compose ve CI

**Files:**
- Create: `demo/app.py`, `demo/api_client.py`, `demo/requirements.txt`, `services/api/credit/management/commands/seed_demo.py`, `services/api/Dockerfile`, `services/scoring/Dockerfile`, `demo/Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`, `README.md`
- Modify: `ml/export_features.py` (20 kişilik `data/demo/applicants.jsonl`, `application_test` verisinden)
- Test: `tests/integration/test_compose.py`, `tests/integration/test_demo.py`

**Interfaces:**
- Consumes: `POST /api/v1/token/`, `POST /api/v1/applications`
- Produces: `ApiClient(base_url).login(user, password)`, `ApiClient.submit(application: dict, model: str) -> dict`, `model_label(data_scope: str) -> str`

- [ ] **Step 1: Failing test**

```python
def test_scoring_has_no_public_port():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    assert "ports" not in compose["services"]["scoring"]

def test_all_services_defined():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    assert {"api", "worker", "scoring", "postgres", "redis", "demo"} <= set(compose["services"])

def test_synthetic_label_is_explicit():
    assert "sentetik" in model_label("synthetic_demo").lower()
    assert "sentetik" not in model_label("real").lower()
```

- [ ] **Step 2:** Testi çalıştır. Beklenen: FAIL.
- [ ] **Step 3:** Kodu yaz. Ekran kişi seçer, formu düzenletir, model seçtirir, olasılığı, bandı, üç nedeni, SHAP katkı grafiğini ve model etiketini gösterir. `seed_demo` kullanıcıyı `.env` değerlerinden oluşturur. CI, PostgreSQL servisiyle `ruff check .` ve `pytest` çalıştırır.
- [ ] **Step 4:** Testi çalıştır. Beklenen: PASS.
- [ ] **Step 5:** `docker compose up --build -d` çalıştır. Beklenen: tüm servisler `healthy` veya `running`, `http://localhost:8501` açılır ve örnek başvuru uçtan uca skorlanır.
- [ ] **Step 6:** README'ye kurulum, sonuç dosyaları, sentetik veri uyarısı ve KVKK m. 11 tartışmasını ekle. **Commit:** `feat: add demo compose ci and documentation`
