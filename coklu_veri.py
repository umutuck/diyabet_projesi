import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier
import optuna
import json, os

optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=" * 65)
print("COKLU VERI SETI KARSILASTIRMASI")
print("=" * 65)

# ============================================================
# YARDIMCI FONKSIYONLAR
# ============================================================

def sinif_dengele(X, y, hedef_oran=2.0):
    """Cok sayida buyuk sinifi downsample eder."""
    df = X.copy()
    df['__target__'] = y.values
    counts = df['__target__'].value_counts()
    buyuk = counts.idxmax()
    kucuk = counts.idxmin()
    oran = counts[buyuk] / counts[kucuk]
    if oran > hedef_oran:
        hedef = int(counts[kucuk] * hedef_oran)
        df_buyuk = df[df['__target__'] == buyuk].sample(hedef, random_state=42)
        df_kucuk = df[df['__target__'] == kucuk]
        df = pd.concat([df_buyuk, df_kucuk]).sample(frac=1, random_state=42)
    y_out = df['__target__']
    X_out = df.drop('__target__', axis=1)
    return X_out, y_out

def modelleri_olustur(best_rf, best_xgb):
    return [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("Decision Tree",       DecisionTreeClassifier(random_state=42)),
        ("KNN",                 KNeighborsClassifier()),
        ("Random Forest",       RandomForestClassifier(**best_rf, random_state=42)),
        ("XGBoost",             XGBClassifier(**best_xgb, random_state=42,
                                              eval_metric='logloss', verbosity=0)),
    ]

def pipeline_calistir(X, y, veri_adi, params_dosya):
    """Tek bir veri seti icin tam pipeline."""
    print(f"\n{'='*65}")
    print(f"  VERI SETI: {veri_adi}")
    print(f"  Boyut: {X.shape[0]} satirx{X.shape[1]} ozellik")
    print(f"  Sinif dagilimi: {dict(y.value_counts().sort_index())}")
    print(f"{'='*65}")

    # ---- Train / Test Bolme ----------------------------------------
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    splits = list(rskf.split(X, y))
    train_idx, test_idx = splits[0]
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test  = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test  = y.iloc[test_idx].reset_index(drop=True)

    print(f"\n  Egitim: {len(X_train)} | Test: {len(X_test)}")

    # ---- Sinif Dengeleme -------------------------------------------
    X_train, y_train = sinif_dengele(X_train, y_train, hedef_oran=2.0)

    # ---- Feature Selection (Mutual Information) --------------------
    mi = mutual_info_classif(X_train, y_train, random_state=42)
    mi_df = pd.DataFrame({'Feature': X.columns, 'MI': mi}).sort_values('MI', ascending=False)
    n_ozellik = min(5, len(X.columns))
    top_features = mi_df.head(n_ozellik)['Feature'].tolist()
    print(f"  En onemli {n_ozellik} ozellik (MI): {top_features}")

    X_train_top = X_train[top_features]
    X_test_top  = X_test[top_features]

    # ---- Optuna Hiper-Parametre Optimizasyonu (Cache) --------------
    skf_opt = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

    if os.path.exists(params_dosya):
        with open(params_dosya, "r") as f:
            saved = json.load(f)
        best_rf_params  = saved["rf"]
        best_xgb_params = saved["xgb"]
        print(f"  Optuna parametreleri cache'den yuklendi: {params_dosya}")
    else:
        print("  Optuna optimizasyonu basliyor (50 deneme)...")

        def opt_rf(trial):
            p = {
                'n_estimators':      trial.suggest_int('n_estimators', 100, 500),
                'max_depth':         trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features':      trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            }
            m = RandomForestClassifier(**p, random_state=42)
            return cross_val_score(m, X_train_top, y_train, cv=skf_opt, scoring='roc_auc').mean()

        def opt_xgb(trial):
            p = {
                'n_estimators':     trial.suggest_int('n_estimators', 100, 500),
                'max_depth':        trial.suggest_int('max_depth', 3, 10),
                'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            }
            m = XGBClassifier(**p, random_state=42, eval_metric='logloss', verbosity=0)
            return cross_val_score(m, X_train_top, y_train, cv=skf_opt, scoring='roc_auc').mean()

        s_rf = optuna.create_study(direction='maximize')
        s_rf.optimize(opt_rf, n_trials=50)
        best_rf_params = s_rf.best_params

        s_xgb = optuna.create_study(direction='maximize')
        s_xgb.optimize(opt_xgb, n_trials=50)
        best_xgb_params = s_xgb.best_params

        with open(params_dosya, "w") as f:
            json.dump({"rf": best_rf_params, "xgb": best_xgb_params}, f, indent=2)
        print(f"  Parametreler kaydedildi: {params_dosya}")

    # ---- Model Egitimi ve Degerlendirme ----------------------------
    modeller = modelleri_olustur(best_rf_params, best_xgb_params)
    skf_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

    sonuclar = []
    en_iyi_auc = -1
    en_iyi_model = None
    en_iyi_ad = ""
    en_iyi_ozellik = "full"

    for ad, model in modeller:
        for ozellik_seti, Xtr, Xte in [("full", X_train, X_test),
                                        ("top5", X_train_top, X_test_top)]:
            model.fit(Xtr, y_train)
            y_pred  = model.predict(Xte)
            y_proba = model.predict_proba(Xte)[:, 1]
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            cv_auc = cross_val_score(model, Xtr, y_train,
                                     cv=skf_cv, scoring='roc_auc').mean()
            sonuclar.append({
                'Model': ad,
                'Ozellik': ozellik_seti,
                'Accuracy': acc,
                'AUC': auc,
                'CV_AUC': cv_auc,
            })
            if auc > en_iyi_auc:
                en_iyi_auc = auc
                en_iyi_model = model
                en_iyi_ad = ad
                en_iyi_ozellik = ozellik_seti
                en_iyi_X_test = Xte
                en_iyi_y_test = y_test

    # ---- Sonuc Tablosu --------------------------------------------
    df_sn = pd.DataFrame(sonuclar).sort_values('AUC', ascending=False)
    print(f"\n  {'Model':<22} {'Ozellik':<8} {'Accuracy':>9} {'AUC':>7} {'CV AUC':>8}")
    print("  " + "-"*57)
    for _, r in df_sn.iterrows():
        print(f"  {r['Model']:<22} {r['Ozellik']:<8} {r['Accuracy']:>9.4f} {r['AUC']:>7.4f} {r['CV_AUC']:>8.4f}")

    best_row = df_sn.iloc[0]
    print(f"\n  >> En iyi: {en_iyi_ad} ({en_iyi_ozellik})")
    print(f"     Accuracy: {best_row['Accuracy']:.4f} | AUC: {best_row['AUC']:.4f}")

    return {
        'Veri Seti':  veri_adi,
        'Satirlar':   X.shape[0],
        'Ozellikler': X.shape[1],
        'En iyi Model': f"{en_iyi_ad} ({en_iyi_ozellik})",
        'Accuracy':   round(best_row['Accuracy'], 4),
        'AUC':        round(best_row['AUC'], 4),
        'CV AUC':     round(best_row['CV_AUC'], 4),
    }


# ============================================================
# VERİ SETLERİ
# ============================================================

def yukle_pima():
    df = pd.read_csv("data/diabetes.csv")
    sifir_kolonlar = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in sifir_kolonlar:
        df[col] = df[col].replace(0, df[col].median())
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    return X, y

def yukle_erken_evre():
    df = pd.read_csv("data/early_stage_diabetes.csv")
    X = df.drop('class', axis=1).copy()
    y = df['class'].copy()
    le = LabelEncoder()
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = le.fit_transform(X[col].astype(str))
    y = (y == 'Positive').astype(int)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    return X, y

def yukle_kalp():
    df = pd.read_csv("data/heart_disease.csv")
    X = df.drop('target', axis=1).copy()
    y = df['target'].copy()
    # Eksik degerler
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.median())
    # Binary: 0=saglikli, 1+=hasta
    y = (y > 0).astype(int)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    return X, y


# ============================================================
# ANA CALISMA
# ============================================================

veri_setleri = [
    ("Pima Indians Diabetes",      yukle_pima,       "cache/params_pima.json"),
    ("Early Stage Diabetes Risk",  yukle_erken_evre, "cache/params_erken.json"),
    ("Heart Disease (Cleveland)",  yukle_kalp,       "cache/params_kalp.json"),
]

tum_sonuclar = []

for veri_adi, yukle_fn, params_dosya in veri_setleri:
    print(f"\nVeri seti yukleniyor: {veri_adi} ...")
    X, y = yukle_fn()
    sonuc = pipeline_calistir(X, y, veri_adi, params_dosya)
    tum_sonuclar.append(sonuc)


# ============================================================
# GENEL KARSILASTIRMA TABLOSU
# ============================================================

print("\n\n" + "=" * 65)
print("GENEL KARSILASTIRMA")
print("=" * 65)
df_karsi = pd.DataFrame(tum_sonuclar)
print(df_karsi.to_string(index=False))

# ---- Karsilastirma Grafigi ------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metrikler = ['Accuracy', 'AUC', 'CV AUC']
renkler   = ['#3498db', '#e74c3c', '#2ecc71']

for i, metrik in enumerate(metrikler):
    bars = axes[i].bar(df_karsi['Veri Seti'], df_karsi[metrik], color=renkler[i], alpha=0.85)
    axes[i].set_title(metrik, fontsize=13, fontweight='bold')
    axes[i].set_ylim(0, 1.05)
    axes[i].set_ylabel(metrik)
    axes[i].tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, df_karsi[metrik]):
        axes[i].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=10)

plt.suptitle("3 Veri Seti Karsilastirmasi", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("output/coklu_veri_karsilastirma.png", dpi=150)
plt.close()

print("\noutput/coklu_veri_karsilastirma.png kaydedildi")
print("\n" + "=" * 65)
print("TAMAMLANDI")
print("=" * 65)
