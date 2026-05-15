import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # wmic CPU uyarisini engelle

import pickle
import copy
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, classification_report)
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import mutual_info_classif, chi2, f_classif, RFE, SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Klasorleri olustur
for klasor in ["output/pima_diabetes", "output/early_stage_diabetes",
               "output/heart_disease", "output/karsilastirma"]:
    os.makedirs(klasor, exist_ok=True)

print("=" * 65)
print("COKLU VERI SETI KARSILASTIRMASI")
print("=" * 65)


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
    return X.reset_index(drop=True), y.reset_index(drop=True), df

def yukle_erken_evre():
    df = pd.read_csv("data/early_stage_diabetes.csv")
    X = df.drop('class', axis=1).copy()
    y = df['class'].copy()
    le = LabelEncoder()
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = le.fit_transform(X[col].astype(str))
    y = (y == 'Positive').astype(int)
    df_full = X.copy()
    df_full['class'] = y.values
    return X.reset_index(drop=True), y.reset_index(drop=True), df_full

def yukle_kalp():
    df = pd.read_csv("data/heart_disease.csv")
    X = df.drop('target', axis=1).copy()
    y = df['target'].copy()
    X = X.apply(pd.to_numeric, errors='coerce').fillna(X.median())
    y = (y > 0).astype(int)
    df_full = X.copy()
    df_full['target'] = y.values
    return X.reset_index(drop=True), y.reset_index(drop=True), df_full


# ============================================================
# ANA PIPELINE
# ============================================================

def pipeline_calistir(X, y, df_full, veri_adi, params_dosya, output_klasor, model_pkl):

    print(f"\n{'='*65}")
    print(f"  VERI SETI : {veri_adi}")
    print(f"  Boyut     : {X.shape[0]} satir x {X.shape[1]} ozellik")
    print(f"  Sinif     : {dict(y.value_counts().sort_index())}")
    print(f"{'='*65}")

    # ---- [1] Korelasyon Matrisi ------------------------------------
    plt.figure(figsize=(9, 7))
    sns.heatmap(df_full.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title(f"Korelasyon Matrisi - {veri_adi}")
    plt.tight_layout()
    plt.savefig(f"{output_klasor}/korelasyon.png", dpi=150)
    plt.close()
    print(f"\n  [1] {output_klasor}/korelasyon.png kaydedildi")

    # ---- [2] Train / Test Bolme (10 iterasyon) ---------------------
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    splits = list(rskf.split(X, y))
    train_idx, test_idx = splits[0]
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test  = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test  = y.iloc[test_idx].reset_index(drop=True)
    print(f"  [2] Train/Test: {len(X_train)} egitim | {len(X_test)} test (10 iterasyon)")

    # ---- [3] Sinif Dengeleme (Downsampling) ------------------------
    hedef_oran = 2.0
    df_tr = X_train.copy()
    df_tr['__y__'] = y_train.values
    counts = df_tr['__y__'].value_counts()
    buyuk, kucuk = counts.idxmax(), counts.idxmin()
    oran = counts[buyuk] / counts[kucuk]
    if oran > hedef_oran:
        hedef = int(counts[kucuk] * hedef_oran)
        df_tr = pd.concat([
            df_tr[df_tr['__y__'] == buyuk].sample(hedef, random_state=42),
            df_tr[df_tr['__y__'] == kucuk]
        ]).sample(frac=1, random_state=42)
        print(f"  [3] Downsampling: {oran:.2f}:1 -> {hedef_oran:.1f}:1")
    else:
        print(f"  [3] Downsampling gerekmedi: oran {oran:.2f}:1")
    y_train = df_tr['__y__']
    X_train = df_tr.drop('__y__', axis=1)

    # ---- [4] Feature Selection - 5 Yontem -------------------------
    print(f"  [4] Feature Selection (5 yontem)...")
    mi_scores      = mutual_info_classif(X_train, y_train, random_state=42)
    X_mm           = MinMaxScaler().fit_transform(X_train)
    chi2_scores, _ = chi2(X_mm, y_train)
    f_scores, _    = f_classif(X_train, y_train)

    n_sec = min(5, X_train.shape[1])
    rfe_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rfe = RFE(estimator=rfe_model, n_features_to_select=n_sec)
    rfe.fit(X_train, y_train)

    sfm_model = RandomForestClassifier(n_estimators=100, random_state=42)
    sfm_model.fit(X_train, y_train)
    sfm = SelectFromModel(sfm_model, prefit=True)

    fs_df = pd.DataFrame({
        'Feature':      X_train.columns,
        'MI_Score':     mi_scores,
        'Chi2_Score':   chi2_scores,
        'F_Score':      f_scores,
        'RFE_Rank':     rfe.ranking_,
        'SFM_Selected': sfm.get_support().astype(int),
    }).sort_values('MI_Score', ascending=False)

    print(fs_df.to_string(index=False))

    # Feature selection grafigi
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col, title, color in zip(
        axes,
        ['MI_Score', 'Chi2_Score', 'F_Score'],
        ['Mutual Information', 'Chi-Squared', 'ANOVA F-test'],
        ['#3498db', '#e74c3c', '#2ecc71']
    ):
        sdf = fs_df.sort_values(col, ascending=True)
        ax.barh(sdf['Feature'], sdf[col], color=color, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel('Skor')
    plt.suptitle(f"Feature Selection Karsilastirmasi - {veri_adi}", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{output_klasor}/feature_selection_karsilastirma.png", dpi=150)
    plt.close()
    print(f"       {output_klasor}/feature_selection_karsilastirma.png kaydedildi")

    top_5 = fs_df.head(n_sec)['Feature'].tolist()
    print(f"       En onemli {n_sec} ozellik (MI): {top_5}")

    X_train_top = X_train[top_5]
    X_test_top  = X_test[top_5]

    # ---- [5] Optuna Hiperparametre Optimizasyonu -------------------
    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

    if os.path.exists(params_dosya):
        with open(params_dosya, "r") as f:
            saved = json.load(f)
        best_rf  = saved["rf"]
        best_xgb = saved["xgb"]
        print(f"\n  [5] Optuna cache'den yuklendi: {params_dosya}")
    else:
        print(f"\n  [5] Optuna optimizasyonu basliyor (50 deneme)...")

        def opt_rf(trial):
            p = {
                'n_estimators':      trial.suggest_int('n_estimators', 100, 500),
                'max_depth':         trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features':      trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            }
            return cross_val_score(RandomForestClassifier(**p, random_state=42),
                                   X_train_top, y_train, cv=skf, scoring='roc_auc').mean()

        def opt_xgb(trial):
            p = {
                'n_estimators':     trial.suggest_int('n_estimators', 100, 500),
                'max_depth':        trial.suggest_int('max_depth', 3, 10),
                'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            }
            return cross_val_score(XGBClassifier(**p, random_state=42,
                                                 eval_metric='logloss', verbosity=0),
                                   X_train_top, y_train, cv=skf, scoring='roc_auc').mean()

        s_rf = optuna.create_study(direction='maximize')
        s_rf.optimize(opt_rf, n_trials=50)
        best_rf = s_rf.best_params

        s_xgb = optuna.create_study(direction='maximize')
        s_xgb.optimize(opt_xgb, n_trials=50)
        best_xgb = s_xgb.best_params

        with open(params_dosya, "w") as f:
            json.dump({"rf": best_rf, "xgb": best_xgb}, f, indent=2)
        print(f"       Parametreler kaydedildi: {params_dosya}")

    # ---- [6] 6 Algoritma x 2 Ozellik Seti = 12 Kombinasyon --------
    def modeller_olustur():
        return {
            "Logistic Regression": Pipeline([
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(max_iter=1000, random_state=42))
            ]),
            "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
            "KNN": Pipeline([
                ('scaler', StandardScaler()),
                ('model', KNeighborsClassifier(n_neighbors=5))
            ]),
            "SVM": Pipeline([
                ('scaler', StandardScaler()),
                ('model', SVC(kernel='rbf', probability=True, random_state=42))
            ]),
            "Random Forest": RandomForestClassifier(**best_rf, random_state=42),
            "XGBoost": XGBClassifier(**best_xgb, random_state=42,
                                     eval_metric='logloss', verbosity=0),
            "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "Naive Bayes": Pipeline([
                ('scaler', StandardScaler()),
                ('model', GaussianNB())
            ]),
        }

    print(f"\n  [6] 9 Algoritma x 2 Ozellik Seti = 18 Kombinasyon")
    print(f"      {'Kombinasyon':<32} {'Accuracy':>10} {'AUC':>10} {'CV AUC':>10}")
    print("      " + "-"*60)

    tum_kombinasyonlar = {}
    roc_modeller = []  # ROC grafigi icin

    for isim, model in modeller_olustur().items():
        for etiket, Xtr, Xte in [("Full", X_train, X_test),
                                  ("Top5", X_train_top, X_test_top)]:
            m = copy.deepcopy(model)
            cv_auc = cross_val_score(m, Xtr, y_train, cv=skf, scoring='roc_auc').mean()
            m.fit(Xtr, y_train)
            y_pred  = m.predict(Xte)
            y_proba = m.predict_proba(Xte)[:, 1]
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            ad  = f"{isim} - {etiket}"
            tum_kombinasyonlar[ad] = {
                'model': m, 'features': Xte.columns.tolist(),
                'acc': acc, 'auc': auc, 'cv_auc': cv_auc,
                'y_pred': y_pred, 'Xte': Xte
            }
            print(f"      {ad:<32} {acc:>10.4f} {auc:>10.4f} {cv_auc:>10.4f}")
            if etiket == "Full":
                roc_modeller.append((isim, m, Xte, y_proba))

    # En iyi model (CV AUC oncelikli)
    en_iyi_ad = max(tum_kombinasyonlar,
                    key=lambda x: (tum_kombinasyonlar[x]['cv_auc'],
                                   tum_kombinasyonlar[x]['auc']))
    en_iyi = tum_kombinasyonlar[en_iyi_ad]
    print(f"\n      Kazanan: {en_iyi_ad}")
    print(f"      CV AUC: {en_iyi['cv_auc']:.4f} | AUC: {en_iyi['auc']:.4f} | Accuracy: {en_iyi['acc']:.4f}")

    # ---- [7] Esik Optimizasyonu ------------------------------------
    final_features = en_iyi['features']
    Xtr_final = X_train[final_features]
    Xte_final = X_test[final_features]

    m_template = clone(en_iyi['model'])
    Xsub, Xval, ysub, yval = train_test_split(
        Xtr_final, y_train, test_size=0.2, random_state=42, stratify=y_train)
    m_template.fit(Xsub, ysub)
    yproba_val = m_template.predict_proba(Xval)[:, 1]
    _, _, thresholds = roc_curve(yval, yproba_val)
    f1s = [f1_score(yval, (yproba_val >= t).astype(int)) for t in thresholds]
    optimal_threshold = thresholds[np.argmax(f1s)]

    final_model = clone(en_iyi['model'])
    final_model.fit(Xtr_final, y_train)
    y_pred_final = final_model.predict(Xte_final)
    yproba_final = final_model.predict_proba(Xte_final)[:, 1]
    y_pred_opt   = (yproba_final >= optimal_threshold).astype(int)

    cm_default = confusion_matrix(y_test, y_pred_final)
    cm_opt     = confusion_matrix(y_test, y_pred_opt)
    print(f"\n  [7] Esik Optimizasyonu: {optimal_threshold:.2f}")
    print(f"      Varsayilan (0.50): kacirilan = {cm_default[1][0]} / {sum(y_test==1)}")
    print(f"      Optimize ({optimal_threshold:.2f}) : kacirilan = {cm_opt[1][0]} / {sum(y_test==1)}")

    # ---- [8] Confusion Matrix --------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm_default, display_labels=["Negatif (0)", "Pozitif (1)"]).plot(
        ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix\n{veri_adi}", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{output_klasor}/confusion_matrix.png", dpi=150)
    plt.close()
    print(f"  [8] {output_klasor}/confusion_matrix.png kaydedildi")

    # ---- [9] ROC Egrisi --------------------------------------------
    plt.figure(figsize=(8, 6))
    for isim, m, Xte, yproba in roc_modeller:
        auc = roc_auc_score(y_test, yproba)
        fpr, tpr, _ = roc_curve(y_test, yproba)
        plt.plot(fpr, tpr, label=f"{isim} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Rastgele (AUC=0.500)')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Egrisi - {veri_adi}")
    plt.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{output_klasor}/roc_curve.png", dpi=150)
    plt.close()
    print(f"  [9] {output_klasor}/roc_curve.png kaydedildi")

    # ---- [10] Kalibrasyon Egrisi -----------------------------------
    plt.figure(figsize=(8, 6))
    for isim, m, Xte, yproba in roc_modeller:
        try:
            prob_true, prob_pred = calibration_curve(y_test, yproba, n_bins=5)
            plt.plot(prob_pred, prob_true, marker='o', label=isim)
        except Exception:
            pass
    plt.plot([0, 1], [0, 1], 'k--', label='Mukemmel Kalibrasyon')
    plt.xlabel("Tahmin Edilen Olasilik")
    plt.ylabel("Gercek Olasilik")
    plt.title(f"Kalibrasyon Egrisi\n{veri_adi}")
    plt.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{output_klasor}/kalibrasyon.png", dpi=150)
    plt.close()
    print(f"  [10] {output_klasor}/kalibrasyon.png kaydedildi")

    # ---- [11] Feature Importance -----------------------------------
    # Final model agac tabanli degilse (SVM, LR, KNN) RF ile hesapla
    if hasattr(final_model, 'feature_importances_'):
        fi_model = final_model
        fi_features = final_features
        fi_baslik = "Feature Importance (Final Model)"
    else:
        fi_model = RandomForestClassifier(n_estimators=100, random_state=42)
        fi_model.fit(X_train[top_5], y_train)
        fi_features = top_5
        fi_baslik = "Feature Importance (Random Forest - Referans)"

    fi = pd.DataFrame({
        'Feature':    fi_features,
        'Importance': fi_model.feature_importances_
    }).sort_values('Importance', ascending=True)
    plt.figure(figsize=(7, 5))
    plt.barh(fi['Feature'], fi['Importance'], color='#9b59b6', alpha=0.85)
    plt.xlabel("Onem Skoru")
    plt.title(f"{fi_baslik}\n{veri_adi}")
    plt.tight_layout()
    plt.savefig(f"{output_klasor}/feature_importance.png", dpi=150)
    plt.close()
    print(f"  [10] {output_klasor}/feature_importance.png kaydedildi")

    # ---- [11] Model Kaydet -----------------------------------------
    with open(model_pkl, "wb") as f:
        pickle.dump({
            'model':     final_model,
            'features':  final_features,
            'threshold': optimal_threshold
        }, f)
    print(f"  [11] {model_pkl} kaydedildi")

    return {
        'Veri Seti':    veri_adi,
        'Satirlar':     X.shape[0],
        'Ozellikler':   X.shape[1],
        'En iyi Model': en_iyi_ad,
        'Accuracy':     round(en_iyi['acc'], 4),
        'AUC':          round(en_iyi['auc'], 4),
        'CV AUC':       round(en_iyi['cv_auc'], 4),
        'Threshold':    round(float(optimal_threshold), 2),
    }


# ============================================================
# ANA CALISMA
# ============================================================

veri_setleri = [
    ("Pima Indians Diabetes",     yukle_pima,       "cache/params_pima.json",
     "output/pima_diabetes",      "model_pima.pkl"),
    ("Early Stage Diabetes Risk", yukle_erken_evre, "cache/params_erken.json",
     "output/early_stage_diabetes","model_erken.pkl"),
    ("Heart Disease (Cleveland)", yukle_kalp,       "cache/params_kalp.json",
     "output/heart_disease",      "model_kalp.pkl"),
]

tum_sonuclar = []

for veri_adi, yukle_fn, params_dosya, output_klasor, model_pkl in veri_setleri:
    print(f"\nVeri seti yukleniyor: {veri_adi} ...")
    X, y, df_full = yukle_fn()
    sonuc = pipeline_calistir(X, y, df_full, veri_adi, params_dosya, output_klasor, model_pkl)
    tum_sonuclar.append(sonuc)


# ============================================================
# GENEL KARSILASTIRMA
# ============================================================

print("\n\n" + "=" * 65)
print("GENEL KARSILASTIRMA")
print("=" * 65)
df_karsi = pd.DataFrame(tum_sonuclar)
print(df_karsi.to_string(index=False))

# Bar chart
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for i, (metrik, renk) in enumerate(zip(['Accuracy', 'AUC', 'CV AUC'],
                                        ['#3498db', '#e74c3c', '#2ecc71'])):
    bars = axes[i].bar(df_karsi['Veri Seti'], df_karsi[metrik], color=renk, alpha=0.85)
    axes[i].set_title(metrik, fontsize=13, fontweight='bold')
    axes[i].set_ylim(0, 1.1)
    axes[i].tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, df_karsi[metrik]):
        axes[i].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=10)
plt.suptitle("3 Veri Seti Karsilastirmasi", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("output/karsilastirma/karsilastirma_grafik.png", dpi=150)
plt.close()

# Radar chart
kategoriler = ['Accuracy', 'AUC', 'CV AUC']
N = len(kategoriler)
angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
for idx, row in df_karsi.iterrows():
    values = [row['Accuracy'], row['AUC'], row['CV AUC']] + [row['Accuracy']]
    color = ['#3498db', '#e74c3c', '#2ecc71'][idx]
    ax.plot(angles, values, 'o-', linewidth=2, color=color, label=row['Veri Seti'])
    ax.fill(angles, values, alpha=0.1, color=color)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(kategoriler, fontsize=11)
ax.set_ylim(0, 1)
ax.set_title("Radar Karsilastirma", fontsize=13, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
plt.tight_layout()
plt.savefig("output/karsilastirma/radar_karsilastirma.png", dpi=150)
plt.close()

print("\noutput/karsilastirma/ grafikleri kaydedildi")
print("\n" + "=" * 65)
print("TAMAMLANDI")
print("=" * 65)
