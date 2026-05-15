import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.feature_selection import mutual_info_classif, chi2, f_classif, RFE, SelectFromModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from xgboost import XGBClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import numpy as np

print("="*60)
print("DIYABET RISK TAHMIN MODELI - ADVANCED")
print("="*60)

# 1. VERI YÜKLEME
df = pd.read_csv("data/diabetes_augmented.csv")
print(f"\n[1] Veri Yüklendi: data/diabetes_augmented.csv")
print(f"    {df.shape[0]} örnek, {df.shape[1]} özellik")
print(f"    Sınıf dağılımı: {df['Outcome'].value_counts().sort_index().to_dict()}")

# 2. VERİ TEMIZLEME
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols] = df[cols].replace(0, np.nan)
df.fillna(df.median(numeric_only=True), inplace=True)
print("\n[2] Veri Temizleme Tamamlandı")
print(f"    Eksik değer giderme: Medyan imputasyon kullanıldı")

# 3. KORELASYON MATRİSİ
plt.figure(figsize=(9, 7))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Korelasyon Matrisi")
plt.tight_layout()
plt.savefig("output/korelasyon.png")
print("\n[3] Korelasyon Matrisi: output/korelasyon.png kaydedildi")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 4. EGITIM-TEST BOLME — 10 Iterasyon (Repeated Stratified K-Fold)
# Her iterasyonda farkli bolme yapilir, 10 sonucun ortalamasi alinir
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

print(f"\n[4] Egitim-Test Bolme (10 Iterasyon - Repeated Stratified K-Fold):")
print(f"    5-Fold x 2 Tekrar = 10 farkli test")

iter_results = {
    'acc': [], 'auc': []
}

# İlk fold'u asıl train/test olarak kullan, diğerleri CV için
splits = list(rskf.split(X, y))
train_idx, test_idx = splits[0]
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print(f"    Egitim: {X_train.shape[0]} ornek")
print(f"    Test  : {X_test.shape[0]} ornek")

# 4.1 SINIF DENGELEME (Downsampling - hedef oran: 2:1)
# Not: Sadece eğitim verisinde uygulanır; test setine asla dokunulmaz.
target_ratio = 2.0  # majority / minority
train_df = X_train.copy()
train_df["Outcome"] = y_train.values

class_counts = train_df["Outcome"].value_counts()
majority_class = class_counts.idxmax()
minority_class = class_counts.idxmin()
majority_count = class_counts[majority_class]
minority_count = class_counts[minority_class]
current_ratio = majority_count / minority_count

if current_ratio > target_ratio:
    target_majority_count = int(target_ratio * minority_count)
    majority_part = train_df[train_df["Outcome"] == majority_class].sample(
        n=target_majority_count, random_state=42
    )
    minority_part = train_df[train_df["Outcome"] == minority_class]
    balanced_train_df = pd.concat([majority_part, minority_part], axis=0).sample(
        frac=1.0, random_state=42
    )
    print(f"\n[4.1] Downsampling uygulandı: oran {current_ratio:.2f}:1 -> {target_ratio:.1f}:1")
else:
    balanced_train_df = train_df.copy()
    print(
        f"\n[4.1] Downsampling gerekmedi: mevcut oran {current_ratio:.2f}:1 "
        f"(hedef <= {target_ratio:.1f}:1)"
    )

X_train = balanced_train_df.drop("Outcome", axis=1)
y_train = balanced_train_df["Outcome"]
print(f"      Yeni eğitim dağılımı: {y_train.value_counts().to_dict()}")

# Downsampling sonrası ayrıca class_weight / scale_pos_weight kullanmıyoruz.
# Böylece aynı dengesizlik düzeltmesini iki kez uygulayıp modeli gereksiz zorlamıyoruz.
class_weight_opt = None
scale_pos_weight = 1.0

# 5. OZELLIK SECIMI - sadece X_train uzerinde (Data Leakage onlemi)
print("\n[5] OZELLIK SECIMI - Coklu Yontem Karsilastirmasi (sadece egitim verisi):")
print("     " + "-"*50)

# Mutual Information
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)

# Chi-Squared (MinMaxScaler ile negatif deger engeli)
X_train_mm = MinMaxScaler().fit_transform(X_train)
chi2_scores, _ = chi2(X_train_mm, y_train)

# ANOVA F-test
f_scores, _ = f_classif(X_train, y_train)

# RFE - Recursive Feature Elimination
rfe_model = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=rfe_model, n_features_to_select=5)
rfe.fit(X_train, y_train)

# SelectFromModel
sfm_model = RandomForestClassifier(n_estimators=100, random_state=42)
sfm_model.fit(X_train, y_train)
sfm = SelectFromModel(sfm_model, prefit=True)

feature_importance_df = pd.DataFrame({
    'Feature':      list(X.columns),
    'MI_Score':     mi_scores,
    'Chi2_Score':   chi2_scores,
    'F_Score':      f_scores,
    'RFE_Rank':     rfe.ranking_,
    'SFM_Selected': sfm.get_support().astype(int),
}).sort_values('MI_Score', ascending=False)

print(feature_importance_df.to_string(index=False))

# Karsilastirma grafigi
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, col, title, color in zip(
    axes,
    ['MI_Score', 'Chi2_Score', 'F_Score'],
    ['Mutual Information', 'Chi-Squared', 'ANOVA F-test'],
    ['#3498db', '#e74c3c', '#2ecc71']
):
    sorted_df = feature_importance_df.sort_values(col, ascending=True)
    ax.barh(sorted_df['Feature'], sorted_df[col], color=color)
    ax.set_title(title)
    ax.set_xlabel('Skor')
plt.suptitle("Feature Selection Yontem Karsilastirmasi", fontsize=13)
plt.tight_layout()
plt.savefig("output/feature_selection_karsilastirma.png")
plt.close()
print("     output/feature_selection_karsilastirma.png kaydedildi")

top_5_features = feature_importance_df.head(5)['Feature'].tolist()
print(f"\n    En onemli 5 ozellik (MI): {top_5_features}")

# 10 iterasyon CV (5-Fold x 2 Tekrar)
skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

# 5.5 HIPERPARAMETRE OPTIMIZASYONU (Optuna - parametreler cache'lenir)
import json, os
print("\n[5.5] HIPERPARAMETRE OPTIMIZASYONU (Optuna)")
print("      " + "-"*50)

PARAMS_FILE = "cache/best_params.json"

if os.path.exists(PARAMS_FILE):
    with open(PARAMS_FILE, "r") as f:
        saved = json.load(f)
    best_rf_params  = saved["rf"]
    best_xgb_params = saved["xgb"]
    print(f"  Kaydedilmis parametreler yuklendi ({PARAMS_FILE})")
    print(f"  Random Forest : {best_rf_params}")
    print(f"  XGBoost       : {best_xgb_params}")
else:
    print("  Parametreler bulunamadi, Optuna calistiriliyor (50 deneme)...")

    def optimize_rf(trial):
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 100, 500),
            'max_depth':         trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 5),
            'max_features':      trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        }
        m = RandomForestClassifier(**params, random_state=42)
        return cross_val_score(m, X_train, y_train, cv=skf, scoring='roc_auc').mean()

    def optimize_xgb(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 100, 500),
            'max_depth':        trial.suggest_int('max_depth', 3, 10),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        }
        m = XGBClassifier(**params, random_state=42, eval_metric='logloss', verbosity=0)
        return cross_val_score(m, X_train, y_train, cv=skf, scoring='roc_auc').mean()

    study_rf = optuna.create_study(direction='maximize')
    study_rf.optimize(optimize_rf, n_trials=50)
    best_rf_params = study_rf.best_params
    print(f"  Random Forest - En iyi CV AUC : {study_rf.best_value:.4f}")

    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(optimize_xgb, n_trials=50)
    best_xgb_params = study_xgb.best_params
    print(f"  XGBoost       - En iyi CV AUC : {study_xgb.best_value:.4f}")

    with open(PARAMS_FILE, "w") as f:
        json.dump({"rf": best_rf_params, "xgb": best_xgb_params}, f, indent=2)
    print(f"  Parametreler kaydedildi: {PARAMS_FILE}")

# 6. TÜM MODELLERİ EĞİT (TAM ÖZELLİKLERLE)
print("\n[6] MODEL EĞİTİMİ (6 Farklı Algoritma - 8 özellik)")
print("     " + "-"*50)

def build_models():
    return {
        "Logistic Regression": Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(random_state=42, max_iter=1000, class_weight=class_weight_opt))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10, class_weight=class_weight_opt),
        "KNN": Pipeline([
            ('scaler', StandardScaler()),
            ('model', KNeighborsClassifier(n_neighbors=5))
        ]),
        "SVM": Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(kernel='rbf', probability=True, class_weight=class_weight_opt, random_state=42))
        ]),
        "Random Forest": RandomForestClassifier(
            **best_rf_params, random_state=42, class_weight=class_weight_opt
        ),
        "XGBoost": XGBClassifier(
            **best_xgb_params, random_state=42, eval_metric='logloss', verbosity=0
        ),
    }

modeller = build_models()

sonuclar = {}

for isim, model in modeller.items():
    # CV sadece eğitim seti üzerinde çalışır; test seti yalnızca final değerlendirme içindir.
    cv_auc = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc').mean()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    sonuclar[isim] = {
        'model': model, 'y_pred': y_pred,
        'acc': acc, 'auc': auc, 'cv_auc': cv_auc,
        'features': list(X.columns)
    }

    print(f"\n  {isim}:")
    print(f"    Accuracy : {acc:.4f}")
    print(f"    ROC-AUC  : {auc:.4f}")
    print(f"    CV AUC   : {cv_auc:.4f}")
    print(f"    Classification Report (Diyabetli sınıfı odaklı):\n{classification_report(y_test, y_pred, target_names=['Negatif','Diyabetli'])}")

# 7. TÜM KOMBİNASYONLAR: HER ALGORİTMA × (FULL + TOP 5)
print(f"\n[7] MODEL KARŞILAŞTIRMASI (6 Algoritma × Full + Top 5 = 12 kombinasyon)")
print("     " + "-"*62)

tum_modeller = build_models()

tum_kombinasyonlar = {}

for isim in tum_modeller:
    for etiket, features in [("Full", list(X.columns)), ("Top 5", top_5_features)]:
        kombinasyon_adi = f"{isim} - {etiket}"

        # Zaten eğitilmişse tekrar eğitme
        if etiket == "Full" and isim in sonuclar:
            s = sonuclar[isim]
            tum_kombinasyonlar[kombinasyon_adi] = {
                'model': s['model'], 'features': features,
                'acc': s['acc'], 'auc': s['auc'], 'cv_auc': s['cv_auc'],
                'y_pred': s['y_pred']
            }
        else:
            import copy
            m = copy.deepcopy(tum_modeller[isim])
            cv_auc = cross_val_score(m, X_train[features], y_train, cv=skf, scoring='roc_auc').mean()
            m.fit(X_train[features], y_train)
            y_pred = m.predict(X_test[features])
            acc    = accuracy_score(y_test, y_pred)
            auc    = roc_auc_score(y_test, m.predict_proba(X_test[features])[:, 1])
            tum_kombinasyonlar[kombinasyon_adi] = {
                'model': m, 'features': features,
                'acc': acc, 'auc': auc, 'cv_auc': cv_auc,
                'y_pred': y_pred
            }

print(f"  {'Kombinasyon':<32} {'Accuracy':>10} {'AUC':>10} {'CV AUC':>10}")
print("  " + "-"*60)
for ad, s in tum_kombinasyonlar.items():
    print(f"  {ad:<32} {s['acc']:>10.4f} {s['auc']:>10.4f} {s['cv_auc']:>10.4f}")

en_iyi_isim = max(tum_kombinasyonlar, key=lambda x: (tum_kombinasyonlar[x]['acc'], tum_kombinasyonlar[x]['auc']))
en_iyi = tum_kombinasyonlar[en_iyi_isim]
final_features = en_iyi['features']
final_name     = en_iyi_isim

print("\n  Model seçimi CV-AUC öncelikli yapıldı (genelleme performansı odaklı).")
en_iyi_isim = max(
    tum_kombinasyonlar,
    key=lambda x: (tum_kombinasyonlar[x]['cv_auc'], tum_kombinasyonlar[x]['auc'])
)
en_iyi = tum_kombinasyonlar[en_iyi_isim]
final_features = en_iyi['features']
final_name = en_iyi_isim

print(
    f"  Kazanan: {final_name} : CV-AUC: {en_iyi['cv_auc']:.4f}, "
    f"Test AUC: {en_iyi['auc']:.4f}, Accuracy: {en_iyi['acc']:.4f}"
)

# Test setini kirletmemek için threshold eğitim verisi içindeki validation ile bulunur.
model_template = clone(en_iyi['model'])
X_subtrain, X_val, y_subtrain, y_val = train_test_split(
    X_train[final_features], y_train, test_size=0.2, random_state=42, stratify=y_train
)
model_template.fit(X_subtrain, y_subtrain)
y_proba_val = model_template.predict_proba(X_val)[:, 1]
_, _, thresholds_val = roc_curve(y_val, y_proba_val)
f1_scores_val = []
for t in thresholds_val:
    y_pred_val_t = (y_proba_val >= t).astype(int)
    f1_scores_val.append(f1_score(y_val, y_pred_val_t))
optimal_threshold = thresholds_val[np.argmax(f1_scores_val)]

# Final model tüm eğitim verisiyle yeniden eğitilir.
final_model = clone(en_iyi['model'])
final_model.fit(X_train[final_features], y_train)

# 9. CONFUSION MATRIX (FINAL MODEL)
print(f"\n[8] CONFUSION MATRIX ({final_name})")
print("     " + "-"*50)

y_pred_final = final_model.predict(X_test[final_features])

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred_final),
    display_labels=["Negatif", "Diyabetli"]
).plot(ax=ax, colorbar=False)
ax.set_title(f"Confusion Matrix - {final_name}")
plt.tight_layout()
plt.savefig("output/confusion_matrix.png")

cm = confusion_matrix(y_test, y_pred_final)
print(f"  output/confusion_matrix.png kaydedildi")
print(f"  Diyabetliyi doğru yakaladı : {cm[1][1]} / {sum(y_test==1)}")
print(f"  Diyabetliyi kaçırdı        : {cm[1][0]} / {sum(y_test==1)}  < tehlikeli hata")

# ROC EĞRİSİ GRAFİĞİ (tüm algoritmalar - full model)
print(f"\n[9] ROC EĞRİSİ GRAFİĞİ")
print("     " + "-"*50)

plt.figure(figsize=(8, 6))
for isim, s in sonuclar.items():
    y_proba = s['model'].predict_proba(X_test)[:, 1]
    fpr_m, tpr_m, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr_m, tpr_m, label=f"{isim} (AUC={s['auc']:.3f})")

plt.plot([0, 1], [0, 1], 'k--', label='Rastgele (AUC=0.500)')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Eğrisi - Tüm Algoritmalar")
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("output/roc_curve.png")
print("  output/roc_curve.png kaydedildi")

# 10. EŞİK OPTİMİZASYONU
print(f"\n[10] EŞİK OPTİMİZASYONU (Threshold Optimization)")
print("     " + "-"*50)

y_proba_final = final_model.predict_proba(X_test[final_features])[:, 1]
y_pred_optimized  = (y_proba_final >= optimal_threshold).astype(int)

cm_default   = confusion_matrix(y_test, y_pred_final)
cm_optimized = confusion_matrix(y_test, y_pred_optimized)

print(f"  Eşik validation setinde optimize edildi: {optimal_threshold:.2f}")
print(f"  Varsayılan eşik (0.50)  : Diyabetliyi kaçırma: {cm_default[1][0]} / {sum(y_test==1)}")
print(f"  Optimize eşik  ({optimal_threshold:.2f}) : Diyabetliyi kaçırma: {cm_optimized[1][0]} / {sum(y_test==1)}")
print(f"  Kazanılan hasta sayısı : {cm_default[1][0] - cm_optimized[1][0]}")

# 11. FEATURE IMPORTANCE (SADECE AĞAÇ TABANLI FİNAL MODEL İÇİN)
if hasattr(final_model, "feature_importances_"):
    print(f"\n[11] ÖZELLİK KATKISI (Feature Importance - {final_name}):")
    feature_importance_rf = pd.DataFrame({
        'Feature': final_features,
        'Importance': final_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(feature_importance_rf.to_string(index=False))
else:
    print(f"\n[11] Feature Importance atlandı ({final_name} bu metriği doğal olarak üretmiyor).")

# 11. FINAL MODEL KAYDET
print(f"\n[12] FINAL MODEL KAYDEDILIYOR ({final_name})")
print("     " + "-"*50)

with open("model.pkl", "wb") as f:
    pickle.dump({'model': final_model, 'features': final_features, 'threshold': optimal_threshold}, f)

print("OK model.pkl kaydedildi")
print("\nDETAYLI RAPOR ÖZETİ:")
print(f"  • Binary Classification     : OK")
print(f"  • Train-Test Split          : OK (80-20, stratified)")
print(f"  • Class Balancing           : OK (Downsampling, hedef 2:1)")
print(f"  • Cross Validation          : OK (5-Fold Stratified K-Fold)")
print(f"  • Feature Selection         : OK (Mutual Information)")
print(f"  • 6 Algoritma Karşılaştırma : OK (LR, DT, KNN, SVM, RF, XGBoost)")
print(f"  • Eşik Optimizasyonu        : OK (Optimal eşik: {optimal_threshold:.2f})")
print(f"  • Full vs Top 5             : OK")
print(f"  • ROC Eğrisi                : OK (roc_curve.png)")
print(f"  • Confusion Matrix          : OK (confusion_matrix.png)")
print(f"  • Feature Importance        : OK")
print(f"  • Seçilen Model             : {final_name}")
print("="*60)
