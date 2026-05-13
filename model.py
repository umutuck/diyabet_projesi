import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, f1_score
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif, chi2
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier


def build_models(scale_pos_weight):
    return {
        "Logistic Regression": Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10, class_weight='balanced'),
        "KNN": Pipeline([
            ('scaler', StandardScaler()),
            ('model', KNeighborsClassifier(n_neighbors=5))
        ]),
        "SVM": Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, class_weight='balanced'
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, random_state=42,
            scale_pos_weight=scale_pos_weight, eval_metric='logloss', verbosity=0
        ),
    }


print("=" * 60)
print("DIYABET RISK TAHMIN MODELI - ALGORITHM x FEATURE METHOD")
print("=" * 60)

# 1. VERI YUKLEME
df = pd.read_csv("diabetes.csv")
print(f"\n[1] Veri Yüklendi: {df.shape[0]} örnek, {df.shape[1]} özellik")
print(f"    Sınıf dağılımı: {df['Outcome'].value_counts().to_dict()}")

# 2. VERI TEMIZLEME
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols] = df[cols].replace(0, np.nan)
df.fillna(df.median(numeric_only=True), inplace=True)
print("\n[2] Veri Temizleme Tamamlandı")
print("    Eksik değer giderme: Medyan imputasyon kullanıldı")

# 3. KORELASYON MATRISI
plt.figure(figsize=(9, 7))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Korelasyon Matrisi")
plt.tight_layout()
plt.savefig("korelasyon.png")
print("\n[3] Korelasyon Matrisi: korelasyon.png kaydedildi")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 4. EGITIM-TEST BOLME
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\n[4] Eğitim-Test Bölme (80-20):")
print(f"    Eğitim: {X_train.shape[0]} örnek")
print(f"    Test: {X_test.shape[0]} örnek")

# 4.1 SINIF DENGELEME (Downsampling - hedef oran: 2:1)
target_ratio = 2.0
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

# 5. OZELLIK SECIMI YONTEMLERI (Sadece eğitim verisi)
all_features = list(X.columns)
top_k = 5

mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
mi_df = pd.DataFrame({
    'Feature': all_features,
    'Score': mi_scores
}).sort_values('Score', ascending=False)
mi_top_features = mi_df.head(top_k)['Feature'].tolist()

anova_scores, _ = f_classif(X_train, y_train)
anova_df = pd.DataFrame({
    'Feature': all_features,
    'Score': anova_scores
}).sort_values('Score', ascending=False)
anova_top_features = anova_df.head(top_k)['Feature'].tolist()

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
rf_selector = RandomForestClassifier(
    n_estimators=200, max_depth=10, random_state=42, class_weight='balanced'
)
rf_selector.fit(X_train[all_features], y_train)
rf_imp_df = pd.DataFrame({
    'Feature': all_features,
    'Score': rf_selector.feature_importances_
}).sort_values('Score', ascending=False)
rf_imp_top_features = rf_imp_df.head(top_k)['Feature'].tolist()

# Chi-Square için özelliklerin non-negative olması gerekir.
chi2_scaler = MinMaxScaler()
X_train_chi2 = chi2_scaler.fit_transform(X_train[all_features])
chi2_scores, _ = chi2(X_train_chi2, y_train)
chi2_df = pd.DataFrame({
    'Feature': all_features,
    'Score': chi2_scores
}).sort_values('Score', ascending=False)
chi2_top_features = chi2_df.head(top_k)['Feature'].tolist()

# Permutation importance: model-agnostic önem yaklaşımı.
perm = permutation_importance(
    rf_selector, X_train[all_features], y_train,
    n_repeats=10, random_state=42, scoring='roc_auc'
)
perm_df = pd.DataFrame({
    'Feature': all_features,
    'Score': perm.importances_mean
}).sort_values('Score', ascending=False)
perm_top_features = perm_df.head(top_k)['Feature'].tolist()

feature_sets = {
    "Full": all_features,
    "Mutual Information Top 5": mi_top_features,
    "ANOVA F-score Top 5": anova_top_features,
    "RF Importance Top 5": rf_imp_top_features,
    "Chi-Square Top 5": chi2_top_features,
    "Permutation Top 5": perm_top_features,
}

print("\n[5] ÖZELLİK SEÇİM YÖNTEMLERİ (sadece eğitim verisi):")
print(f"    Mutual Information Top 5 : {mi_top_features}")
print(f"    ANOVA F-score Top 5      : {anova_top_features}")
print(f"    RF Importance Top 5      : {rf_imp_top_features}")
print(f"    Chi-Square Top 5         : {chi2_top_features}")
print(f"    Permutation Top 5        : {perm_top_features}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 6. TUM ALGORITMALAR x TUM FEATURE METHODLARI
print("\n[6] ALGORITHM x FEATURE METHOD KARŞILAŞTIRMASI")
print("     " + "-" * 50)
models = build_models(scale_pos_weight=scale_pos_weight)

all_results = []
best_result = None

for model_name, model in models.items():
    for fs_name, features in feature_sets.items():
        cv_auc = cross_val_score(model, X_train[features], y_train, cv=skf, scoring='roc_auc').mean()
        model.fit(X_train[features], y_train)
        y_pred = model.predict(X_test[features])
        y_proba = model.predict_proba(X_test[features])[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        result = {
            'algorithm': model_name,
            'feature_method': fs_name,
            'feature_count': len(features),
            'accuracy': acc,
            'test_auc': auc,
            'cv_auc': cv_auc,
            'model': model,
            'features': features,
            'y_pred': y_pred
        }
        all_results.append(result)

        if best_result is None or (cv_auc, auc) > (best_result['cv_auc'], best_result['test_auc']):
            best_result = result

        print(f"\n  {model_name} + {fs_name}:")
        print(f"    Accuracy : {acc:.4f}")
        print(f"    ROC-AUC  : {auc:.4f}")
        print(f"    CV AUC   : {cv_auc:.4f}")

results_df = pd.DataFrame([{
    'Algorithm': r['algorithm'],
    'Feature Method': r['feature_method'],
    'Feature Count': r['feature_count'],
    'Accuracy': r['accuracy'],
    'Test AUC': r['test_auc'],
    'CV AUC': r['cv_auc'],
} for r in all_results])

results_df = results_df.sort_values(['CV AUC', 'Test AUC', 'Accuracy'], ascending=False)
print("\n[6.1] KARŞILAŞTIRMA TABLOSU (CV AUC'ye göre sıralı):")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
results_df.to_csv("algorithm_feature_comparison.csv", index=False)
print("      algorithm_feature_comparison.csv kaydedildi")

final_model = best_result['model']
final_features = best_result['features']
final_name = f"{best_result['algorithm']} + {best_result['feature_method']}"
print(
    f"\n  Seçilen model: {final_name} -> CV-AUC: {best_result['cv_auc']:.4f}, "
    f"Test AUC: {best_result['test_auc']:.4f}, Accuracy: {best_result['accuracy']:.4f}"
)

# 7. THRESHOLD OPTIMIZASYONU (validation üzerinde)
X_subtrain, X_val, y_subtrain, y_val = train_test_split(
    X_train[final_features], y_train, test_size=0.2, random_state=42, stratify=y_train
)
val_model = build_models(scale_pos_weight=scale_pos_weight)[best_result['algorithm']]
val_model.fit(X_subtrain, y_subtrain)
y_proba_val = val_model.predict_proba(X_val)[:, 1]
_, _, thresholds_val = roc_curve(y_val, y_proba_val)
f1_scores_val = []
for t in thresholds_val:
    y_pred_val_t = (y_proba_val >= t).astype(int)
    f1_scores_val.append(f1_score(y_val, y_pred_val_t))
optimal_threshold = thresholds_val[np.argmax(f1_scores_val)]

# Final modeli tüm eğitim verisiyle kur
final_model.fit(X_train[final_features], y_train)

# 8. CONFUSION MATRIX
print(f"\n[8] CONFUSION MATRIX ({final_name})")
print("     " + "-" * 50)
y_pred_final = final_model.predict(X_test[final_features])

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred_final),
    display_labels=["Negatif", "Diyabetli"]
).plot(ax=ax, colorbar=False)
ax.set_title(f"Confusion Matrix - {final_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png")

cm = confusion_matrix(y_test, y_pred_final)
print("  confusion_matrix.png kaydedildi")
print(f"  Diyabetliyi doğru yakaladı : {cm[1][1]} / {sum(y_test == 1)}")
print(f"  Diyabetliyi kaçırdı        : {cm[1][0]} / {sum(y_test == 1)}  <- tehlikeli hata")

# 9. ROC EGRISI (tek algoritma)
print("\n[9] ROC EĞRİSİ GRAFİĞİ")
print("     " + "-" * 50)
y_proba_final = final_model.predict_proba(X_test[final_features])[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba_final)
auc_final = roc_auc_score(y_test, y_proba_final)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"{final_name} (AUC={auc_final:.3f})")
plt.plot([0, 1], [0, 1], 'k--', label='Rastgele (AUC=0.500)')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Eğrisi - Random Forest")
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("roc_curve.png")
print("  roc_curve.png kaydedildi")

# 10. TESTTE ESIK ETKISI (esik validation'dan gelir)
print("\n[10] EŞİK OPTİMİZASYONU (Threshold Optimization)")
print("     " + "-" * 50)
y_pred_optimized = (y_proba_final >= optimal_threshold).astype(int)
cm_default = confusion_matrix(y_test, y_pred_final)
cm_optimized = confusion_matrix(y_test, y_pred_optimized)
print(f"  Eşik validation setinde optimize edildi: {optimal_threshold:.2f}")
print(f"  Varsayılan eşik (0.50) -> Diyabetliyi kaçırma: {cm_default[1][0]} / {sum(y_test == 1)}")
print(f"  Optimize eşik ({optimal_threshold:.2f}) -> Diyabetliyi kaçırma: {cm_optimized[1][0]} / {sum(y_test == 1)}")
print(f"  Kazanılan hasta sayısı : {cm_default[1][0] - cm_optimized[1][0]}")

# 11. FEATURE IMPORTANCE
if hasattr(final_model, "feature_importances_"):
    print(f"\n[11] ÖZELLİK KATKISI (Feature Importance - {final_name}):")
    feature_importance_rf = pd.DataFrame({
        'Feature': final_features,
        'Importance': final_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(feature_importance_rf.to_string(index=False))
else:
    print(f"\n[11] Feature importance bu modelde doğrudan yok ({best_result['algorithm']}).")

# 12. FINAL MODEL KAYDET
print(f"\n[12] FINAL MODEL KAYDEDILIYOR ({final_name})")
print("     " + "-" * 50)
with open("model.pkl", "wb") as f:
    pickle.dump({'model': final_model, 'features': final_features, 'threshold': optimal_threshold}, f)

print("✓ model.pkl kaydedildi")
print("\nDETAYLI RAPOR ÖZETİ:")
print("  • Binary Classification     : ✓")
print("  • Train-Test Split          : ✓ (80-20, stratified)")
print("  • Class Balancing           : ✓ (Downsampling, hedef 2:1)")
print("  • Cross Validation          : ✓ (5-Fold Stratified K-Fold)")
print("  • Feature Selection         : ✓ (Full, MI, ANOVA, RF Imp, Chi2, Permutation)")
print("  • Algoritma Karşılaştırma   : ✓ (LR, DT, KNN, SVM, RF, XGBoost)")
print("  • Kombinasyon Testi         : ✓ (Algoritma x Feature Method)")
print(f"  • Eşik Optimizasyonu        : ✓ (Optimal eşik: {optimal_threshold:.2f})")
print("  • ROC Eğrisi                : ✓ (roc_curve.png)")
print("  • Confusion Matrix          : ✓ (confusion_matrix.png)")
print("  • Sonuç Tablosu             : ✓ (algorithm_feature_comparison.csv)")
print("  • Feature Importance        : ✓ (model uygunsa)")
print(f"  • Seçilen Model             : {final_name}")
print("=" * 60)
