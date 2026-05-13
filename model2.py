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
from sklearn.feature_selection import mutual_info_classif, f_classif, chi2
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier


def build_models(scale_pos_weight):
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")),
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10, class_weight="balanced"),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, random_state=42,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss", verbosity=0
        ),
    }


print("=" * 64)
print("DIYABET RISK TAHMIN MODELI - DATASET OF DIABETES")
print("=" * 64)

# 1. VERI YUKLEME
df = pd.read_csv("Dataset of Diabetes .csv")
print(f"\n[1] Veri Yüklendi: {df.shape[0]} örnek, {df.shape[1]} özellik")

# 2. HEDEF KOLON ve TEMIZLIK
df["CLASS"] = df["CLASS"].astype(str).str.strip()
class_map = {"N": 0, "Y": 1, "P": 1}
df = df[df["CLASS"].isin(class_map.keys())].copy()
df["Outcome"] = df["CLASS"].map(class_map)
print(f"    Sınıf dağılımı (Outcome): {df['Outcome'].value_counts().to_dict()}")
print("    Not: CLASS=Y ve CLASS=P pozitif (1) kabul edildi.")

# 3. OZELLIK HAZIRLAMA
# ID benzeri kolonları çıkarıyoruz; modele bilgi sızıntısı yaratabilir.
drop_cols = ["ID", "No_Pation", "CLASS", "Outcome"]
feature_df = df.drop(columns=drop_cols, errors="ignore").copy()

# Gender sütununu sayısallaştır.
if "Gender" in feature_df.columns:
    feature_df["Gender"] = feature_df["Gender"].astype(str).str.upper().str.strip().map({"F": 0, "M": 1})

# Güvenlik: tamamen sayısal olmayan kolon kalırsa numerik'e zorla.
for col in feature_df.columns:
    feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")

feature_df.fillna(feature_df.median(numeric_only=True), inplace=True)
print(f"[2] Kullanılacak özellikler: {list(feature_df.columns)}")

# Korelasyon görseli
plt.figure(figsize=(10, 8))
sns.heatmap(pd.concat([feature_df, df["Outcome"]], axis=1).corr(numeric_only=True), cmap="coolwarm")
plt.title("Korelasyon Matrisi (Dataset of Diabetes)")
plt.tight_layout()
plt.savefig("korelasyon_model2.png")
print("[3] korelasyon_model2.png kaydedildi")

X = feature_df
y = df["Outcome"]

# 4. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[4] Eğitim-Test Bölme: Eğitim={X_train.shape[0]} Test={X_test.shape[0]}")

# 4.1 DENGELEME (hedef 2:1)
target_ratio = 2.0
train_df = X_train.copy()
train_df["Outcome"] = y_train.values

counts = train_df["Outcome"].value_counts()
majority_class = counts.idxmax()
minority_class = counts.idxmin()
majority_count = counts[majority_class]
minority_count = counts[minority_class]
current_ratio = majority_count / minority_count

if current_ratio > target_ratio:
    target_majority_count = int(target_ratio * minority_count)
    majority_part = train_df[train_df["Outcome"] == majority_class].sample(
        n=target_majority_count, random_state=42
    )
    minority_part = train_df[train_df["Outcome"] == minority_class]
    balanced_train_df = pd.concat([majority_part, minority_part], axis=0).sample(frac=1.0, random_state=42)
    print(f"[4.1] Downsampling uygulandı: {current_ratio:.2f}:1 -> {target_ratio:.1f}:1")
else:
    balanced_train_df = train_df.copy()
    print(f"[4.1] Downsampling gerekmedi: {current_ratio:.2f}:1")

X_train = balanced_train_df.drop("Outcome", axis=1)
y_train = balanced_train_df["Outcome"]
print(f"      Yeni eğitim dağılımı: {y_train.value_counts().to_dict()}")

# 5. FEATURE METHODLARI
all_features = list(X.columns)
top_k = min(5, len(all_features))

mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
mi_df = pd.DataFrame({"Feature": all_features, "Score": mi_scores}).sort_values("Score", ascending=False)
mi_top_features = mi_df.head(top_k)["Feature"].tolist()

anova_scores, _ = f_classif(X_train, y_train)
anova_df = pd.DataFrame({"Feature": all_features, "Score": anova_scores}).sort_values("Score", ascending=False)
anova_top_features = anova_df.head(top_k)["Feature"].tolist()

scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
rf_selector = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight="balanced")
rf_selector.fit(X_train[all_features], y_train)
rf_imp_df = pd.DataFrame(
    {"Feature": all_features, "Score": rf_selector.feature_importances_}
).sort_values("Score", ascending=False)
rf_imp_top_features = rf_imp_df.head(top_k)["Feature"].tolist()

chi2_scaler = MinMaxScaler()
X_train_chi2 = chi2_scaler.fit_transform(X_train[all_features])
chi2_scores, _ = chi2(X_train_chi2, y_train)
chi2_df = pd.DataFrame({"Feature": all_features, "Score": chi2_scores}).sort_values("Score", ascending=False)
chi2_top_features = chi2_df.head(top_k)["Feature"].tolist()

perm = permutation_importance(
    rf_selector, X_train[all_features], y_train, n_repeats=10, random_state=42, scoring="roc_auc"
)
perm_df = pd.DataFrame({"Feature": all_features, "Score": perm.importances_mean}).sort_values("Score", ascending=False)
perm_top_features = perm_df.head(top_k)["Feature"].tolist()

feature_sets = {
    "Full": all_features,
    "Mutual Information Top 5": mi_top_features,
    "ANOVA F-score Top 5": anova_top_features,
    "RF Importance Top 5": rf_imp_top_features,
    "Chi-Square Top 5": chi2_top_features,
    "Permutation Top 5": perm_top_features,
}

print("\n[5] Feature methodleri:")
for k, v in feature_sets.items():
    if k == "Full":
        print(f"    {k}: {len(v)} özellik")
    else:
        print(f"    {k}: {v}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = build_models(scale_pos_weight=scale_pos_weight)

# 6. ALGORITHM x FEATURE METHOD
print("\n[6] Algorithm x Feature Method karşılaştırması")
all_results = []
best_result = None

for model_name, model in models.items():
    for fs_name, features in feature_sets.items():
        cv_auc = cross_val_score(model, X_train[features], y_train, cv=skf, scoring="roc_auc").mean()
        model.fit(X_train[features], y_train)
        y_pred = model.predict(X_test[features])
        y_proba = model.predict_proba(X_test[features])[:, 1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        row = {
            "algorithm": model_name,
            "feature_method": fs_name,
            "feature_count": len(features),
            "accuracy": acc,
            "test_auc": auc,
            "cv_auc": cv_auc,
            "model": model,
            "features": features,
        }
        all_results.append(row)

        if best_result is None or (cv_auc, auc) > (best_result["cv_auc"], best_result["test_auc"]):
            best_result = row

results_df = pd.DataFrame([{
    "Algorithm": r["algorithm"],
    "Feature Method": r["feature_method"],
    "Feature Count": r["feature_count"],
    "Accuracy": r["accuracy"],
    "Test AUC": r["test_auc"],
    "CV AUC": r["cv_auc"],
} for r in all_results]).sort_values(["CV AUC", "Test AUC", "Accuracy"], ascending=False)

print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
results_df.to_csv("algorithm_feature_comparison_model2.csv", index=False)
print("[6.1] algorithm_feature_comparison_model2.csv kaydedildi")

final_model = best_result["model"]
final_features = best_result["features"]
final_name = f"{best_result['algorithm']} + {best_result['feature_method']}"
print(
    f"\nSeçilen model: {final_name} "
    f"(CV-AUC={best_result['cv_auc']:.4f}, Test AUC={best_result['test_auc']:.4f}, Acc={best_result['accuracy']:.4f})"
)

# 7. Eşik optimizasyonu (validation)
X_subtrain, X_val, y_subtrain, y_val = train_test_split(
    X_train[final_features], y_train, test_size=0.2, random_state=42, stratify=y_train
)
val_model = build_models(scale_pos_weight=scale_pos_weight)[best_result["algorithm"]]
val_model.fit(X_subtrain, y_subtrain)
y_proba_val = val_model.predict_proba(X_val)[:, 1]
_, _, thresholds_val = roc_curve(y_val, y_proba_val)
f1_scores_val = [f1_score(y_val, (y_proba_val >= t).astype(int)) for t in thresholds_val]
optimal_threshold = thresholds_val[int(np.argmax(f1_scores_val))]

final_model.fit(X_train[final_features], y_train)
y_pred_final = final_model.predict(X_test[final_features])
y_proba_final = final_model.predict_proba(X_test[final_features])[:, 1]

# 8. Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred_final),
    display_labels=["Negatif", "Pozitif"]
).plot(ax=ax, colorbar=False)
ax.set_title(f"Confusion Matrix - {final_name}")
plt.tight_layout()
plt.savefig("confusion_matrix_model2.png")

# 9. ROC
fpr, tpr, _ = roc_curve(y_test, y_proba_final)
auc_final = roc_auc_score(y_test, y_proba_final)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"{final_name} (AUC={auc_final:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.500)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - model2")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve_model2.png")

# 10. threshold etkisi
y_pred_opt = (y_proba_final >= optimal_threshold).astype(int)
cm_default = confusion_matrix(y_test, y_pred_final)
cm_opt = confusion_matrix(y_test, y_pred_opt)
print(f"\n[10] Eşik (validation): {optimal_threshold:.2f}")
print(f"     Varsayılan eşik kaçırılan pozitif: {cm_default[1][0]} / {sum(y_test == 1)}")
print(f"     Optimize eşik kaçırılan pozitif : {cm_opt[1][0]} / {sum(y_test == 1)}")

# 11. importance (model uygunsa)
if hasattr(final_model, "feature_importances_"):
    imp_df = pd.DataFrame({
        "Feature": final_features,
        "Importance": final_model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    print("\n[11] Feature Importance:")
    print(imp_df.to_string(index=False))
else:
    print(f"\n[11] {best_result['algorithm']} için doğrudan feature_importances_ yok.")

# 12. kaydet
with open("model2.pkl", "wb") as f:
    pickle.dump({"model": final_model, "features": final_features, "threshold": optimal_threshold}, f)
print("\n[12] model2.pkl kaydedildi")
print(classification_report(y_test, y_pred_final, target_names=["Negatif", "Pozitif"]))
