import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, f1_score
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

def build_models():
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
            ('model', SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'))
        ]),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced'),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, random_state=42, eval_metric='mlogloss')
    }

print("=" * 60)
print("DIYABET ÇOK SINIFLI RISK TAHMIN MODELI (0-1-2)")
print("=" * 60)

# 2. VERİ YÜKLEME VE TEMİZLEME
df = pd.read_csv("final.csv")
df = df.fillna(df.median(numeric_only=True))

# DUPLICATE TEMİZLEME
onceki = df.shape[0]
df = df.drop_duplicates()
sonraki = df.shape[0]
print(f"\n[0] Duplicate Kontrolü:")
print(f"    Önceki satır sayısı : {onceki}")
print(f"    Silinen duplicate   : {onceki - sonraki}")
print(f"    Kalan satır sayısı  : {sonraki}")

X = df.drop("Target", axis=1)
y = df["Target"]

print(f"\n[1] Veri Hazır: {df.shape[0]} satır, {df.shape[1]} sütun")
print(f"    Sınıf Dağılımı (orijinal): {y.value_counts().to_dict()}")

# 3. EĞİTİM-TEST AYRIMI
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SMOTE — sadece train setine uygula, test dokunulmasın
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)
print(f"    Sınıf Dağılımı (SMOTE sonrası): {pd.Series(y_train).value_counts().to_dict()}")

# 4. ÖZELLİK SEÇİMİ
top_k = 5
all_features = sorted(list(X.columns))

mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
mi_top = sorted(
    pd.DataFrame({'F': list(X.columns), 'S': mi_scores})
    .sort_values('S', ascending=False)['F'].head(top_k).tolist()
)

f_scores, _ = f_classif(X_train, y_train)
anova_top = sorted(
    pd.DataFrame({'F': list(X.columns), 'S': f_scores})
    .sort_values('S', ascending=False)['F'].head(top_k).tolist()
)

rf_sel = RandomForestClassifier(random_state=42).fit(X_train[all_features], y_train)
rf_top = sorted(
    pd.DataFrame({'F': all_features, 'S': rf_sel.feature_importances_})
    .sort_values('S', ascending=False)['F'].head(top_k).tolist()
)

feature_sets = {
    "Full Set":    all_features,
    "MI Top 5":    mi_top,
    "ANOVA Top 5": anova_top,
    "RF Top 5":    rf_top,
}

print(f"\n[2] En İyi Özellikler (MI): {mi_top}")

# 5. MODEL KARŞILAŞTIRMA DÖNGÜSÜ
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_results = []
best_auc = 0
best_model_name = None
best_model_key = None
best_feat_list = None

print("\n[3] Modeller Eğitiliyor ve Karşılaştırılıyor...")

for m_name, model in build_models().items():
    for fs_name, f_list in feature_sets.items():
        cv_score = cross_val_score(
            model, X_train[f_list], y_train,
            cv=skf, scoring='roc_auc_ovr'
        ).mean()

        model.fit(X_train[f_list], y_train)
        y_proba = model.predict_proba(X_test[f_list])
        test_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

        all_results.append({
            'Model': m_name,
            'Features': fs_name,
            'CV_AUC': round(cv_score, 6),
            'Test_AUC': round(test_auc, 6)
        })

        if test_auc > best_auc:
            best_auc = test_auc
            best_model_name = f"{m_name} ({fs_name})"
            best_model_key = m_name
            best_feat_list = f_list

# 6. SONUÇLARI GÖSTER
results_df = pd.DataFrame(all_results).sort_values('Test_AUC', ascending=False)
print("\n[4] Performans Tablosu (İlk 10):")
print(results_df.head(10).to_string(index=False))
print(f"\n[5] EN İYİ MODEL: {best_model_name}")
print(f"    Test AUC   : {best_auc:.6f}")
print(f"    Özellikler : {best_feat_list}")

# 7. EN İYİ MODELİ YENİDEN FİT ET
final_models = build_models()
final_model_obj = final_models[best_model_key]
final_model_obj.fit(X_train[best_feat_list], y_train)

# 8. TAHMİN VE METRİKLER
y_pred  = final_model_obj.predict(X_test[best_feat_list])
y_proba = final_model_obj.predict_proba(X_test[best_feat_list])

acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average='weighted')
auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

print(f"\n[6] Test Accuracy : {acc:.4f}")
print(f"    Weighted F1   : {f1:.4f}")
print(f"    Test AUC      : {auc:.6f}")

# 9. CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay(cm, display_labels=["Sağlıklı", "Riskli", "Diyabetli"]).plot(
    cmap='Blues', ax=ax
)
ax.set_title(f"Confusion Matrix\n{best_model_name}", fontsize=13)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# 10. MODEL KARŞILAŞTIRMA GRAFIĞI
fig, ax = plt.subplots(figsize=(13, 6))
pivot = results_df.pivot(index='Model', columns='Features', values='Test_AUC')
pivot.plot(kind='bar', ax=ax, colormap='tab10', edgecolor='black', linewidth=0.5)
ax.set_title("Model vs Feature Set — Test AUC Karşılaştırması", fontsize=13)
ax.set_ylabel("Test AUC (OVR)")
ax.set_ylim(results_df['Test_AUC'].min() - 0.02, 1.0)
ax.legend(title="Feature Set", bbox_to_anchor=(1.01, 1), loc='upper left')
ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.close()

# 11. MODEL KAYDETME
model_data = {
    'model':          final_model_obj,
    'features':       best_feat_list,
    'model_name':     best_model_name,
    'target_mapping': {0: 'Sağlıklı', 1: 'Riskli', 2: 'Diyabetli'}
}

with open("model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("\n[7] ✓ Kaydedilen dosyalar:")
print("    → model.pkl")
print("    → confusion_matrix.png")
print("    → model_comparison.png")
print("=" * 60)