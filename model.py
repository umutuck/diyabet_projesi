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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import numpy as np

print("="*60)
print("DIYABET RISK TAHMIN MODELI - ADVANCED")
print("="*60)

# 1. VERI YÜKLEME
df = pd.read_csv("diabetes.csv")
print(f"\n[1] Veri Yüklendi: {df.shape[0]} örnek, {df.shape[1]} özellik")
print(f"    Sınıf dağılımı: {df['Outcome'].value_counts().to_dict()}")

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
plt.savefig("korelasyon.png")
print("\n[3] Korelasyon Matrisi: korelasyon.png kaydedildi")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 4. EĞITIM-TEST BÖLME (Özellik seçiminden önce - Data Leakage önlemi)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[4] Eğitim-Test Bölme (80-20):")
print(f"    Eğitim: {X_train.shape[0]} örnek")
print(f"    Test: {X_test.shape[0]} örnek")

# 5. ÖZELLİK SEÇİMİ - sadece X_train üzerinde (Data Leakage önlemi)
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'MI_Score': mi_scores
}).sort_values('MI_Score', ascending=False)

print("\n[5] ÖZELLİK ÖNEMLİLİĞİ (Mutual Information - sadece eğitim verisi):")
print(feature_importance_df.to_string(index=False))

top_5_features = feature_importance_df.head(5)['Feature'].tolist()
print(f"\n    En önemli 5 özellik: {top_5_features}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 6. TÜM MODELLERİ EĞİT (TAM ÖZELLİKLERLE)
print("\n[6] MODEL EĞİTİMİ (5 Farklı Algoritma - 8 özellik)")
print("     " + "-"*50)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

modeller = {
    "Logistic Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
    ]),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10, class_weight='balanced'),
    "KNN": Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier(n_neighbors=5))
    ]),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, class_weight='balanced'),
    "XGBoost": XGBClassifier(n_estimators=100, random_state=42, max_depth=6,
                              scale_pos_weight=scale_pos_weight,
                              eval_metric='logloss', verbosity=0),
}

sonuclar = {}

for isim, model in modeller.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    cv_auc = cross_val_score(model, X, y, cv=skf, scoring='roc_auc').mean()

    sonuclar[isim] = {
        'model': model, 'y_pred': y_pred,
        'acc': acc, 'auc': auc, 'cv_auc': cv_auc,
        'features': list(X.columns)
    }

    print(f"\n  {isim}:")
    print(f"    Accuracy : {acc:.4f}")
    print(f"    ROC-AUC  : {auc:.4f}")
    print(f"    CV AUC   : {cv_auc:.4f}")
    print(f"    Classification Report:\n{classification_report(y_test, y_pred, target_names=['Sağlıklı','Diyabetli'])}")

# 7. TÜM KOMBİNASYONLAR: HER ALGORİTMA × (FULL + TOP 5)
print(f"\n[7] MODEL KARŞILAŞTIRMASI (5 Algoritma × Full + Top 5 = 10 kombinasyon)")
print("     " + "-"*62)

tum_modeller = {
    "Logistic Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
    ]),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10, class_weight='balanced'),
    "KNN": Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier(n_neighbors=5))
    ]),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, class_weight='balanced'),
    "XGBoost": XGBClassifier(n_estimators=100, random_state=42, max_depth=6,
                              scale_pos_weight=scale_pos_weight,
                              eval_metric='logloss', verbosity=0),
}

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
            m.fit(X_train[features], y_train)
            y_pred = m.predict(X_test[features])
            acc    = accuracy_score(y_test, y_pred)
            auc    = roc_auc_score(y_test, m.predict_proba(X_test[features])[:, 1])
            cv_auc = cross_val_score(m, X[features], y, cv=skf, scoring='roc_auc').mean()
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
final_model    = en_iyi['model']
final_features = en_iyi['features']
final_name     = en_iyi_isim

print(f"\n  Kazanan: {final_name} → Accuracy: {en_iyi['acc']:.4f}, AUC: {en_iyi['auc']:.4f}")

# 9. CONFUSION MATRIX (FINAL MODEL)
print(f"\n[8] CONFUSION MATRIX ({final_name})")
print("     " + "-"*50)

y_pred_final = en_iyi['y_pred']

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred_final),
    display_labels=["Sağlıklı", "Diyabetli"]
).plot(ax=ax, colorbar=False)
ax.set_title(f"Confusion Matrix - {final_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png")

cm = confusion_matrix(y_test, y_pred_final)
print(f"  confusion_matrix.png kaydedildi")
print(f"  Diyabetliyi doğru yakaladı : {cm[1][1]} / {sum(y_test==1)}")
print(f"  Diyabetliyi kaçırdı        : {cm[1][0]} / {sum(y_test==1)}  ← tehlikeli hata")

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
plt.savefig("roc_curve.png")
print("  roc_curve.png kaydedildi")

# 10. EŞİK OPTİMİZASYONU
print(f"\n[10] EŞİK OPTİMİZASYONU (Threshold Optimization)")
print("     " + "-"*50)

y_proba_final = final_model.predict_proba(X_test[final_features])[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba_final)

f1_scores = []
for t in thresholds:
    y_pred_t = (y_proba_final >= t).astype(int)
    f1_scores.append(f1_score(y_test, y_pred_t))

optimal_threshold = thresholds[np.argmax(f1_scores)]
y_pred_optimized  = (y_proba_final >= optimal_threshold).astype(int)

cm_default   = confusion_matrix(y_test, y_pred_final)
cm_optimized = confusion_matrix(y_test, y_pred_optimized)

print(f"  Varsayılan eşik (0.50)  → Diyabetliyi kaçırma: {cm_default[1][0]} / {sum(y_test==1)}")
print(f"  Optimize eşik  ({optimal_threshold:.2f}) → Diyabetliyi kaçırma: {cm_optimized[1][0]} / {sum(y_test==1)}")
print(f"  Kazanılan hasta sayısı : {cm_default[1][0] - cm_optimized[1][0]}")

# 10. FEATURE IMPORTANCE (SADECE RANDOM FOREST İÇİN)
if "Random Forest" in sonuclar:
    print("\n[11] ÖZELLİK KATKISI (Feature Importance - Random Forest):")
    rf_model = sonuclar["Random Forest"]['model']
    feature_importance_rf = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(feature_importance_rf.to_string(index=False))

# 11. FINAL MODEL KAYDET
print(f"\n[12] FINAL MODEL KAYDEDILIYOR ({final_name})")
print("     " + "-"*50)

with open("model.pkl", "wb") as f:
    pickle.dump({'model': final_model, 'features': final_features, 'threshold': optimal_threshold}, f)

print("✓ model.pkl kaydedildi")
print("\nDETAYLI RAPOR ÖZETİ:")
print(f"  • Binary Classification     : ✓")
print(f"  • Train-Test Split          : ✓ (80-20, stratified)")
print(f"  • Cross Validation          : ✓ (5-Fold Stratified K-Fold)")
print(f"  • Feature Selection         : ✓ (Mutual Information)")
print(f"  • 5 Algoritma Karşılaştırma : ✓ (LR, DT, KNN, RF, XGBoost)")
print(f"  • Eşik Optimizasyonu        : ✓ (Optimal eşik: {optimal_threshold:.2f})")
print(f"  • Full vs Top 5             : ✓")
print(f"  • ROC Eğrisi                : ✓ (roc_curve.png)")
print(f"  • Confusion Matrix          : ✓ (confusion_matrix.png)")
print(f"  • Feature Importance        : ✓")
print(f"  • Seçilen Model             : {final_name}")
print("="*60)
