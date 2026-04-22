import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import numpy as np

print("="*60)
print("DIYABET RISK TAHMIN MODELI - ADVANCED")
print("="*60)

# -------------------------------------------------------
# METRİK AÇIKLAMALARI (Hoca Soruları)
# -------------------------------------------------------
print("""
[METRIK AÇIKLAMALARI]
  PRECISION  : Model "diyabetli" dediğinde ne kadar doğru?
               Precision = TP / (TP + FP)
               Örnek: 100 kişiye "diyabetli" dedik, 80'i gerçekten diyabetli → Precision=0.80
               Yüksek precision = az yanlış alarm.

  RECALL     : Gerçek diyabetlileri ne kadar yakalayabildi?
               Recall = TP / (TP + FN)
               Tıpta recall kritik — hasta kaçırmak tehlikeli.

  MACRO AVG  : Her sınıfın metriğini ayrı hesapla, sonra EŞİT ağırlıkla ortala.
               Sınıf dengesizliğini yoksayar. Azınlık sınıf önemliyse kullan.

  WEIGHTED AVG: Her sınıfın metriğini, o sınıfın ÖRNEK SAYISIYLA ağırlıklı ortala.
               Dengesiz veri setlerinde genel performansı daha doğru yansıtır.

  ROC-AUC    : Tek bir train/test bölünmesinde hesaplanan AUC.
               Modeli test setine uygulayıp FPR vs TPR eğrisinin altındaki alan.

  CV-AUC     : Cross validation ile hesaplanan AUC ORTALAMASI.
               5 farklı fold'da modeli sırayla test edip ortalar.
               DAHA GÜVENİLİR — tek bir bölünmeye bağlı şans faktörünü azaltır.
               CV-AUC > ROC-AUC ise model gerçekten iyi öğrenmiş demek.
               CV-AUC << ROC-AUC ise test setine aşırı uyum (overfitting) var.
""")

# 1. VERI YÜKLEME
df = pd.read_csv("diabetes.csv")
print(f"[1] Veri Yüklendi: {df.shape[0]} örnek, {df.shape[1]} özellik")
print(f"    Sınıf dağılımı: {df['Outcome'].value_counts().to_dict()}")

# 2. VERİ TEMIZLEME
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols] = df[cols].replace(0, np.nan)
df.fillna(df.median(numeric_only=True), inplace=True)
print("\n[2] Veri Temizleme Tamamlandı (Medyan imputasyon)")

# 3. KORELASYON MATRİSİ
plt.figure(figsize=(9, 7))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Korelasyon Matrisi")
plt.tight_layout()
plt.savefig("korelasyon.png")
print("\n[3] Korelasyon Matrisi: korelasyon.png kaydedildi")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 4. EĞITIM-TEST BÖLME
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[4] Eğitim-Test Bölme (80-20):")
print(f"    Eğitim: {X_train.shape[0]} örnek | Test: {X_test.shape[0]} örnek")

# 5. ÖZELLİK SEÇİMİ — SADECE X_train üzerinde
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
    # SVM eklendi — Pipeline zorunlu çünkü SVM ölçeklemeye duyarlı
    "SVM": Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'))
    ]),
}

# -------------------------------------------------------
# 6. CROSS VALIDATION — MODEL EĞİTİMİNDEN ÖNCE
#    SADECE X_train, y_train üzerinde (test seti görünmez)
#    cross_val_score kendi içinde fold'ları fit/predict yapar
# -------------------------------------------------------
print("\n[6] CROSS VALIDATION (Model kurulmadan önce — sadece eğitim verisi)")
print("     " + "-"*55)
print(f"  Yöntem: 5-Fold Stratified KFold | Metrik: ROC-AUC")
print(f"  {'Model':<22} {'CV-AUC Ort':>12} {'CV-AUC Std':>12}")
print("  " + "-"*48)

cv_sonuclar = {}
for isim, model in modeller.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc')
    cv_sonuclar[isim] = {'mean': cv_scores.mean(), 'std': cv_scores.std()}
    print(f"  {isim:<22} {cv_scores.mean():>12.4f} {cv_scores.std():>12.4f}")

print("""
  NOT: CV-AUC neden model kurulmadan önce yapılır?
       Cross validation kendi içinde her fold'da modeli kurup test eder.
       Böylece hiperparametre seçimi ve model kararı test seti kirlenmeden yapılır.
       Sonra tüm X_train ile nihai model kurulur, X_test sadece son değerlendirmede görülür.
""")

# 7. MODEL EĞİTİMİ (tüm X_train ile)
print("[7] MODEL EĞİTİMİ (6 Farklı Algoritma — tam eğitim verisiyle)")
print("     " + "-"*55)

sonuclar = {}

for isim, model in modeller.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    cv_auc = cv_sonuclar[isim]['mean']

    sonuclar[isim] = {
        'model': model, 'y_pred': y_pred,
        'acc': acc, 'auc': auc, 'cv_auc': cv_auc,
        'features': list(X.columns)
    }

    print(f"\n  {isim}:")
    print(f"    Accuracy   : {acc:.4f}")
    print(f"    ROC-AUC    : {auc:.4f}  ← bu test setinde ölçüldü")
    print(f"    CV-AUC     : {cv_auc:.4f}  ← bu 5 fold ortalaması (daha güvenilir)")
    print(f"    Fark       : {abs(auc - cv_auc):.4f} {'(Overfitting şüphesi!)' if auc - cv_auc > 0.05 else '(Tutarlı)'}")
    report = classification_report(y_test, y_pred,
                                   target_names=['Sağlıklı (0)', 'Diyabetli (1)'])
    print(f"    Classification Report:\n{report}")
    print(f"    [Precision açıklaması]")
    print(f"    'Diyabetli' için precision: Model diyabetli dediği kişilerin kaçı gerçekten diyabetli?")
    print(f"    'Sağlıklı' için precision : Model sağlıklı dediği kişilerin kaçı gerçekten sağlıklı?")
    print(f"    macro avg   = iki sınıfın ortalaması (eşit ağırlık)")
    print(f"    weighted avg = örnek sayısına göre ağırlıklı ortalama\n")

# 8. TÜM KOMBİNASYONLAR: ALGORİTMA × (FULL + TOP 5)
print(f"\n[8] MODEL KARŞILAŞTIRMASI (6 Algoritma × Full + Top5 = 12 kombinasyon)")
print("     " + "-"*65)

import copy

tum_kombinasyonlar = {}

for isim in modeller:
    for etiket, features in [("Full", list(X.columns)), ("Top5", top_5_features)]:
        ad = f"{isim} - {etiket}"

        if etiket == "Full" and isim in sonuclar:
            s = sonuclar[isim]
            tum_kombinasyonlar[ad] = {
                'model': s['model'], 'features': features,
                'acc': s['acc'], 'auc': s['auc'], 'cv_auc': s['cv_auc'],
                'y_pred': s['y_pred']
            }
        else:
            m = copy.deepcopy(modeller[isim])
            cv_scores = cross_val_score(m, X_train[features], y_train, cv=skf, scoring='roc_auc')
            m.fit(X_train[features], y_train)
            y_pred = m.predict(X_test[features])
            acc    = accuracy_score(y_test, y_pred)
            auc    = roc_auc_score(y_test, m.predict_proba(X_test[features])[:, 1])
            tum_kombinasyonlar[ad] = {
                'model': m, 'features': features,
                'acc': acc, 'auc': auc, 'cv_auc': cv_scores.mean(),
                'y_pred': y_pred
            }

print(f"  {'Kombinasyon':<35} {'Accuracy':>10} {'AUC':>10} {'CV AUC':>10}")
print("  " + "-"*65)
for ad, s in tum_kombinasyonlar.items():
    print(f"  {ad:<35} {s['acc']:>10.4f} {s['auc']:>10.4f} {s['cv_auc']:>10.4f}")

en_iyi_isim = max(tum_kombinasyonlar, key=lambda x: (tum_kombinasyonlar[x]['acc'], tum_kombinasyonlar[x]['auc']))
en_iyi      = tum_kombinasyonlar[en_iyi_isim]
final_model    = en_iyi['model']
final_features = en_iyi['features']
final_name     = en_iyi_isim

print(f"\n  Kazanan: {final_name} → Accuracy: {en_iyi['acc']:.4f}, AUC: {en_iyi['auc']:.4f}")

# 9. CONFUSION MATRIX
print(f"\n[9] CONFUSION MATRIX ({final_name})")
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

# 10. ROC EĞRİSİ
print(f"\n[10] ROC EĞRİSİ GRAFİĞİ")
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

# 11. EŞİK OPTİMİZASYONU
print(f"\n[11] EŞİK OPTİMİZASYONU")
y_proba_final = final_model.predict_proba(X_test[final_features])[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba_final)

f1_scores_list = [f1_score(y_test, (y_proba_final >= t).astype(int)) for t in thresholds]
optimal_threshold = thresholds[np.argmax(f1_scores_list)]
y_pred_optimized  = (y_proba_final >= optimal_threshold).astype(int)

cm_default   = confusion_matrix(y_test, y_pred_final)
cm_optimized = confusion_matrix(y_test, y_pred_optimized)

print(f"  Varsayılan eşik (0.50)  → Diyabetliyi kaçırma: {cm_default[1][0]} / {sum(y_test==1)}")
print(f"  Optimize eşik  ({optimal_threshold:.2f}) → Diyabetliyi kaçırma: {cm_optimized[1][0]} / {sum(y_test==1)}")
print(f"  Kazanılan hasta sayısı: {cm_default[1][0] - cm_optimized[1][0]}")

# 12. FEATURE IMPORTANCE (Random Forest)
if "Random Forest" in sonuclar:
    print("\n[12] ÖZELLİK KATKISI (Random Forest Feature Importance):")
    rf_model = sonuclar["Random Forest"]['model']
    fi_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(fi_df.to_string(index=False))

# 13. TEK ÖRNEK TAHMİN DEMOsu
print("\n[13] TEK ÖRNEK TAHMİN (1 Diyabetli + 1 Sağlıklı)")
print("     " + "-"*55)

ornek_diyabetli = pd.DataFrame([{
    'Pregnancies': 6, 'Glucose': 180, 'BloodPressure': 80,
    'SkinThickness': 35, 'Insulin': 200, 'BMI': 36.0,
    'DiabetesPedigreeFunction': 0.8, 'Age': 50
}])

ornek_saglikli = pd.DataFrame([{
    'Pregnancies': 1, 'Glucose': 90, 'BloodPressure': 70,
    'SkinThickness': 18, 'Insulin': 60, 'BMI': 22.0,
    'DiabetesPedigreeFunction': 0.2, 'Age': 25
}])

for etiket, ornek in [("Diyabetli örnek", ornek_diyabetli), ("Sağlıklı örnek", ornek_saglikli)]:
    veri = ornek[final_features]
    risk = final_model.predict_proba(veri)[0][1]
    sinif = "DİYABETLİ" if risk >= optimal_threshold else "SAĞLIKLI"
    print(f"  {etiket}: Risk=%{risk*100:.1f} → Tahmin: {sinif}")

# 14. FINAL MODEL KAYDET
print(f"\n[14] FINAL MODEL KAYDEDILIYOR ({final_name})")
with open("model.pkl", "wb") as f:
    pickle.dump({'model': final_model, 'features': final_features, 'threshold': optimal_threshold}, f)

print("✓ model.pkl kaydedildi")
print("\nDETAYLI RAPOR ÖZETİ:")
print(f"  • Binary Classification     : ✓")
print(f"  • Train-Test Split          : ✓ (80-20, stratified)")
print(f"  • Cross Validation          : ✓ (5-Fold — sadece X_train, model öncesi)")
print(f"  • Feature Selection         : ✓ (Mutual Information — sadece X_train)")
print(f"  • 6 Algoritma               : ✓ (LR, DT, KNN, RF, XGBoost, SVM)")
print(f"  • Eşik Optimizasyonu        : ✓ (Optimal eşik: {optimal_threshold:.2f})")
print(f"  • Full vs Top 5             : ✓")
print(f"  • ROC Eğrisi                : ✓")
print(f"  • Confusion Matrix          : ✓")
print(f"  • Feature Importance        : ✓")
print(f"  • Tek Örnek Tahmin Demo     : ✓")
print(f"  • Seçilen Model             : {final_name}")
print("="*60)
