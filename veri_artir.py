import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

print("="*60)
print("VERİ ARTIRMA (Synthetic Data Generation)")
print("="*60)

# 1. Orijinal veriyi yükle
df_original = pd.read_csv("data/diabetes.csv")
print(f"\n[1] Orijinal veri: {df_original.shape[0]} satır")
print(f"    Sınıf dağılımı: {df_original['Outcome'].value_counts().to_dict()}")

# 2. Metadata tanımla
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_original)
metadata.update_column('Outcome', sdtype='categorical')

# 3. GaussianCopula ile sentetik veri üret (sınıf bazında ayrı ayrı)
print("\n[2] GaussianCopula modeli eğitiliyor...")

hedef_toplam = 2000  # Ulaşmak istediğimiz toplam satır sayısı

# Orijinal sınıf oranını koru
oran_0 = (df_original['Outcome'] == 0).sum() / len(df_original)
oran_1 = 1 - oran_0

hedef_0 = int(hedef_toplam * oran_0)
hedef_1 = int(hedef_toplam * oran_1)

uretilecek_0 = max(0, hedef_0 - (df_original['Outcome'] == 0).sum())
uretilecek_1 = max(0, hedef_1 - (df_original['Outcome'] == 1).sum())

print(f"    Hedef: {hedef_toplam} satır (Class 0: {hedef_0}, Class 1: {hedef_1})")
print(f"    Üretilecek: Class 0: {uretilecek_0}, Class 1: {uretilecek_1}")

sentetik_parcalar = []

for sinif, uretilecek in [(0, uretilecek_0), (1, uretilecek_1)]:
    if uretilecek == 0:
        continue
    df_sinif = df_original[df_original['Outcome'] == sinif].copy()
    meta_sinif = SingleTableMetadata()
    meta_sinif.detect_from_dataframe(df_sinif)

    synth = GaussianCopulaSynthesizer(meta_sinif)
    synth.fit(df_sinif)
    df_sentetik = synth.sample(num_rows=uretilecek)
    df_sentetik['Outcome'] = sinif
    sentetik_parcalar.append(df_sentetik)
    print(f"    Class {sinif}: {uretilecek} sentetik satır üretildi")

# 4. Sentetik veriyi temizle
df_sentetik_tum = pd.concat(sentetik_parcalar, ignore_index=True)

# Negatif değerleri temizle (tıbbi veriler negatif olamaz)
for col in df_sentetik_tum.columns:
    if col != 'Outcome':
        df_sentetik_tum[col] = df_sentetik_tum[col].clip(lower=0)

df_sentetik_tum['Outcome'] = df_sentetik_tum['Outcome'].astype(int)

# Orijinal + sentetik birleştir
df_augmented = pd.concat([df_original, df_sentetik_tum], ignore_index=True)

print(f"\n[3] Sentetik veri: {len(df_sentetik_tum)} satır")
print(f"    Sınıf dağılımı: {df_sentetik_tum['Outcome'].value_counts().sort_index().to_dict()}")
print(f"\n    Birleşik toplam: {df_augmented.shape[0]} satır")

# 5. Dağılım karşılaştırma grafiği
print("\n[4] Dağılım karşılaştırma grafiği oluşturuluyor...")
features = [c for c in df_original.columns if c != 'Outcome']
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, col in enumerate(features):
    axes[i].hist(df_original[col], bins=20, alpha=0.6, label='Orijinal', color='#3498db', density=True)
    axes[i].hist(df_sentetik_tum[col], bins=20, alpha=0.6, label='Sentetik', color='#e74c3c', density=True)
    axes[i].set_title(col)
    axes[i].legend(fontsize=8)

plt.suptitle("Orijinal vs Sentetik Veri Dağılımı", fontsize=13)
plt.tight_layout()
plt.savefig("output/dagilim_karsilastirma.png")
plt.close()
print("    output/dagilim_karsilastirma.png kaydedildi")

# 6. Kaydet — sentetik ayrı, birleşik ayrı
df_sentetik_tum.to_csv("data/diabetes_synthetic.csv", index=False)
df_augmented.to_csv("data/diabetes_augmented.csv", index=False)

print(f"\n[5] Kaydedilen dosyalar:")
print(f"    data/diabetes_synthetic.csv  : sadece sentetik ({len(df_sentetik_tum)} satir)")
print(f"    data/diabetes_augmented.csv  : orijinal + sentetik ({df_augmented.shape[0]} satir)")

print("\n" + "="*60)
print("ÖZET")
print("="*60)
print(f"  Orijinal       : {df_original.shape[0]} satır")
print(f"  Sentetik       : {len(df_sentetik_tum)} satır")
print(f"  Birleşik toplam: {df_augmented.shape[0]} satır")
print(f"  Artış          : %{((df_augmented.shape[0]/df_original.shape[0])-1)*100:.0f}")
print("="*60)
