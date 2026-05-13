import pandas as pd
import numpy as np

def birlestir_ve_kaydet(data_yolu, data2_yolu, cikis_adi):
    # 1. Veri setlerini yükle
    data1 = pd.read_csv(data_yolu)
    data2 = pd.read_csv(data2_yolu)

    # --- data DATASET HAZIRLIĞI ---
    # Gerekli sütunları seç (Glucose, Insulin, BMI, Age, Outcome)
    data_hazir = data1[['Glucose', 'Insulin', 'BMI', 'Age', 'Outcome']].copy()
    
    # data'da HbA1c yok, boş sütun ekle
    data_hazir['HbA1c'] = np.nan
    
    # Hedef Dönüşümü: 0 (Sağlıklı) -> 0, 1 (Diyabetik) -> 2
    data_hazir['Target'] = data_hazir['Outcome'].map({0: 0, 1: 2})
    data_hazir = data_hazir.drop(columns=['Outcome'])

    # --- data2 DATASET HAZIRLIĞI ---
    # Gerekli sütunları seç (AGE, HbA1c, BMI, CLASS)
    data2_hazir = data2[['AGE', 'HbA1c', 'BMI', 'CLASS']].copy()
    
    # Sütun ismini data ile eşitle
    data2_hazir.rename(columns={'AGE': 'Age'}, inplace=True)
    
    # data2'da Glucose ve Insulin yok, boş sütun ekle
    data2_hazir['Glucose'] = np.nan
    data2_hazir['Insulin'] = np.nan
    
    # Hedef Dönüşümü: N -> 0, P -> 1, Y -> 2
    data2_hazir['Target'] = data2_hazir['CLASS'].map({'N': 0, 'P': 1, 'Y': 2})
    data2_hazir = data2_hazir.drop(columns=['CLASS'])

    # --- BİRLEŞTİRME ---
    # İki tabloyu alt alta ekle
    birlesik_df = pd.concat([data_hazir, data2_hazir], ignore_index=True)

    # Sütunları daha düzenli bir sıraya diz
    sutun_sirasi = ['Age', 'BMI', 'Glucose', 'Insulin', 'HbA1c', 'Target']
    birlesik_df = birlesik_df[sutun_sirasi]

    # Kaydet
    birlesik_df.to_csv(cikis_adi, index=False)
    print(f"Başarıyla kaydedildi: {cikis_adi}")
    return birlesik_df

# Fonksiyonu çalıştır
# Dosya isimlerinin tam olarak senin bilgisayarındakilerle aynı olduğundan emin ol
birlestir_ve_kaydet('diabetes.csv', 'Dataset of Diabetes .csv', 'final.csv')