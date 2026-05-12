# Çalıştırma Kılavuzu

## Proje Yapısı

```
diyabet_projesi/
├── model.py                   ← Modeli eğitir, model.pkl üretir
├── app.py                     ← Streamlit arayüzü
├── tek_tikla_calistir.bat     ← Her ikisini otomatik çalıştırır (Windows)
├── diabetes.csv               ← Veri seti
└── model.pkl                  ← Eğitim sonrası oluşur
```

---

## En Kolay Yol — Çift Tıkla Çalıştır

```
tek_tikla_calistir.bat
```

Bu dosyaya çift tıkla. model.py eğitir, sonra Streamlit'i başlatır.

---

## Manuel Çalıştırma

### Adım 1 — Modeli Eğit

```bash
cd C:\Users\UMUT\Desktop\diyabet_projesi
.venv\Scripts\python.exe model.py
```

- Süresi: ~1-2 dakika
- Üretilen dosyalar:
  - `model.pkl` — kaydedilen model
  - `korelasyon.png` — korelasyon matrisi
  - `confusion_matrix.png` — karmaşıklık matrisi
  - `roc_curve.png` — ROC eğrisi

### Adım 2 — Streamlit Arayüzünü Başlat

```bash
.venv\Scripts\streamlit.exe run app.py
```

Tarayıcıda otomatik açılır: http://localhost:8501

---

## Özet (Hızlı Kopya-Yapıştır)

```bash
cd C:\Users\UMUT\Desktop\diyabet_projesi
.venv\Scripts\python.exe model.py
.venv\Scripts\streamlit.exe run app.py
```

---

## Notlar

- `model.py` her çalıştırıldığında model sıfırdan eğitilir ve `model.pkl` güncellenir.
- Streamlit açıkken `model.py` tekrar çalıştırırsan sayfayı yenilemek yeterli.
- `.venv` klasörü yoksa: `python -m venv .venv` komutuyla oluştur, ardından `pip install` yap.
