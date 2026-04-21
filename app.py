import streamlit as st
import pickle
import pandas as pd

with open("model.pkl", "rb") as f:
    saved = pickle.load(f)

model     = saved['model']
features  = saved['features']
threshold = saved.get('threshold', 0.5)

st.title("Diyabet Risk Tahmin Sistemi")
st.write("Hasta bilgilerini girin:")

pregnancies     = st.slider("Hamilelik sayısı", 0, 17, 1)
glucose         = st.slider("Glikoz (mg/dL)", 50, 200, 120)
blood_pressure  = st.slider("Kan Basıncı (mm Hg)", 40, 130, 70)
skin_thickness  = st.slider("Deri Kalınlığı (mm)", 5, 60, 20)
insulin         = st.slider("İnsülin (mu U/ml)", 10, 350, 80)
bmi             = st.slider("BMI", 15.0, 55.0, 25.0)
dpf             = st.slider("Diyabet Soygeçmişi (0-2.5)", 0.0, 2.5, 0.3)
age             = st.slider("Yaş", 18, 80, 30)

if st.button("Riski Hesapla"):
    tum_veri = {
        'Pregnancies':             pregnancies,
        'Glucose':                 glucose,
        'BloodPressure':           blood_pressure,
        'SkinThickness':           skin_thickness,
        'Insulin':                 insulin,
        'BMI':                     bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age':                     age
    }

    veri = pd.DataFrame([{k: tum_veri[k] for k in features}])
    risk = model.predict_proba(veri)[0][1]

    st.metric("Diyabet Riski", f"%{risk*100:.1f}")
    if risk >= threshold:
        st.error("Yüksek risk — bir doktora başvurun.")
    else:
        st.success("Düşük risk.")
