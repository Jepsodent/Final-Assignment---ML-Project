from utils.sidebar import render_sidebar
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Interactive Demo", layout="wide")

st.title(" Interactive Prediction Demo")
st.write("Masukkan data fisik dan intensitas latihan Anda untuk memprediksi jumlah kalori yang terbakar secara real-time.")

render_sidebar("Demo")

# Validasi apakah model dan scaler sudah siap
if "model" not in st.session_state or "scaler" not in st.session_state:
    st.warning("⚠️ Model atau Scaler belum dilatih. Silakan selesaikan tahap Preprocessing & Training terlebih dahulu!")
    # Tombol shortcut ke halaman training
    if st.button("Latih Model Sekarang"):
        st.switch_page("pages/4_Training-Evaluation.py")
    st.stop()

# Ambil model, scaler, dan X_train dari session state
model = st.session_state["model"]
model_name = st.session_state["model_name"]
scaler = st.session_state["scaler"]
X_train = st.session_state["X_train"]

# Ambil nilai rata-rata dari data asli untuk nilai default formulir (agar realistis)
avg_age = int(X_train["Age"].mean()) if "Age" in X_train.columns else 27
avg_height = float(X_train["Height"].mean()) if "Height" in X_train.columns else 174.0
avg_weight = float(X_train["Weight"].mean()) if "Weight" in X_train.columns else 74.0
avg_duration = float(X_train["Duration"].mean()) if "Duration" in X_train.columns else 15.0
avg_heart_rate = float(X_train["Heart_Rate"].mean()) if "Heart_Rate" in X_train.columns else 95.0
avg_body_temp = float(X_train["Body_Temp"].mean()) if "Body_Temp" in X_train.columns else 40.0

# Layout halaman: Kiri untuk input form, Kanan untuk hasil visualisasi prediksi
col_form, col_result = st.columns([1, 1], gap="large")

with col_form:
    st.subheader(" Input Parameter Latihan")
    
    # Input Gender dengan pemetaan string
    gender_input = st.selectbox(
        "Gender (Jenis Kelamin)",
        options=["Male (Laki-laki)", "Female (Perempuan)"],
        index=0
    )
    gender_val = 0 if gender_input.startswith("Male") else 1
    
    # Input parameter lainnya dengan rentang dan default rata-rata
    age = st.slider("Age (Usia)", 1, 100, avg_age, 1)
    
    col_wh = st.columns(2)
    with col_wh[0]:
        height = st.number_input(
            "Height (Tinggi Badan - cm)", 
            min_value=50.0, 
            max_value=250.0, 
            value=round(avg_height, 1), 
            step=1.0
        )
    with col_wh[1]:
        weight = st.number_input(
            "Weight (Berat Badan - kg)", 
            min_value=10.0, 
            max_value=300.0, 
            value=round(avg_weight, 1), 
            step=1.0
        )
        
    duration = st.slider(
        "Duration (Durasi Latihan - menit)", 
        min_value=1.0, 
        max_value=60.0, 
        value=round(avg_duration, 1), 
        step=1.0
    )
    
    col_hr_temp = st.columns(2)
    with col_hr_temp[0]:
        heart_rate = st.number_input(
            "Heart Rate (Detak Jantung - bpm)", 
            min_value=40.0, 
            max_value=220.0, 
            value=round(avg_heart_rate, 1), 
            step=1.0
        )
    with col_hr_temp[1]:
        body_temp = st.number_input(
            "Body Temperature (Suhu Tubuh - °C)", 
            min_value=35.0, 
            max_value=43.0, 
            value=round(avg_body_temp, 1), 
            step=0.1
        )

# Jalankan prediksi setiap kali ada perubahan input (reactive secara instan!)
input_dict = {
    "Gender": [gender_val],
    "Age": [age],
    "Height": [height],
    "Weight": [weight],
    "Duration": [duration],
    "Heart_Rate": [heart_rate],
    "Body_Temp": [body_temp]
}

# Memastikan urutan kolom input sama persis dengan X_train
input_df = pd.DataFrame(input_dict)[X_train.columns]

# Lakukan scaling & prediksi
input_scaled = scaler.transform(input_df)
pred_calories = model.predict(input_scaled)[0]

# Pembatasan prediksi agar tidak bernilai negatif
pred_calories = max(0.0, pred_calories)

with col_result:
    st.subheader("🎯 Hasil Analisis Prediksi")
    st.success(f"Model Aktif: **{model_name}**")
    
    # Render Card Hasil Prediksi Premium menggunakan HTML & CSS
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border-left: 6px solid #ff7f50;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
            margin-bottom: 25px;
        ">
            <div style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.1em; color: #ff7f50; font-weight: 700; margin-bottom: 8px;">
                🔥 Estimasi Kalori Terbakar
            </div>
            <div style="font-size: 3.5rem; font-weight: 900; color: #ffffff; line-height: 1; margin-bottom: 12px;">
                {pred_calories:.2f} <span style="font-size: 1.5rem; font-weight: 400; color: #94a3b8;">kcal</span>
            </div>
            <div style="color: #94a3b8; font-size: 0.95rem; line-height: 1.5;">
                Prediksi kalori ini disesuaikan dengan profil fisiologis Anda (Usia: {age} tahun, Berat: {weight} kg) 
                dan intensitas olahraga Anda selama {duration:.1f} menit.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Indikator Intensitas Latihan Sederhana
    st.write("📊 **Rasio Intensitas Latihan:**")
    if duration > 0:
        calories_per_minute = pred_calories / duration
        st.write(f"Rata-rata pembakaran kalori: **{calories_per_minute:.2f} kcal / menit**")
        
        # Nilai acuan: 15 kcal/menit dianggap intensitas sangat tinggi
        intensity_ratio = min(1.0, calories_per_minute / 15.0)
        
        if intensity_ratio > 0.7:
            st.error("🔥 **Kategori Intensitas: Sangat Tinggi (HIIT / Kardio Berat)**")
        elif intensity_ratio >= 0.4:
            st.warning("⚡ **Kategori Intensitas: Sedang (Jogging / Bersepeda)**")
        else:
            st.info("🚶 **Kategori Intensitas: Ringan (Jalan Santai / Peregangan)**")
            
        st.progress(intensity_ratio)
    
    st.markdown("---")
    st.markdown("### 💡 Tips Kesehatan & Latihan")
    if duration >= 30:
        st.info("Durasi latihan Anda cukup panjang. Pastikan Anda minum air yang cukup untuk mencegah dehidrasi.")
    else:
        st.info("Latihan singkat tetap sangat bermanfaat! Cobalah untuk konsisten berolahraga minimal 15-30 menit setiap hari.")

    if heart_rate > 140:
        st.error("Detak jantung Anda berada di zona pembakaran lemak yang intensif. Kurangi intensitas jika Anda merasa pusing atau terlalu lelah.")
