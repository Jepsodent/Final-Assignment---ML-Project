import streamlit as st
from utils.sidebar import render_sidebar

# Konfigurasi halaman utama
st.set_page_config(page_title="Calories Burn Prediction AI", layout="wide")

# Panggil sidebar navigasi kustom kita (aktif di menu 'Home')
render_sidebar("Home")

# Bagian Header Beranda dengan layout 2 kolom
col_hero_left, col_hero_right = st.columns([3, 2], gap="large")

with col_hero_left:
    st.title("Calories Burn Prediction AI")
    st.subheader("Tugas Akhir Machine Learning — Binus University")
    st.write(
        """
        Selamat datang di aplikasi **Calories Burn Prediction AI**! Aplikasi ini memandu Anda 
        melalui seluruh proses siklus kerja (pipeline) Machine Learning untuk memprediksi 
        jumlah kalori yang terbakar selama aktivitas fisik.
        
        Aplikasi ini dirancang sebagai proyek akhir untuk menunjukkan implementasi pipeline data, 
        eksplorasi korelasi fitur, training model regresi (Linear Regression, Gradient Boosting, 
        dan Random Forest), perbandingan metrik evaluasi model secara real-time, hingga demo prediksi interaktif.
        """
    )
    
    # Menampilkan Nama Anggota Kelompok
    st.markdown("### 👥 Anggota Kelompok:")
    st.markdown(
        """
        1. **Jefferson Gautama Swanto**
        2. **Darrel Nicholas Tandean**
        3. **Timothy Alexandro Sibarani**
        """
    )

with col_hero_right:
    # Menampilkan gambar dengan menyesuaikan ukuran desain
    st.image(
        "images/sport.jpg", 
        caption="Calories Burn Prediction Project", 
        use_container_width=True
    )

st.markdown("---")

# Menggunakan kolom untuk menampilkan tahapan Pipeline secara visual
st.subheader("⚙️ Alur Kerja (ML Pipeline)")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("### 📂 1. Dataset & EDA")
    st.write(
        """
        * **Dataset Overview**: Melihat data mentah, dimensi data, tipe fitur, dan deskripsi kolom.
        * **Exploratory Data Analysis**: Menganalisis korelasi dan distribusi data secara visual melalui grafik interaktif.
        """
    )

with col2:
    st.success("### 🔧 2. Prep & Training")
    st.write(
        """
        * **Preprocessing**: Pembersihan data (encoding gender), membagi data latih/uji, dan scaling data.
        * **Model Training & Evaluation**: Menyesuaikan parameter dan melatih model Linear Regression, Gradient Boosting, dan Random Forest secara real-time.
        """
    )

with col3:
    st.warning("### 🎯 3. Demo & Prediction")
    st.write(
        """
        * **Model Comparison**: Membandingkan performa model menggunakan MAE, MSE, RMSE, dan R² score secara berdampingan.
        * **Interactive Demo**: Melakukan prediksi instan kalori terbakar berdasarkan profil fisik dan aktivitas Anda menggunakan model terbaik.
        """
    )

st.markdown("---")

# Tombol ajakan bertindak (Call to Action) di bawah beranda
col_cta_left, col_cta_right = st.columns([4, 1])
with col_cta_left:
    st.write("Silakan gunakan menu navigasi di sidebar sebelah kiri untuk mulai menjelajahi Dataset.")
with col_cta_right:
    if st.button("Mulai Sekarang! 👉", use_container_width=True):
        st.switch_page("pages/1_Dataset.py")
