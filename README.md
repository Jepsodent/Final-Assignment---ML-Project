# Calories Burn Prediction AI

> **Tugas Akhir Mata Kuliah Machine Learning — Binus University**

Aplikasi berbasis web ini dikembangkan menggunakan **Streamlit** untuk memprediksi jumlah kalori yang terbakar selama aktivitas fisik. Proyek ini menerapkan siklus kerja (pipeline) Machine Learning yang lengkap mulai dari pemuatan dataset, Exploratory Data Analysis (EDA), Preprocessing data, pelatihan dan perbandingan model regresi secara real-time, hingga demo prediksi interaktif.

---

## 👥 Anggota Kelompok

1. **Jefferson Gautama Swanto** (NIM: `2802474476`)
2. **Darrel Nicholas Tandean** (NIM: `2802393081`)
3. **Timothy Alexandro Sibarani** (NIM: `2802475024`)

---

## 📂 Informasi Dataset

Dataset yang digunakan dalam proyek ini diambil dari Kaggle:

- **Link Dataset:** [Kaggle - Calories Burnt Prediction](https://www.kaggle.com/datasets/ruchikakumbhar/calories-burnt-prediction)
- **Deskripsi Fitur:**
  - `User_ID` : ID unik untuk setiap pengguna (dihapus saat preprocessing karena tidak relevan).
  - `Gender` : Jenis kelamin pengguna (`Male` / `Female`).
  - `Age` : Usia pengguna (tahun).
  - `Height` : Tinggi badan (cm).
  - `Weight` : Berat badan (kg).
  - `Duration` : Durasi aktivitas fisik (menit).
  - `Heart_Rate` : Rata-rata detak jantung selama beraktivitas (bpm).
  - `Body_Temp` : Suhu tubuh setelah beraktivitas (°C).
  - `Calories` (**Target**) : Jumlah kalori yang terbakar (kcal).

---

## ⚙️ Alur Kerja Aplikasi (ML Pipeline)

### 1. Dataset & Exploratory Data Analysis (EDA)

- Memuat dan menampilkan dataset mentah, tipe data, serta nilai statistik dasar.
- Visualisasi korelasi fitur menggunakan Heatmap untuk melihat hubungan antar variabel independen terhadap target (`Calories`).
- Visualisasi distribusi data individual (Histogram) dan visualisasi relasi dua fitur (Scatter Plot).

### 2. Preprocessing Data

- Mengonversi fitur kategorik `Gender` menjadi numerik (0 untuk Male, 1 untuk Female).
- Membagi dataset menjadi data latih (_Training Set_) dan data uji (_Test Set_) secara dinamis.
- Melakukan penskalaan fitur (_Feature Scaling_) menggunakan `StandardScaler` atau `MinMaxScaler` untuk meningkatkan stabilitas model regresi.

### 3. Model Training & Comparison (Eksperimen Interaktif)

Melatih **3 model regresi** secara bersamaan dengan parameter yang dapat disesuaikan langsung oleh pengguna melalui antarmuka web:

- **Linear Regression** (Parameter: `Fit Intercept`).
- **Gradient Boosting Regressor** (Parameter: `Learning Rate` dan `n_estimators`).
- **Random Forest Regressor** (Parameter: `n_estimators`).

### 4. Evaluasi Performa Model

- Membandingkan hasil prediksi ketiga model secara berdampingan menggunakan empat metrik regresi standar:
  - **MAE** (_Mean Absolute Error_)
  - **MSE** (_Mean Squared Error_)
  - **RMSE** (_Root Mean Squared Error_)
  - **R² Score** (_Akurasi/Koefisien Determinasi_)
- Menyorot secara otomatis model terbaik (berdasarkan nilai R² tertinggi) dengan simbol ⭐ dan tag **(Terbaik)**.
- Menampilkan kontribusi fitur (_Feature Importance_) dari model terbaik untuk memberikan wawasan biologi/fisiologis.

### 5. Interactive Demo (Prediksi Real-Time)

- Formulir dinamis untuk menginput profil fisik (Usia, Tinggi, Berat, Gender) dan parameter latihan (Durasi, Detak Jantung, Suhu Tubuh).
- Menghitung dan menampilkan estimasi kalori terbakar secara instan menggunakan model terbaik yang telah terpilih secara otomatis.
- Memberikan status kategori intensitas latihan (Ringan, Sedang, Sangat Tinggi) dan tips kesehatan pendukung secara personal.

---

## 🚀 Instalasi & Cara Menjalankan Aplikasi

Aplikasi ini dapat dijalankan dengan dua cara alternatif: **Secara Lokal (Python & Pip)** atau **Menggunakan Docker (Kontainer)**.

### Alternatif A: Menjalankan Secara Lokal (Python & Pip)

#### 1. Clone Repositori
```bash
git clone https://github.com/Jepsodent/Final-Assignment---ML-Project.git
cd Final-Assignment---ML-Project
```

#### 2. Setup Virtual Environment (Direkomendasikan)
* **Windows:**
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```
* **macOS/Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

#### 3. Install Dependensi
```bash
pip install -r requirements.txt
```

#### 4. Jalankan Aplikasi
```bash
streamlit run app.py
```

---

### Alternatif B: Menjalankan Menggunakan Docker (Lebih Mudah & Praktis)

Pastikan Anda sudah menginstal [Docker](https://www.docker.com/) di komputer Anda.

#### 1. Build Docker Image
Buka terminal di direktori proyek dan jalankan perintah untuk membangun image:
```bash
docker build -t calories-prediction-app .
```

#### 2. Jalankan Container
Jalankan kontainer dengan memetakan port `8501`:
```bash
docker run -p 8501:8501 calories-prediction-app
```

Setelah berjalan, buka browser dan akses aplikasi melalui alamat: **`http://localhost:8501`**.

---

## 🛠️ Struktur Direktori Proyek

```text
Final-Assignment---ML-Project/
│
├── app.py                      # Halaman Utama (Beranda Aplikasi)
├── requirements.txt            # Daftar pustaka/dependensi Python
├── Dockerfile                  # Konfigurasi containerisasi Docker
│
├── pages/                      # Halaman Navigasi Aplikasi
│   ├── 1_Dataset.py            # Menu Dataset Overview
│   ├── 2_EDA.py                # Menu Analisis EDA Visual
│   ├── 3_Preprocessing.py      # Menu Preprocessing
│   ├── 4_Training-Evaluation.py# Menu Pelatihan & Perbandingan Model (LR, GB, RF)
│   └── 6_Demo.py               # Menu Demo Prediksi Interaktif
│
├── utils/                      # Fungsi Utilitas Pendukung
│   ├── data_access.py          # Logika Cache dan Pemuatan Data
│   └── sidebar.py              # Navigasi Sidebar Kustom
│
└── images/                     # Folder Media Gambar
    └── sport.jpg               # Gambar Pendukung Beranda
```
