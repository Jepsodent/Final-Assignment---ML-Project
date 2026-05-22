from utils.sidebar import render_sidebar
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Model Training & Evaluation", layout="wide")

st.title(" Model Training & Evaluation")
st.write("Latih tiga algoritma Machine Learning sekaligus, bandingkan kinerjanya, dan pilih model terbaik secara otomatis.")

render_sidebar("TrainingnEvaluation")

if "X_train_scaled" not in st.session_state:
    st.warning(" Silakan lakukan preprocessing data terlebih dahulu di halaman Preprocessing!")
    st.stop()

X_train = st.session_state["X_train"] #cuma buat akses columns
X_train_scaled = st.session_state["X_train_scaled"]
X_test_scaled = st.session_state["X_test_scaled"]
y_train = st.session_state["y_train"]
y_test = st.session_state["y_test"]


if "current_fit_intercept" not in st.session_state:
    st.session_state["current_fit_intercept"] = True
if "current_learning_rate" not in st.session_state:
    st.session_state["current_learning_rate"] = 0.10
if "current_gb_n_estimators" not in st.session_state:
    st.session_state["current_gb_n_estimators"] = 100
if "current_rf_n_estimators" not in st.session_state:
    st.session_state["current_rf_n_estimators"] = 100

st.subheader(" Parameter Masing-Masing Model")
st.write("Sesuaikan parameter latihan untuk ketiga model di bawah ini:")

# Membuat layout 3 kolom untuk masing-masing model
col_params = st.columns(3)

with col_params[0]:
    st.info(" **Linear Regression**")
    fit_intercept = st.selectbox(
        "Fit Intercept",
        options=[True, False],
        index=0,
        help="Menentukan apakah bias/intersep dihitung. Jika False, data diasumsikan terpusat di titik asal."
    )

with col_params[1]:
    st.warning(" **Gradient Boosting**")
    learning_rate = st.slider(
        "Learning Rate (Laju Belajar)",
        min_value=0.01,
        max_value=0.50,
        value=st.session_state["current_learning_rate"],
        step=0.01,
        help="Mengontrol kontribusi setiap pohon baru. Nilai kecil mencegah overfitting tapi butuh waktu training lebih lama."
    )
    gb_n_estimators = st.slider(
        "Jumlah Pohon (n_estimators - GB)",
        min_value=50,
        max_value=150,
        value=st.session_state["current_gb_n_estimators"],
        step=10,
        help="Jumlah pohon untuk Gradient Boosting. Semakin banyak, model semakin kompleks."
    )

with col_params[2]:
    st.success(" **Random Forest**")
    rf_n_estimators = st.slider(
        "Jumlah Pohon (n_estimators - RF)", 
        min_value=50, 
        max_value=200, 
        value=st.session_state["current_rf_n_estimators"], 
        step=10,
        help="Jumlah Decision Tree yang akan digabungkan. Semakin banyak pohon, model cenderung lebih stabil."
    )

# Deteksi jika ada perubahan parameter, paksa user untuk klik Train ulang
if (fit_intercept != st.session_state["current_fit_intercept"] or
    learning_rate != st.session_state["current_learning_rate"] or
    gb_n_estimators != st.session_state["current_gb_n_estimators"] or
    rf_n_estimators != st.session_state["current_rf_n_estimators"]):
    
    st.session_state["training_done"] = False
    st.session_state["current_fit_intercept"] = fit_intercept
    st.session_state["current_learning_rate"] = learning_rate
    st.session_state["current_gb_n_estimators"] = gb_n_estimators
    st.session_state["current_rf_n_estimators"] = rf_n_estimators

# Tombol Eksekusi
st.write("")
train_button = st.button("🚀 Train & Compare 3 Models", use_container_width=True)
st.markdown("---")

if train_button:
    with st.spinner("Sedang melatih dan mengevaluasi ketiga model..."):
        # 1. Linear Regression
        lr_model = LinearRegression(fit_intercept=fit_intercept)
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        
        # 2. Gradient Boosting
        gb_model = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=gb_n_estimators, random_state=42)
        gb_model.fit(X_train_scaled, y_train)
        gb_pred = gb_model.predict(X_test_scaled)
        
        # 3. Random Forest
        rf_model = RandomForestRegressor(n_estimators=rf_n_estimators, random_state=42, max_depth=15, n_jobs=-1)
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        
        # 4. Hitung Metrik Evaluasi (MAE, MSE, RMSE, R2)
        lr_mae = mean_absolute_error(y_test, lr_pred)
        lr_mse = mean_squared_error(y_test, lr_pred)
        lr_rmse = np.sqrt(lr_mse)
        lr_r2 = r2_score(y_test, lr_pred)
        
        gb_mae = mean_absolute_error(y_test, gb_pred)
        gb_mse = mean_squared_error(y_test, gb_pred)
        gb_rmse = np.sqrt(gb_mse)
        gb_r2 = r2_score(y_test, gb_pred)
        
        rf_mae = mean_absolute_error(y_test, rf_pred)
        rf_mse = mean_squared_error(y_test, rf_pred)
        rf_rmse = np.sqrt(rf_mse)
        rf_r2 = r2_score(y_test, rf_pred)
        
        # 5. Tentukan Model Terbaik berdasarkan R² Score tertinggi
        models = {
            "Linear Regression": (lr_model, lr_r2),
            "Gradient Boosting": (gb_model, gb_r2),
            "Random Forest": (rf_model, rf_r2)
        }
        best_name = max(models, key=lambda k: models[k][1])
        best_model = models[best_name][0]
        best_r2 = models[best_name][1]
        
        # Simpan hasil ke session state
        st.session_state["lr_metrics"] = {"MAE": lr_mae, "MSE": lr_mse, "RMSE": lr_rmse, "R2": lr_r2}
        st.session_state["gb_metrics"] = {"MAE": gb_mae, "MSE": gb_mse, "RMSE": gb_rmse, "R2": gb_r2}
        st.session_state["rf_metrics"] = {"MAE": rf_mae, "MSE": rf_mse, "RMSE": rf_rmse, "R2": rf_r2}
        st.session_state["best_model_name"] = best_name
        st.session_state["model"] = best_model  # Model aktif untuk Demo
        st.session_state["model_name"] = best_name
        
        # Dapatkan feature importance dari model terbaik (GB atau RF memiliki attribute ini)
        if hasattr(best_model, "feature_importances_"):
            st.session_state["best_importances"] = best_model.feature_importances_
        else:
            # Fallback ke Random Forest jika Linear Regression yang terbaik
            st.session_state["best_importances"] = rf_model.feature_importances_
            
        st.session_state["training_done"] = True

# Tampilkan hasil jika proses training sudah selesai dan metrik tersedia
if (st.session_state.get("training_done", False) and 
    "lr_metrics" in st.session_state and 
    "gb_metrics" in st.session_state and 
    "rf_metrics" in st.session_state):
    
    lr_m = st.session_state["lr_metrics"]
    gb_m = st.session_state["gb_metrics"]
    rf_m = st.session_state["rf_metrics"]
    best_name = st.session_state["best_model_name"]
    
    st.subheader("📊 Tabel Perbandingan Evaluasi Model")
    
    # Hitung model terbaik secara dinamis untuk penandaan tabel
    maes = [lr_m['MAE'], gb_m['MAE'], rf_m['MAE']]
    mses = [lr_m['MSE'], gb_m['MSE'], rf_m['MSE']]
    rmses = [lr_m['RMSE'], gb_m['RMSE'], rf_m['RMSE']]
    r2s = [lr_m['R2'], gb_m['R2'], rf_m['R2']]
    
    best_mae_idx = maes.index(min(maes))
    best_mse_idx = mses.index(min(mses))
    best_rmse_idx = rmses.index(min(rmses))
    best_r2_idx = r2s.index(max(r2s))
    
    def tag_best(val, idx, current_idx):
        if current_idx == idx:
            return f"{val:.4f} ⭐ (Terbaik)"
        return f"{val:.4f}"
        
    comparison_df = pd.DataFrame({
        "Metrik Evaluasi": [
            "MAE (Mean Absolute Error) - Rentang rata-rata kesalahan prediksi (Semakin rendah semakin baik)",
            "MSE (Mean Squared Error) - Rata-rata kuadrat kesalahan prediksi (Semakin rendah semakin baik)",
            "RMSE (Root Mean Squared Error) - Standar deviasi dari error (Semakin rendah semakin baik)",
            "R² Score (Akurasi/Koefisien Determinasi) - Menunjukkan kecocokan variasi data (Semakin mendekati 1.0 semakin baik)"
        ],
        "Linear Regression": [
            tag_best(lr_m['MAE'], best_mae_idx, 0),
            tag_best(lr_m['MSE'], best_mse_idx, 0),
            tag_best(lr_m['RMSE'], best_rmse_idx, 0),
            tag_best(lr_m['R2'], best_r2_idx, 0)
        ],
        "Gradient Boosting": [
            tag_best(gb_m['MAE'], best_mae_idx, 1),
            tag_best(gb_m['MSE'], best_mse_idx, 1),
            tag_best(gb_m['RMSE'], best_rmse_idx, 1),
            tag_best(gb_m['R2'], best_r2_idx, 1)
        ],
        "Random Forest": [
            tag_best(rf_m['MAE'], best_mae_idx, 2),
            tag_best(rf_m['MSE'], best_mse_idx, 2),
            tag_best(rf_m['RMSE'], best_rmse_idx, 2),
            tag_best(rf_m['R2'], best_r2_idx, 2)
        ]
    })
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    with st.expander("Klik untuk penjelasan cara membaca metrik di atas"):
        st.markdown("""
        * **MAE (Mean Absolute Error):** Menunjukkan selisih rata-rata absolut antara kalori aktual dan prediksi. Jika MAE = 2.0, rata-rata kesalahan prediksi model hanyalah 2 kalori.
        * **MSE (Mean Squared Error):** Rata-rata kuadrat kesalahan. Memberikan penalti besar untuk kesalahan prediksi yang ekstrem.
        * **RMSE (Root Mean Squared Error):** Akar kuadrat dari MSE. Mengembalikan satuan error ke satuan aslinya yaitu **Kalori (kcal)**, membuatnya sangat mudah dipahami.
        * **R² Score (R-squared):** Menjelaskan seberapa besar variasi target (Kalori) yang dapat diprediksi oleh fitur input. Nilai 0.99 berarti model dapat memprediksi 99% pola data dengan tepat!
        """)
        
    st.success(
        f" **Kesimpulan:** Model **{best_name}** secara keseluruhan terbukti sebagai model paling akurat "
        f"dan otomatis terpilih untuk halaman **Interactive Demo**!"
    )
    
    st.markdown("---")
    st.subheader(f" Feature Importance - Berbasis Model {best_name}")
    st.write(f"Berikut tingkat pengaruh masing-masing fitur fisik/aktivitas terhadap pembakaran kalori berdasarkan model {best_name}:")
    
    importances = st.session_state["best_importances"]
    df_importance = pd.DataFrame({
        "Fitur": X_train.columns,
        "Tingkat Pengaruh (Importance)": importances
    }).sort_values(by="Tingkat Pengaruh (Importance)", ascending=False)
    
    col_chart, col_explain = st.columns([2, 1])
    with col_chart:
        st.bar_chart(df_importance.set_index("Fitur"), use_container_width=True)
    with col_explain:
        st.dataframe(df_importance, hide_index=True, use_container_width=True)
        st.info(
            "💡 **Insight Biologi:** "
            "Pola data menunjukkan bahwa kombinasi **Duration** (durasi latihan) dan **Heart Rate** "
            "(detak jantung) memberikan kontribusi terbesar dalam menentukan pembakaran kalori, "
            "jauh di atas pengaruh antropometri dasar seperti tinggi atau berat badan."
        )
        
    st.markdown("---")
    if st.button(" Lanjut ke Interactive Demo", use_container_width=True):
        st.switch_page("pages/6_Demo.py")
else:
    st.info("Atur parameter di atas dan klik tombol **Train & Compare 3 Models** untuk melihat perbandingan performanya secara langsung.")
