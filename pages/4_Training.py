import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


st.title("Model Training")

if "n_estimators" not in st.session_state:
    st.session_state["n_estimators"] = None

if "X_train_scaled" not in st.session_state:
    st.warning("Silahkan lakukan preprocessing terlebih dahulu!")
    st.stop()


model_choice = st.selectbox(
    "Pilih model",
    ["Linear Regression", "Random Forest"]
)

if model_choice == "Random Forest":
    n_estimators = st.slider("Jumlah Trees: ", 50, 200, 100)

if st.button("Train Model"):
    X_train =st.session_state["X_train_scaled"]
    y_train = st.session_state["y_train"]
    if model_choice == "Linear Regression":
        model = LinearRegression()
        st.session_state["n_estimators"] = None
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, max_depth=15)
        st.session_state["n_estimators"] = n_estimators
    with st.spinner("Training model..."):
        model.fit(X_train, y_train)
    
    st.success("Model berhasil di training!")
    st.session_state["model"] = model
    st.session_state["model_name"] = model_choice


if "model" in st.session_state:
    st.subheader("Model Info")
    if st.session_state['model_name'] == "Random Forest":
        st.write(f"Model yang digunakan: **{st.session_state['model_name']}** dengan {st.session_state["n_estimators"]} trees")
    else:
        st.write(f"Model yang digunakan: **{st.session_state['model_name']}**")
    if st.button("Lanjut ke Evaluation"):
        st.switch_page("pages/5_Evaluation.py")
