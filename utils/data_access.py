from pathlib import Path
import pandas as pd
import streamlit as st

DATASET_SESSION_KEY = "calories_dataset"
DATASET_SOURCE_KEY = "calories_dataset_source"


DEFAULT_DATA_PATHS = (Path("data/calories.csv"), Path("calories.csv"))

COLUMN_ALIASES = {
    "User_Id": "User_ID",
    "User id": "User_ID",
    "User ID": "User_ID",
    "Heart_rate": "Heart_Rate",
    "Heart rate": "Heart_Rate",
    "HeartRate": "Heart_Rate",
    "Body_temp": "Body_Temp",
    "Body temp": "Body_Temp",
    "BodyTemperature": "Body_Temp",
}

def normalize_calories_columns(df):
    #"Nyamain nama column dan buang identitas user_id"
    df = df.copy()

    rename_map = {column: COLUMN_ALIASES.get(column, column) for column in df.columns}

    if "User_ID" in df.columns: 
        df = df.drop(columns=["User_ID"])

    return df

def load_calories_dataframe():
    "Cek data dari memori , atau file lokal csv"
    if DATASET_SESSION_KEY in st.session_state: 
        return st.session_state[DATASET_SESSION_KEY].copy() , st.session_state.get(DATASET_SOURCE_KEY, "session state") 

    for path in DEFAULT_DATA_PATHS: 
        if path.exists():
            df_cleaned = normalize_calories_columns(pd.read_csv(path))
            st.session_state[DATASET_SESSION_KEY] = df_cleaned
            st.session_state[DATASET_SOURCE_KEY] = str(path)
            return df_cleaned, str(path)
    return None, None

def store_calories_dataframe(df, source="uploadedfile"):
    normalized_df = normalize_calories_columns(df)
    st.session_state[DATASET_SESSION_KEY] = normalized_df
    st.session_state[DATASET_SOURCE_KEY] = source
    return normalized_df