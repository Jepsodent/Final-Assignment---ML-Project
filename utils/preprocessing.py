import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def clean_data(df):
    df = df.copy()

    if "User_ID" in df.columns:
        df = df.drop(columns=["User_ID"])

    df["Gender"] = df["Gender"].map({"male": 0, "female": 1})

    return df

def preprocess_data(df, scaler_type="standard", ts=0.2):
    df= clean_data(df)
    X = df.drop("Calories", axis=1)
    y = df["Calories"]

    X_train, X_test,  y_train, y_test = train_test_split(X,y,random_state=42,test_size=ts)

    if scaler_type == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled =scaler.transform(X_test)
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler