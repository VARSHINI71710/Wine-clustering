import streamlit as st
import pandas as pd
import numpy as np
import joblib


scaler = joblib.load("scaler.pkl")       
pca = joblib.load("pca.pkl")             
kmeans = joblib.load("kmeans_model.pkl") 

features = ['Alcohol', 'Malic_Acid', 'Ash', 'Ash_Alcanity', 'Magnesium',
            'Total_Phenols', 'Flavanoids', 'Nonflavanoid_Phenols',
            'Proanthocyanins', 'Color_Intensity', 'Hue', 'OD280', 'Proline']


USER_CREDENTIALS = {
    "admin": "admin123",
    "user1": "pass123"
}


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "cluster" not in st.session_state:
    st.session_state.cluster = None


st.sidebar.header("Login")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
login_button = st.sidebar.button("Login")

if login_button:
    if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
        st.session_state.logged_in = True
        st.success(f"Welcome {username}! 🍷")
    else:
        st.error("❌ Invalid username or password")


if st.session_state.logged_in:
    st.header("Wine Clustering Prediction")
    st.write("Enter 13 wine feature values to predict the cluster using KMeans.")

    cols = st.columns(4)
    input_values = []
    for idx, feature in enumerate(features):
        with cols[idx % 4]:
            value = st.number_input(f"{feature}", value=0.0, step=0.1, format="%.2f", key=feature)
            input_values.append(value)

    if st.button("Predict Cluster"):
        data = pd.DataFrame([input_values], columns=features)
        X_scaled = scaler.transform(data)
        X_pca = pca.transform(X_scaled)
        st.session_state.cluster = kmeans.predict(X_pca)[0]
        st.session_state.prediction_done = True

    if st.session_state.prediction_done:
        st.balloons()
        st.success(f"🎉 This input belongs to Cluster {st.session_state.cluster}")
        cluster_info = {
            0: "Cluster 0: Light wines",
            1: "Cluster 1: Medium wines",
            2: "Cluster 2: Strong wines",
            3: "Cluster 3: Very strong wines"
        }
        st.info(cluster_info.get(st.session_state.cluster, "Cluster information not available"))

else:
    st.info("Please login using the sidebar to access the app.")
