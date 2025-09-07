import gradio as gr
import pandas as pd
import numpy as np
import joblib

scaler = joblib.load("scaler.pkl")       
pca = joblib.load("pca.pkl")             
kmeans = joblib.load("kmeans_model.pkl") 

features = ['Alcohol', 'Malic_Acid', 'Ash', 'Ash_Alcanity', 'Magnesium',
            'Total_Phenols', 'Flavanoids', 'Nonflavanoid_Phenols',
            'Proanthocyanins', 'Color_Intensity', 'Hue', 'OD280', 'Proline']

def predict_cluster(*inputs):
    data = pd.DataFrame([inputs], columns=features)
    
    X_scaled = scaler.transform(data)
    
    X_pca = pca.transform(X_scaled)
    
    cluster = kmeans.predict(X_pca)[0]
    
    return f"This input belongs to Cluster {cluster}"

inputs = [gr.Number(label=f) for f in features]

iface = gr.Interface(
    fn=predict_cluster,
    inputs=inputs,
    outputs="text",
    title="Wine Clustering Prediction",
    description="Enter 13 wine feature values to predict the cluster using KMeans."
)

iface.launch()
