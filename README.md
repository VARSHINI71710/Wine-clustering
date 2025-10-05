🍷 Wine Clustering App

This application leverages the Wine dataset from Kaggle to perform clustering using various machine learning algorithms.

Try this app in streamlit-https://wine-clustering-0-3.streamlit.app/

Try this app in gradio-https://huggingface.co/spaces/Varshini-07/Wine_clustering

📊 Dataset

Source: Wine Quality Dataset - Kaggle

Description: The dataset describes the amount of various chemicals present in wine and their effect on its quality. It can be viewed as classification or regression tasks.

Features:

Alcohol

Malic_Acid

Ash

Ash_Alcanity

Magnesium

Total_Phenols

Flavanoids

Nonflavanoid_Phenols

Proanthocyanins

Color_Intensity

Hue

OD280

Proline

🧪 Algorithms Used

DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

Identifies clusters based on density, useful for data with noise and outliers.

Hierarchical Clustering

Builds a tree of clusters, allowing for a multi-level clustering structure.

KMeans Clustering

Partitions data into k clusters, minimizing within-cluster variance.

🔐 Login Page

The app includes a login system using Streamlit sidebar.

Default Users:

Username	Password

Username	admin123

password	pass123

Username and password in gradio:

Username  admin

password	password123

Users must log in via the sidebar to access the clustering prediction page.

🗂 Project Structure

wine_clustering_app/

│
├── app.py                  # Main Streamlit app

├── requirements.txt        # Python dependencies

├── scaler.pkl              # Saved StandardScaler object

├── pca.pkl                 # Saved PCA object

├── kmeans_model.pkl        # Trained KMeans model

├── dbscan_model.pkl        # Trained DBSCAN model

├── hierarchical_model.pkl  # Trained Hierarchical model

├── README.md               # Project documentation

└── dataset.csv             # Original Wine dataset (optional)

📦 Installation

Clone the repository:

git clone https://github.com/yourusername/wine_clustering_app.git

cd wine_clustering_app


Install dependencies:

pip install -r requirements.txt


requirements.txt example:

streamlit
pandas
numpy
scikit-learn
joblib

🚀 Usage

Run the Streamlit app:

streamlit run app.py

Enter the 13 wine features in the input fields.

Click "Predict Cluster" → the app shows the predicted cluster number and description.

🧪 Sample Input
Feature	Value
Alcohol	13.0
Malic_Acid	2.0
Ash	2.4
Ash_Alcanity	18.0
Magnesium	100.0
Total_Phenols	2.5
Flavanoids	2.0
Nonflavanoid_Phenols	0.3
Proanthocyanins	1.5
Color_Intensity	5.0
Hue	1.0
OD280	3.0
Proline	750.0

🎯 Expected Output

🎉 This input belongs to Cluster 3

Cluster 3: Very strong wines

Clustering graph:

<img width="546" height="435" alt="image" src="https://github.com/user-attachments/assets/fb6b3f3b-f801-4aca-aec0-d7ae3bb0d57b" />

<img width="546" height="435" alt="image" src="https://github.com/user-attachments/assets/37f71993-9208-44c0-9c4a-c60978b00569" />

<img width="546" height="435" alt="image" src="https://github.com/user-attachments/assets/d5f5e337-c435-4cfa-b2f0-dde86e881e3d" />



