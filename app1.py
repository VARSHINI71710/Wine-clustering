import gradio as gr
import pandas as pd
import joblib


scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
kmeans = joblib.load("kmeans_model.pkl")


USER_CREDENTIALS = {
    "admin": "password123",
    "user": "user123"
}


def login(username, password):
    if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
        return True, f"Welcome, {username}! Go to Cluster Prediction."
    else:
        return False, "Incorrect username or password."


def predict_cluster(
    Alcohol, Malic_Acid, Ash, Ash_Alcanity, Magnesium,
    Total_Phenols, Flavanoids, Nonflavanoid_Phenols,
    Proanthocyanins, Color_Intensity, Hue, OD280, Proline
):
    data = pd.DataFrame([{
        'Alcohol': Alcohol,
        'Malic_Acid': Malic_Acid,
        'Ash': Ash,
        'Ash_Alcanity': Ash_Alcanity,
        'Magnesium': Magnesium,
        'Total_Phenols': Total_Phenols,
        'Flavanoids': Flavanoids,
        'Nonflavanoid_Phenols': Nonflavanoid_Phenols,
        'Proanthocyanins': Proanthocyanins,
        'Color_Intensity': Color_Intensity,
        'Hue': Hue,
        'OD280': OD280,
        'Proline': Proline
    }])

    X_scaled = scaler.transform(data)
    X_pca = pca.transform(X_scaled)
    cluster = kmeans.predict(X_pca)[0]
    
    return f"Predicted Cluster: {cluster}"


with gr.Blocks() as app:
    with gr.Tab("Login"):
        gr.Markdown("## Wine Cluster Prediction - Login")
        username = gr.Textbox(label="Username")
        password = gr.Textbox(label="Password", type="password")
        login_btn = gr.Button("Login")
        login_output = gr.Textbox(label="Login Status", interactive=False)

    with gr.Tab("Cluster Prediction") as cluster_tab:
        gr.Markdown("## Enter Wine Features")
        with gr.Row():
            Alcohol_in = gr.Number(label="Alcohol")
            Malic_Acid_in = gr.Number(label="Malic Acid")
            Ash_in = gr.Number(label="Ash")
            Ash_Alcanity_in = gr.Number(label="Ash Alcanity")
            Magnesium_in = gr.Number(label="Magnesium")
        with gr.Row():
            Total_Phenols_in = gr.Number(label="Total Phenols")
            Flavanoids_in = gr.Number(label="Flavanoids")
            Nonflavanoid_Phenols_in = gr.Number(label="Nonflavanoid Phenols")
            Proanthocyanins_in = gr.Number(label="Proanthocyanins")
        with gr.Row():
            Color_Intensity_in = gr.Number(label="Color Intensity")
            Hue_in = gr.Number(label="Hue")
            OD280_in = gr.Number(label="OD280")
            Proline_in = gr.Number(label="Proline")
        
        predict_btn = gr.Button("Predict Cluster")
        cluster_output = gr.Textbox(label="Cluster Result", interactive=False)

   
    login_btn.click(
        fn=login,
        inputs=[username, password],
        outputs=[gr.State(), login_output]
    )

    predict_btn.click(
        fn=predict_cluster,
        inputs=[
            Alcohol_in, Malic_Acid_in, Ash_in, Ash_Alcanity_in, Magnesium_in,
            Total_Phenols_in, Flavanoids_in, Nonflavanoid_Phenols_in,
            Proanthocyanins_in, Color_Intensity_in, Hue_in, OD280_in, Proline_in
        ],
        outputs=[cluster_output]
    )


app.launch()
