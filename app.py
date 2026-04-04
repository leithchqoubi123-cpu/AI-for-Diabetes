Voici le code optimisé. J'ai retiré toutes les références à l'IA, supprimé les emojis et les lignes superflues pour obtenir un rendu professionnel. L'affichage des résultats a été transformé en un tableau HTML épuré pour éviter l'aspect "bloc de texte".

```python
import gradio as gr
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

# === 1️⃣ Chargement du modèle ===
pipeline = joblib.load("diabetes_model.pkl")
scaler = pipeline.named_steps["scaler"]
model = pipeline.named_steps["model"]

COLUMNS = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
    'BMI', 'DiabetesPedigreeFunction', 'Age', 'Glucose_Insulin_ratio', 
    'BMI2', 'Glucose_BMI', 'Glucose_Age', 'Pregnancies_Age'
]

# === 2️⃣ Fonction de calcul et analyse ===
def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age):
    
    # Feature Engineering
    df_patient = pd.DataFrame([{
        "Pregnancies": Pregnancies,
        "Glucose": Glucose,
        "BloodPressure": BloodPressure,
        "SkinThickness": SkinThickness,
        "Insulin": Insulin,
        "BMI": BMI,
        "DiabetesPedigreeFunction": DPF,
        "Age": Age,
        "Glucose_Insulin_ratio": Glucose / max(Insulin, 1),
        "BMI2": BMI ** 2,
        "Glucose_BMI": Glucose * BMI,
        "Glucose_Age": Glucose / max(Age, 1),
        "Pregnancies_Age": Pregnancies * Age
    }])[COLUMNS]

    # Normalisation et Calcul
    X_scaled = scaler.transform(df_patient)
    y_prob = float(model.predict_proba(X_scaled)[0, 1])
    y_pred = int(y_prob >= 0.5)

    verdict = "Positif" if y_pred == 1 else "Négatif"
    couleur = "#ff6b6b" if y_pred == 1 else "#00bfa6"
    niveau = "Élevé" if y_prob > 0.7 else ("Modéré" if y_prob > 0.4 else "Faible")

    # Rapport structuré en HTML
    html_report = f"""
    <div style="font-family: sans-serif; border: 1px solid #eee; padding: 15px; border-radius: 10px; background-color: #fafafa;">
        <h2 style="color: {couleur}; margin-top: 0;">Diagnostic : {verdict}</h2>
        <p style="margin-bottom: 5px;">Niveau de risque estimé : <b>{niveau}</b></p>
        <p style="margin-top: 0; color: #666;">Indice de probabilité : {y_prob*100:.1f}%</p>
        <hr style="border: 0; border-top: 1px solid #ddd; margin: 15px 0;">
        <table style="width: 100%; border-collapse: collapse;">
            <tr>
                <td style="padding: 5px 0; color: #555;">Glycémie</td>
                <td style="text-align: right; font-weight: bold;">{Glucose} mg/dL</td>
            </tr>
            <tr>
                <td style="padding: 5px 0; color: #555;">IMC</td>
                <td style="text-align: right; font-weight: bold;">{BMI:.1f} kg/m²</td>
            </tr>
            <tr>
                <td style="padding: 5px 0; color: #555;">Âge</td>
                <td style="text-align: right; font-weight: bold;">{Age} ans</td>
            </tr>
            <tr>
                <td style="padding: 5px 0; color: #555;">Insuline</td>
                <td style="text-align: right; font-weight: bold;">{Insulin} µU/mL</td>
            </tr>
        </table>
    </div>
    """

    # Analyse SHAP
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X_scaled)

        shap_df = pd.DataFrame({
            "Feature": COLUMNS,
            "Impact": shap_values.values[0]
        }).sort_values("Impact", key=abs, ascending=False).head(10)

        plt.figure(figsize=(8, 5))
        colors = ["#00bfa6" if v > 0 else "#ff6b6b" for v in shap_df["Impact"]]
        plt.barh(shap_df["Feature"], shap_df["Impact"], color=colors)
        plt.title("Analyse des variables déterminantes (Leith Chqoubi)", fontsize=12)
        plt.xlabel("Influence sur le résultat")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        shap_path = "impact_analysis.png"
        plt.savefig(shap_path, bbox_inches="tight", dpi=120)
        plt.close()

        return html_report, round(y_prob * 100, 2), shap_path

    except Exception as e:
        return html_report + f"<p style='color:red;'>Erreur d'analyse : {e}</p>", round(y_prob * 100, 2), None

# === 3️⃣ Interface Utilisateur ===
with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal", secondary_hue="gray")) as app:
    gr.HTML("""
    <div style='text-align:center; margin-bottom:20px'>
        <h1>Système Médical — Analyse du Diabète</h1>
        <h3>Développé par <b style='color:#00bfa6'>Leith Chqoubi</b></h3>
        <hr style='margin:10px 0'>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            Pregnancies = gr.Slider(0, 15, 2, step=1, label="Grossesses")
            Glucose = gr.Slider(50, 220, 120, step=1, label="Glycémie (mg/dL)")
            BloodPressure = gr.Slider(40, 120, 70, step=1, label="Pression artérielle (mmHg)")
            SkinThickness = gr.Slider(5, 80, 25, step=1, label="Épaisseur de peau (mm)")
            Insulin = gr.Slider(0, 400, 80, step=1, label="Insuline (µU/mL)")
            BMI = gr.Slider(15.0, 60.0, 28.5, step=0.1, label="IMC (kg/m²)")
            DPF = gr.Slider(0.0, 2.5, 0.5, step=0.01, label="Facteur héréditaire (DPF)")
            Age = gr.Slider(18, 90, 35, step=1, label="Âge")
            btn = gr.Button("Lancer l'analyse", variant="primary")

        with gr.Column(scale=1):
            out_report = gr.HTML(label="Rapport d'analyse")
            out_prob = gr.Number(label="Probabilité calculée (%)", precision=2)
            out_img = gr.Image(label="Impact des variables sur le résultat")

    btn.click(
        predict_diabetes,
        inputs=[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age],
        outputs=[out_report, out_prob, out_img]
    )

    gr.HTML("""
    <hr>
    <div style='text-align:center; font-size:14px; color:gray'>
        Développé par <b>Leith Chqoubi</b><br>
        <i>Prototype académique de diagnostic médical</i>
    </div>
    """)

if __name__ == "__main__":
    app.launch()
```
