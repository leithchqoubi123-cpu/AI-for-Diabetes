import gradio as gr
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

# === 1️⃣ Chargement du "Cerveau" (Le modèle pré-entraîné) ===
# Le script charge ton fichier .pkl une seule fois au démarrage du site
pipeline = joblib.load("diabetes_model.pkl")
scaler = pipeline.named_steps["scaler"]
model = pipeline.named_steps["model"]

# Ordre strict des colonnes attendu par ton modèle XGBoost
COLUMNS = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
    'BMI', 'DiabetesPedigreeFunction', 'Age', 'Glucose_Insulin_ratio', 
    'BMI2', 'Glucose_BMI', 'Glucose_Age', 'Pregnancies_Age'
]

# === 2️⃣ Fonction de prédiction pour un patient ===
def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age):
    
    # Recréation des variables dérivées (Feature Engineering) exactement comme à l'entraînement
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

    # Mise à l'échelle et Prédiction
    X_scaled = scaler.transform(df_patient)
    y_prob = float(model.predict_proba(X_scaled)[0, 1])
    y_pred = int(y_prob >= 0.5)

    # Rédaction du rapport médical
    verdict = "🩺 **Diabétique**" if y_pred == 1 else "✅ **Non diabétique**"
    niveau = "élevé" if y_prob > 0.7 else ("modéré" if y_prob > 0.4 else "faible")
    couleur = "🔴" if y_prob > 0.7 else ("🟠" if y_prob > 0.4 else "🟢")

    markdown = f"""
## {couleur} Diagnostic IA : {verdict}

**Probabilité de diabète :** {y_prob*100:.1f}%
**Niveau de risque :** {niveau}

### 📋 Résumé des paramètres :
- **Glycémie** : {Glucose} mg/dL
- **IMC** : {BMI:.1f} kg/m²
- **Âge** : {Age} ans
- **Insuline** : {Insulin} µU/mL
"""

    # Génération du graphique SHAP
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
        plt.title("🧠 Analyse SHAP — Impact des variables (Leith Chqoubi)", fontsize=13, pad=10)
        plt.xlabel("Influence sur le risque de diabète", fontsize=11)
        plt.gca().invert_yaxis()
        plt.tight_layout()

        shap_path = "shap_patient.png"
        plt.savefig(shap_path, bbox_inches="tight", dpi=120)
        plt.close()

        return markdown, round(y_prob * 100, 2), shap_path

    except Exception as e:
        return markdown + f"\n\n⚠️ Erreur lors de la génération SHAP : {e}", round(y_prob * 100, 2), None

# === 3️⃣ Interface Graphique Gradio ===
with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal", secondary_hue="gray")) as app:
    gr.HTML("""
    <div style='text-align:center; margin-bottom:20px'>
        <h1>🧠 IA Médicale — Détection du Diabète</h1>
        <h3>Analyse intelligente développée par <b style='color:#00bfa6'>Leith Chqoubi</b></h3>
        <p>Modèle XGBoost optimisé (SMOTE + standardisation + SHAP explainer)</p>
        <hr style='margin:10px 0'>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            Pregnancies = gr.Slider(0, 15, 2, step=1, label="Grossesses (Pregnancies)")
            Glucose = gr.Slider(50, 220, 120, step=1, label="Glycémie (mg/dL)")
            BloodPressure = gr.Slider(40, 120, 70, step=1, label="Pression artérielle (mmHg)")
            SkinThickness = gr.Slider(5, 80, 25, step=1, label="Épaisseur de peau (mm)")
            Insulin = gr.Slider(0, 400, 80, step=1, label="Insuline (µU/mL)")
            BMI = gr.Slider(15.0, 60.0, 28.5, step=0.1, label="IMC (kg/m²)")
            DPF = gr.Slider(0.0, 2.5, 0.5, step=0.01, label="Facteur héréditaire (DPF)")
            Age = gr.Slider(18, 90, 35, step=1, label="Âge")
            btn = gr.Button("⚡ Lancer l'analyse IA", variant="primary")

        with gr.Column(scale=1):
            out_text = gr.Markdown(label="🧩 Rapport médical IA")
            out_prob = gr.Number(label="Probabilité (%)", precision=2)
            out_img = gr.Image(label="Explication SHAP (impact des variables)")

    btn.click(
        predict_diabetes,
        inputs=[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age],
        outputs=[out_text, out_prob, out_img]
    )

    gr.HTML("""
    <hr>
    <div style='text-align:center; font-size:14px; color:gray'>
        Développé avec 💡 Intelligence Artificielle — <b>Leith Chqoubi</b><br>
        <i>Prototype académique de diagnostic IA médicale (XGBoost + SHAP)</i>
    </div>
    """)

# Lancement de l'application
if __name__ == "__main__":
    app.launch()
