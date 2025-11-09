# AI-for-Diabet
# ============================================================
# === 🧠 IA Médicale — Détection du Diabète (Version Présentation Prof) ===
# ============================================================

!pip -q install imbalanced-learn xgboost shap gradio

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
import gradio as gr
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from google.colab import drive

RANDOM_STATE = 42
plt.style.use("seaborn-v0_8-whitegrid")

# === 1️⃣ Charger les données ===
def load_data():
    drive.mount("/content/drive", force_remount=False)
    path = "/content/drive/MyDrive/IA_Diabete/diabetes.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"⚠️ Fichier introuvable : {path}")
    df = pd.read_csv(path)
    print("✅ Fichier chargé :", path)
    print("Shape :", df.shape)
    return df

# === 2️⃣ Nettoyage et features ===
def clean_and_engineer(df):
    df = df.copy()
    cols_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for c in cols_zero:
        df[c] = df[c].replace(0, np.nan).fillna(df[c].median())
    df["Glucose_Insulin_ratio"] = df["Glucose"] / np.clip(df["Insulin"], 1, None)
    df["BMI2"] = df["BMI"] ** 2
    df["Glucose_BMI"] = df["Glucose"] * df["BMI"]
    df["Glucose_Age"] = df["Glucose"] / np.clip(df["Age"], 1, None)
    df["Pregnancies_Age"] = df["Pregnancies"] * df["Age"]
    return df

# === 3️⃣ Préparer les données ===
def prepare_xy(df):
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_b, y_train_b = sm.fit_resample(X_train, y_train)
    return X, y, X_train_b, X_test, y_train_b, y_test

# === 4️⃣ Entraînement optimisé ===
def tune_xgb(X_train, y_train):
    xgb = XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_estimators=400)
    params = {
        "max_depth": [3, 4, 5],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.5, 1.0],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    pipe = Pipeline([("scaler", StandardScaler()), ("model", xgb)])
    rs = RandomizedSearchCV(pipe, param_distributions={
        "model__" + k: v for k, v in params.items()
    }, n_iter=15, cv=cv, scoring="roc_auc", n_jobs=-1, random_state=RANDOM_STATE)
    rs.fit(X_train, y_train)
    print(f"✅ Meilleur AUC (CV) : {rs.best_score_:.3f}")
    return rs.best_estimator_

# === 5️⃣ Évaluation du modèle ===
def evaluate(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    print("\n🎯 Performance du modèle :")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("ROC AUC :", roc_auc_score(y_test, y_proba))
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print("Matrice de confusion :\n", cm)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "--")
    plt.title("Courbe ROC")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()

# === 6️⃣ Sauvegarde du pipeline ===
def save_pipeline(pipe):
    path = "/content/drive/MyDrive/IA_Diabete/diabetes_pipeline_prof.pkl"
    joblib.dump(pipe, path)
    print("✅ Modèle sauvegardé :", path)

# === 7️⃣ Pipeline complet ===
def main():
    df = load_data()
    df = clean_and_engineer(df)
    X, y, X_train_b, X_test, y_train_b, y_test = prepare_xy(df)
    best_pipe = tune_xgb(X_train_b, y_train_b)
    evaluate(best_pipe, X_test, y_test)
    save_pipeline(best_pipe)
    return best_pipe, X_test

best_pipe, X_test = main()

# === 8️⃣ Mini-App Gradio (Interface de présentation) ===
def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    df_patient = pd.DataFrame([{
        "Pregnancies": Pregnancies,
        "Glucose": Glucose,
        "BloodPressure": BloodPressure,
        "SkinThickness": SkinThickness,
        "Insulin": Insulin,
        "BMI": BMI,
        "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
        "Age": Age,
        "Glucose_Insulin_ratio": Glucose / (Insulin if Insulin != 0 else 1),
        "BMI2": BMI ** 2,
        "Glucose_BMI": Glucose * BMI,
        "Glucose_Age": Glucose / max(Age, 1),
        "Pregnancies_Age": Pregnancies * Age
    }])[X_test.columns]

    scaler = best_pipe.named_steps["scaler"]
    model = best_pipe.named_steps["model"]
    X_scaled = scaler.transform(df_patient)
    y_prob = model.predict_proba(X_scaled)[0, 1]
    y_pred = int(y_prob >= 0.5)

    # Texte d’interprétation
    result = "🩺 Diabétique" if y_pred == 1 else "✅ Non diabétique"
    risk_color = "🔴" if y_prob > 0.7 else ("🟠" if y_prob > 0.4 else "🟢")

    # SHAP rapide
    explainer = shap.Explainer(model)
    shap_values = explainer(X_scaled)
    shap_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Impact": shap_values.values[0]
    }).sort_values("Impact", key=abs, ascending=False)

    plt.figure(figsize=(7, 4))
    plt.barh(shap_df["Feature"], shap_df["Impact"], color="coral")
    plt.title("Importance des variables (Patient)")
    plt.xlabel("Valeur SHAP")
    plt.gca().invert_yaxis()
    shap_path = "/tmp/shap_prof.png"
    plt.savefig(shap_path, bbox_inches="tight")
    plt.close()

    # Résumé médical
    description = (
        f"{risk_color} **Résultat IA : {result}**\n\n"
        f"Probabilité de diabète : **{y_prob*100:.2f}%**\n\n"
        "🧩 Analyse :\n"
        f"- Le modèle estime un risque {'élevé' if y_prob>0.7 else 'modéré' if y_prob>0.4 else 'faible'}.\n"
        "- Les variables ayant le plus influencé la prédiction sont listées ci-dessous."
    )

    # Jauge de probabilité (Gradio)
    return description, y_prob, shap_path

inputs = [
    gr.Slider(0, 15, 2, label="Grossesses (Pregnancies)"),
    gr.Slider(50, 200, 120, label="Glycémie (Glucose)"),
    gr.Slider(40, 120, 70, label="Pression artérielle (Blood Pressure)"),
    gr.Slider(10, 50, 25, label="Épaisseur peau (Skin Thickness)"),
    gr.Slider(0, 400, 80, label="Insuline"),
    gr.Slider(15, 50, 28.5, label="IMC (BMI)"),
    gr.Slider(0.0, 2.5, 0.5, step=0.01, label="Facteur héréditaire"),
    gr.Slider(18, 80, 35, label="Âge")
]
outputs = [
    gr.Markdown(label="Rapport médical IA"),
    gr.Number(label="Probabilité de diabète", precision=2),
    gr.Image(label="Explication SHAP (influence des paramètres)")
]

app = gr.Interface(
    fn=predict_diabetes,
    inputs=inputs,
    outputs=outputs,
    title="🧠 IA Médicale — Détection du Diabète",
    description="Prototype développé par **Leith**. Ce modèle IA basé sur XGBoost évalue le risque de diabète à partir de paramètres biologiques et cliniques.",
    theme="gradio/soft",
)
app.launch(debug=True)