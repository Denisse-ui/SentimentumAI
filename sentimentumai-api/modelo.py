"""
modelo.py — Carga del clasificador de hate speech
Intenta HF Hub primero, fallback a joblib (LR local)
"""
import os, joblib
from pathlib import Path

# ── Intento 1: HF Fine-tuned desde Hub ───────────────────────────────────
def cargar_hf(model_id: str):
    from transformers import pipeline
    print(f"Cargando modelo HF: {model_id}")
    clf = pipeline(
        "text-classification",
        model=model_id,
        token=os.getenv("HF_TOKEN") or None,
        truncation=True,
        max_length=128
    )
    print("✅ Modelo HF cargado")
    return clf, "hf"

# ── Intento 2: LR desde joblib ────────────────────────────────────────────
def cargar_local():
    base = Path(__file__).parent.parent / "outputs" / "models"
    lr_path  = base / "lr_model.pkl"
    vec_path = base / "vectorizer.pkl"

    if not lr_path.exists() or not vec_path.exists():
        raise FileNotFoundError(f"No se encontraron modelos en {base}")

    lr_model   = joblib.load(lr_path)
    vectorizer = joblib.load(vec_path)
    print("✅ Modelo LR local cargado")
    return (lr_model, vectorizer), "local"


def clasificar(modelo, tipo: str, textos: list[str]) -> list[dict]:
    """Clasifica una lista de textos. Devuelve lista de dicts."""
    if tipo == "hf":
        resultados = modelo(textos)
        label_map = {
            "HATE": 1, "NOT-HATE": 0,
            "hate": 1, "not-hate": 0,
            "LABEL_1": 1, "LABEL_0": 0,
            "1": 1, "0": 0
            }
        #label_map = {
        #    "HATE": 1, "NOT-HATE": 0,
        #    "hate": 1, "not-hate": 0,
        #    "LABEL_1": 1, "LABEL_0": 0,
        #}

        print("RAW:", resultados)
        return [
            {
                "texto": t,
                "prediccion": label_map.get(r["label"], 0),
                "etiqueta": "🔴 ODIO" if label_map.get(r["label"], 0) == 1 else "🟢 No odio",
                "confianza": round(r["score"], 4),
                "modelo": "HF Fine-tuned"
            }
            for t, r in zip(textos, resultados)
        ]
    else:
        import re
        from nltk.stem import WordNetLemmatizer
        from nltk.corpus import stopwords
        import nltk
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)

        lr_model, vectorizer = modelo
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()

        contractions = {
            "don't": "do not", "can't": "cannot", "isn't": "is not",
            "won't": "will not", "didn't": "did not", "doesn't": "does not"
        }

        def clean(text):
            text = text.lower()
            text = re.sub(r"http\S+|www\S+", "", text)
            text = re.sub(r"@\w+|#\w+", "", text)
            for c, e in contractions.items():
                text = text.replace(c, e)
            text = re.sub(r"[^a-z ]", "", text)
            tokens = [lemmatizer.lemmatize(w) for w in text.split()
                      if w not in stop_words and len(w) > 2]
            return " ".join(tokens)

        limpios = [clean(t) for t in textos]
        X = vectorizer.transform(limpios)
        preds = lr_model.predict(X)
        probs = lr_model.predict_proba(X)[:, 1]
        print("RAW:", resultados)
        return [
            {
                "texto": t,
                "prediccion": int(p),
                "etiqueta": "🔴 ODIO" if p == 1 else "🟢 No odio",
                "confianza": round(float(pr), 4),
                "modelo": "Logistic Regression"
            }
            for t, p, pr in zip(textos, preds, probs)
        ]


def inicializar():
    """Carga el modelo disponible en orden de preferencia."""
    model_id = os.getenv("HF_MODEL_ID", "Denisse-MR98/sentimentumai-hate-speech")
    modo     = os.getenv("MODELO", "hf")

    if modo == "hf":
        try:
            return cargar_hf(model_id)
        except Exception as e:
            print(f"⚠️  HF falló ({e}), usando modelo local...")
            return cargar_local()
    else:
        return cargar_local()
