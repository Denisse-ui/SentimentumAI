"""
SentimentumAI — API REST
Uso: uvicorn api:app --reload
Docs: http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

from modelo import inicializar, clasificar
from twitter_client import buscar_tweets

# ── Inicializar app y modelo ──────────────────────────────────────────────
app   = FastAPI(
    title="SentimentumAI",
    description="API de detección de hate speech en tweets usando ML y HuggingFace",
    version="1.0.0"
)

print("Cargando modelo...")
modelo, tipo_modelo = inicializar()
print(f"✅ API lista — modelo activo: {tipo_modelo}")


# ── Schemas ───────────────────────────────────────────────────────────────
class AnalizarRequest(BaseModel):
    termino: str = "#racism"
    max_tweets: int = 10

class AnalizarTextoRequest(BaseModel):
    textos: list[str]


# ── Endpoints ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "proyecto": "SentimentumAI",
        "version": "1.0.0",
        "modelo_activo": tipo_modelo,
        "docs": "/docs"
    }


@app.post("/analizar")
def analizar(req: AnalizarRequest):
    """
    Busca tweets por término (Twitter real o demo) y los clasifica.
    """
    if req.max_tweets < 1 or req.max_tweets > 100:
        raise HTTPException(400, "max_tweets debe estar entre 1 y 100")

    tweets, fuente = buscar_tweets(req.termino, req.max_tweets)
    resultados     = clasificar(modelo, tipo_modelo, tweets)

    total  = len(resultados)
    odio   = sum(1 for r in resultados if r["prediccion"] == 1)

    return {
        "termino":      req.termino,
        "fuente":       fuente,
        "total_tweets": total,
        "odio_detectado": odio,
        "porcentaje_odio": round(odio / total * 100, 1) if total else 0,
        "modelo":       tipo_modelo,
        "resultados":   resultados
    }


@app.post("/analizar/texto")
def analizar_texto(req: AnalizarTextoRequest):
    """
    Clasifica textos enviados directamente (sin Twitter).
    """
    if not req.textos:
        raise HTTPException(400, "Envía al menos un texto")

    resultados = clasificar(modelo, tipo_modelo, req.textos)
    odio = sum(1 for r in resultados if r["prediccion"] == 1)

    return {
        "total":          len(resultados),
        "odio_detectado": odio,
        "modelo":         tipo_modelo,
        "resultados":     resultados
    }


@app.get("/estadisticas")
def estadisticas():
    """
    Métricas del modelo en producción (HF Fine-tuned sobre test set de 6,393 tweets).
    """
    return {
        "modelo_recomendado": "HF Fine-tuned (cardiffnlp/twitter-roberta-base-hate-latest)",
        "modelo_activo":      tipo_modelo,
        "dataset": {
            "total_tweets": 31353,
            "ratio_desbalance": "13.1:1",
            "tweets_odio": 2226,
            "tweets_no_odio": 29127
        },
        "metricas_test": {
            "HF Fine-tuned":         {"F1": 0.865, "Recall": 0.893, "Precision": 0.843, "AUC": 0.994},
            "Logistic Regression":   {"F1": 0.586, "Recall": 0.891, "Precision": 0.440, "AUC": 0.957},
            "Linear SVM":            {"F1": 0.583, "Recall": 0.433, "Precision": 0.894, "AUC": 0.964},
            "Complement Naive Bayes":{"F1": 0.516, "Recall": 0.882, "Precision": 0.362, "AUC": 0.950},
            "HF Pre-entrenado":      {"F1": 0.207, "Recall": 0.172, "Precision": 0.261, "AUC": 0.814},
        },
        "umbral_optimo_alertas": {
            "precision_minima": 0.60,
            "HF Fine-tuned":          {"umbral": 0.011, "recall": 0.962},
            "Linear SVM":             {"umbral": 0.160, "recall": 0.886},
            "Logistic Regression":    {"umbral": 0.653, "recall": 0.763},
            "Complement Naive Bayes": {"umbral": 0.697, "recall": 0.717},
        }
    }
