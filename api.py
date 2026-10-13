"""
API FastAPI - Assistant Tifinagh
Expose pipeline.py (deja utilise par app.py) sous forme d'endpoints JSON.
A lancer separement de Streamlit : uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline import retrieve_pertinent, generer_reponse

app = FastAPI(
    title="Tifinagh GPT API",
    description="Assistant qui repond en tifinagh a partir d'un corpus (FAISS + Groq).",
)

# CORS : indispensable des que le frontend (index.html) est servi depuis une
# origine differente de l'API (ex: localhost:5500 vs localhost:8000). Sans ca,
# le navigateur bloque silencieusement la requete -> "Failed to fetch" cote JS,
# alors que Swagger UI (/docs) continue de marcher puisqu'il est sur la meme origine.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # en dev : tout autorise. A restreindre en production.
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionIn(BaseModel):
    question: str
    k: int = 3


class ReponseOut(BaseModel):
    reponse: str
    sources: list[str]


@app.get("/")
def racine():
    return {"message": "Tifinagh GPT API. Utilise POST /question avec {'question': '...'}"}


@app.post("/question", response_model=ReponseOut)
def poser_question(payload: QuestionIn):
    resultats = retrieve_pertinent(payload.question, payload.k)
    contexte = "\n\n".join(r["texte"] for r in resultats) if resultats else None
    sources = sorted(set(r["source"] for r in resultats)) if resultats else []
    reponse = generer_reponse(payload.question, contexte)
    return {"reponse": reponse, "sources": sources}