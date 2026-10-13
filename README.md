# Tifinagh GPT

Assistant RAG en tifinagh : retrieval BM25 + generation via Groq.

## Installation
    pip install -r requirements.txt

## Configuration
1. Cree une cle API gratuite sur console.groq.com
2. Copie .env.example en .env
3. Remplace colle_ta_cle_ici par ta cle

## Lancement
    uvicorn api:app --reload

Puis ouvre index.html dans ton navigateur.
