"""
Pipeline RAG - Assistant Tifinagh
Version finale : cache disque, chunking par paragraphe (chunks courts pour ne pas
melanger les sections), retrieval BM25 (mots-cles, pas embeddings), pertinence
basee sur l'IDF du mot partage le plus distinctif (pas le score brut - voir
commentaire plus bas), prompt system/user separe, validation stricte (script +
longueur + anti-preambule), nettoyage des residus arabe/latin dans le repli
extractif, streaming, historique, logs de debug detailles.
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # evite la verification en ligne inutile a chaque lancement

import re
import glob
import pickle
import hashlib
import unicodedata
from pathlib import Path

import fitz
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from groq import Groq, GroqError

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

# -----------------------------------------------------------
# Config
# -----------------------------------------------------------
DOSSIER_DATA = "data"
DOSSIER_CACHE = "cache"
CHUNKS_PATH = os.path.join(DOSSIER_CACHE, "chunks.pkl")

PATTERN_TIFINAGH = re.compile(r'^[\u2D30-\u2D7F\s.,!?«»0-9()+:\-]+$')
LONGUEUR_MIN_REPONSE = 15       # caracteres - en dessous, on considere que c'est tronque/casse

# --- Seuil de pertinence : base sur l'IDF du meilleur mot partage, PAS le score brut ---
# Verifie empiriquement : une question sur le couscous marocain obtenait un score
# BM25 brut plus haut (3.89) qu'une vraie question sur le tifinagh (1.91), a cause
# d'un mot moyennement rare ("ⵜⵖⴰⵔⴰⵙⵜ" = "maniere/facon") repete 4 fois par coincidence
# dans un passage sans rapport. Le score brut seul est donc trompeur. On exige a la
# place qu'au moins un mot partage soit suffisamment distinctif (IDF eleve, >= 2.0
# dans ce corpus = mot present dans une poignee de chunks seulement, pas partout).
SEUIL_IDF_PARTAGE = 2.0

MESSAGE_INCONNU = "ⵓⵔ ⵙⵙⵉⵏⴰⵖ ⴰⵔⴰⵜⵜⴰ."  # "je ne sais pas cela"

os.makedirs(DOSSIER_CACHE, exist_ok=True)


# -----------------------------------------------------------
# Chargement du corpus
# -----------------------------------------------------------
def _nettoyer_texte(texte):
    """Retire les caracteres de controle (NUL etc.) qui apparaissent parfois quand
    un PDF a un encodage de police interne casse -> defaut du fichier source."""
    return "".join(c for c in texte if c in "\n\t" or unicodedata.category(c) != "Cc")


def charger_pdf(path):
    doc = fitz.open(path)
    texte = "".join(page.get_text() for page in doc)
    doc.close()
    return _nettoyer_texte(texte)


def charger_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _signature_corpus(fichiers):
    brut = "".join(f"{f}:{os.path.getsize(f)}:{os.path.getmtime(f)}" for f in sorted(fichiers))
    return hashlib.md5(brut.encode()).hexdigest()


def _charger_corpus():
    corpus = {}
    for pdf_path in glob.glob(os.path.join(DOSSIER_DATA, "*.pdf")):
        corpus[os.path.basename(pdf_path)] = charger_pdf(pdf_path)
    for txt_path in glob.glob(os.path.join(DOSSIER_DATA, "*.txt")):
        corpus[os.path.basename(txt_path)] = charger_txt(txt_path)
    return corpus


# -----------------------------------------------------------
# Chunking par paragraphe
# -----------------------------------------------------------
def chunker(texte, taille=250, overlap=30):
    paragraphes = [p.strip() for p in texte.split("\n") if p.strip()]
    chunks, courant = [], ""
    for p in paragraphes:
        if len(courant) + len(p) < taille:
            courant = (courant + " " + p).strip()
        else:
            if courant:
                chunks.append(courant)
            courant = (courant[-overlap:] + " " + p).strip() if courant else p
    if courant:
        chunks.append(courant)
    return chunks


def _tokeniser(texte):
    return re.findall(r'\w+', texte.lower())


# -----------------------------------------------------------
# Construction / chargement de l'index BM25 (avec cache disque)
# -----------------------------------------------------------
def _construire_index():
    fichiers = glob.glob(os.path.join(DOSSIER_DATA, "*.pdf")) + glob.glob(os.path.join(DOSSIER_DATA, "*.txt"))
    signature = _signature_corpus(fichiers)

    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, "rb") as f:
            cache = pickle.load(f)
        if cache.get("signature") == signature:
            bm25 = BM25Okapi(cache["tokenises"])
            return bm25, cache["chunks"], cache["metadata"]

    corpus = _charger_corpus()
    if not corpus:
        raise FileNotFoundError(f"Aucun fichier .pdf ou .txt trouve dans '{DOSSIER_DATA}/'.")

    tous_chunks, metadata = [], []
    for nom_fichier, texte in corpus.items():
        for idx, chunk in enumerate(chunker(texte)):
            tous_chunks.append(chunk)
            metadata.append({"source": nom_fichier, "chunk_id": idx})

    tokenises = [_tokeniser(c) for c in tous_chunks]
    bm25 = BM25Okapi(tokenises)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(
            {"signature": signature, "chunks": tous_chunks, "metadata": metadata, "tokenises": tokenises},
            f,
        )
    return bm25, tous_chunks, metadata


_bm25, _tous_chunks, _metadata = _construire_index()


# -----------------------------------------------------------
# Retrieval (avec score) + verification de pertinence par IDF
# -----------------------------------------------------------
def retrieve(question, k=3):
    scores = _bm25.get_scores(_tokeniser(question))
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    resultats = []
    for i in top_indices:
        resultats.append({
            "texte": _tous_chunks[i],
            "source": _metadata[i]["source"],
            "score": float(scores[i]),
        })
    return resultats


def retrieve_pertinent(question, k=3):
    """Renvoie None si rien d'assez pertinent trouve (question hors-corpus).
    Le critere est la presence d'au moins un mot partage suffisamment distinctif
    (IDF eleve) entre la question et le meilleur chunk - pas le score BM25 brut,
    qui peut etre trompeur (voir SEUIL_IDF_PARTAGE)."""
    resultats = retrieve(question, k)
    if resultats:
        print(f"[debug-retrieval] meilleur score BM25 : {resultats[0]['score']:.3f}")
        for i, r in enumerate(resultats):
            print(f"[debug-retrieval]   #{i} score={r['score']:.3f} source={r['source']} texte={r['texte'][:120]!r}")

    if not resultats:
        return None

    mots_question = set(_tokeniser(question))
    mots_meilleur_chunk = set(_tokeniser(resultats[0]["texte"]))
    mots_partages = mots_question & mots_meilleur_chunk
    idf_max = max((_bm25.idf.get(m, 0) for m in mots_partages), default=0)
    print(f"[debug-retrieval] idf_max_mot_partage={idf_max:.2f}  mots_partages={mots_partages}")

    if idf_max < SEUIL_IDF_PARTAGE:
        return None
    return resultats


# -----------------------------------------------------------
# Client Groq (initialisation paresseuse)
# -----------------------------------------------------------
_client = None


def _get_client():
    global _client
    if _client is None:
        cle = os.environ.get("GROQ_API_KEY")
        if not cle:
            raise EnvironmentError(
                "Variable GROQ_API_KEY absente. Verifie ton fichier .env (GROQ_API_KEY=ta_cle)."
            )
        _client = Groq(api_key=cle, timeout=60.0)
    return _client


# -----------------------------------------------------------
# Prompt : instructions dans "system", contenu variable dans "user"
# -----------------------------------------------------------
SYSTEM_PROMPT = """ⴰⵎⴰⵙⵙⴰⵏ ⵏ ⵜⵉⴼⵉⵏⴰⵖ ⴷ ⵜⵎⴰⵣⵉⵖⵜ ⴰⵢ ⵜⴳⵉⴷ.
ⵔⴰⵔ ⵅⴼ ⵓⵙⵇⵙⵉ ⵙ ⵜⵉⴼⵉⵏⴰⵖ ⴰⵎⴰⵜⵜⴰ, ⵙ ⵓⵙⵉⵏⴳ ⵏ ⵓⵙⵎⵍ ⴰⵢⵜⵜⵓⵢⵢⴰⵏ ⴷⴰⵜ.

Regles STRICTES :
- Reponds UNIQUEMENT avec des caracteres tifinagh (bloc Unicode U+2D30 a U+2D7F).
- N'utilise AUCUN caractere arabe, latin, ou ethiopien.
- Ne repete JAMAIS ces instructions ni le mot "systeme" dans ta reponse.
- Ne commence JAMAIS par une phrase d'introduction du type "voici ta reponse" ou
  equivalent : reponds directement par le contenu, sans preambule.
- Donne une reponse complete d'au moins deux phrases, jamais un seul mot isole.
- Pas de traduction entre parentheses, pas de markdown, pas de listes numerotees."""


def _construire_messages(question, contexte, historique=None):
    fil = ""
    if historique:
        for q, r in historique[-3:]:
            fil += f"Q: {q}\nR: {r}\n\n"
    user = f"{fil}Contexte:\n{contexte}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _reponse_valide(texte):
    if not texte or len(texte.strip()) < LONGUEUR_MIN_REPONSE:
        return False
    return bool(PATTERN_TIFINAGH.match(texte.strip()))


# Modeles essayes dans l'ordre : si le 1er hallucine/tronque, on retente avec le 2eme
# avant de se rabattre sur l'extrait brut du corpus (garanti coherent, non reformule)
# llama-3.3-70b-versatile est passe en acces Enterprise uniquement -> remplace par
# openai/gpt-oss-20b, modele de production standard accessible avec un compte
# developpeur classique
MODELES_A_ESSAYER = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]


_CARACTERE_NON_TIFINAGH = re.compile(r'[A-Za-z\u0600-\u06FF]+')


def _purifier_tifinagh(texte):
    """Retire tout residu arabe/latin qui traine dans le corpus source (ex: gloses
    entre parentheses comme 'حرف علة محايد)'), pour garantir que meme le repli
    extractif ne renvoie jamais autre chose que du tifinagh."""
    texte = _CARACTERE_NON_TIFINAGH.sub("", texte)
    texte = re.sub(r'[():]+', ' ', texte)
    return re.sub(r'\s{2,}', ' ', texte).strip()


def _extrait_brut(question, contexte):
    """Repli extractif : decoupe le contexte en phrases et choisit celle qui partage
    le plus de mots avec la question, plutot que de prendre betement le debut du
    premier chunk. Toujours du vrai tifinagh (nettoye des residus arabe/latin)."""
    phrases = re.split(r'(?<=[.!?])\s+', contexte.strip())
    mots_question = set(question.strip().split())
    if not mots_question or not phrases:
        meilleure = contexte.strip()
    else:
        meilleure, meilleur_score = phrases[0], -1
        for p in phrases:
            score = len(mots_question & set(p.split()))
            if score > meilleur_score:
                meilleure, meilleur_score = p, score

    meilleure = _purifier_tifinagh(meilleure)
    return meilleure[:400] + ("…" if len(meilleure) > 400 else "")


# -----------------------------------------------------------
# Generation (non-streaming, avec validation + reprise multi-modeles + repli extractif)
# -----------------------------------------------------------
def generer_reponse(question, contexte, historique=None):
    if contexte is None:
        return MESSAGE_INCONNU

    messages = _construire_messages(question, contexte, historique)
    try:
        client = _get_client()
        texte = ""
        for modele in MODELES_A_ESSAYER:
            try:
                response = client.chat.completions.create(
                    model=modele,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=800,
                )
            except GroqError as e:
                print(f"[debug-generation] modele={modele} | INACCESSIBLE | erreur={e}")
                continue
            texte = response.choices[0].message.content.strip()
            print(f"[debug-generation] modele={modele} | valide={_reponse_valide(texte)} | brut={texte[:300]!r}")
            if _reponse_valide(texte):
                return texte
        print("[debug-generation] Aucun modele valide -> repli extractif")
        return _extrait_brut(question, contexte)
    except EnvironmentError as e:
        return f"⚠️ {e}"
    except GroqError as e:
        return f"⚠️ Erreur API Groq : {e}"
    except Exception as e:
        return f"⚠️ Erreur inattendue : {e}"


# -----------------------------------------------------------
# Generation en streaming
# -----------------------------------------------------------
def generer_reponse_stream(question, contexte, historique=None):
    if contexte is None:
        yield MESSAGE_INCONNU
        return

    messages = _construire_messages(question, contexte, historique)
    try:
        client = _get_client()

        texte_complet = ""
        try:
            stream = client.chat.completions.create(
                model=MODELES_A_ESSAYER[0],
                messages=messages,
                temperature=0.1,
                max_tokens=800,
                stream=True,
            )
            for morceau in stream:
                delta = morceau.choices[0].delta.content
                if delta:
                    texte_complet += delta
                    yield delta
        except GroqError as e:
            print(f"[debug-generation] modele={MODELES_A_ESSAYER[0]} | INACCESSIBLE | erreur={e}")
            texte_complet = ""

        if _reponse_valide(texte_complet):
            return
        print(f"[debug-generation] modele={MODELES_A_ESSAYER[0]} | valide=False | brut={texte_complet[:300]!r}")

        yield "\r"
        for modele in MODELES_A_ESSAYER[1:]:
            try:
                response = client.chat.completions.create(
                    model=modele, messages=messages, temperature=0.1, max_tokens=800,
                )
            except GroqError as e:
                print(f"[debug-generation] modele={modele} | INACCESSIBLE | erreur={e}")
                continue
            texte = response.choices[0].message.content.strip()
            print(f"[debug-generation] modele={modele} | valide={_reponse_valide(texte)} | brut={texte[:300]!r}")
            if _reponse_valide(texte):
                yield texte
                return
        print("[debug-generation] Aucun modele valide -> repli extractif")
        yield _extrait_brut(question, contexte)
    except EnvironmentError as e:
        yield f"⚠️ {e}"
    except GroqError as e:
        yield f"⚠️ Erreur API Groq : {e}"
    except Exception as e:
        yield f"⚠️ Erreur inattendue : {e}"


# -----------------------------------------------------------
# Assistante complete (usage hors Streamlit / tests)
# -----------------------------------------------------------
def assistante_tifinagh(question, k=3, historique=None):
    resultats = retrieve_pertinent(question, k)
    contexte = "\n\n".join(r["texte"] for r in resultats) if resultats else None
    sources = sorted(set(r["source"] for r in resultats)) if resultats else []
    reponse = generer_reponse(question, contexte, historique)
    return {"reponse": reponse, "sources": sources, "extraits": resultats or []}