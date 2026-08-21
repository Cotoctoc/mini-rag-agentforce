"""
Mini RAG API - Retriever externo para Agentforce
--------------------------------------------------
Backend gratuito para probar el patrón:
  Agentforce Topic -> Apex Invocable Action -> este API -> respuesta al agente

Endpoints:
  POST /documents          -> sube un documento (texto plano)
  GET  /documents          -> lista documentos cargados
  DELETE /documents/{id}   -> borra un documento
  POST /query               -> busca los fragmentos mas relevantes para una pregunta
  GET  /health              -> healthcheck

Almacenamiento: en memoria + snapshot a disco (JSON) para persistir entre reinicios
del dyno gratuito de Render (que se duerme tras inactividad).

Busqueda: TF-IDF + similitud coseno (scikit-learn). Liviano, corre bien en el
free tier de Render (512 MB RAM), sin necesidad de descargar modelos pesados
de embeddings ni depender de APIs externas de pago.
"""

import os
import json
import uuid
import re
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------
# Configuracion
# --------------------------------------------------------------------------
DATA_FILE = os.environ.get("DATA_FILE", "data_store.json")
API_KEY = os.environ.get("API_KEY", "")  # si se define, se exige header X-API-Key
CHUNK_SIZE = 800          # caracteres por chunk
CHUNK_OVERLAP = 150       # solapamiento entre chunks

app = FastAPI(
    title="Mini RAG API",
    description="Retriever externo simple para Agentforce (Custom Retriever via Apex)",
    version="1.0.0",
)

# --------------------------------------------------------------------------
# Modelos de datos
# --------------------------------------------------------------------------
class DocumentIn(BaseModel):
    title: str = Field(..., description="Titulo o nombre del documento")
    content: str = Field(..., description="Contenido en texto plano del documento")
    source: Optional[str] = Field(None, description="Origen o referencia opcional")


class DocumentOut(BaseModel):
    id: str
    title: str
    source: Optional[str] = None
    created_at: str
    num_chunks: int


class QueryIn(BaseModel):
    query: str = Field(..., description="Pregunta o termino de busqueda")
    top_k: int = Field(3, ge=1, le=10, description="Cantidad de resultados a devolver")


class QueryResultItem(BaseModel):
    document_id: str
    document_title: str
    chunk_text: str
    score: float


class QueryOut(BaseModel):
    query: str
    results: List[QueryResultItem]


# --------------------------------------------------------------------------
# Almacenamiento simple en memoria con persistencia a disco
# --------------------------------------------------------------------------
# Estructura: { doc_id: {title, source, created_at, chunks: [str, ...]} }
_store = {}


def _load():
    global _store
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                _store = json.load(f)
        except Exception:
            _store = {}
    else:
        _store = {}


def _save():
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(_store, f, ensure_ascii=False, indent=2)


def _chunk_text(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP
        if start < 0:
            start = 0
        if end >= len(text):
            break
    return chunks


def _check_api_key(x_api_key: Optional[str]):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API key invalida o ausente")


_load()

# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "documents": len(_store)}


@app.post("/documents", response_model=DocumentOut)
def create_document(doc: DocumentIn, x_api_key: Optional[str] = Header(None)):
    _check_api_key(x_api_key)
    doc_id = str(uuid.uuid4())
    chunks = _chunk_text(doc.content)
    if not chunks:
        raise HTTPException(status_code=400, detail="El contenido esta vacio")
    _store[doc_id] = {
        "title": doc.title,
        "source": doc.source,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunks": chunks,
    }
    _save()
    return DocumentOut(
        id=doc_id,
        title=doc.title,
        source=doc.source,
        created_at=_store[doc_id]["created_at"],
        num_chunks=len(chunks),
    )


@app.get("/documents", response_model=List[DocumentOut])
def list_documents(x_api_key: Optional[str] = Header(None)):
    _check_api_key(x_api_key)
    return [
        DocumentOut(
            id=doc_id,
            title=d["title"],
            source=d.get("source"),
            created_at=d["created_at"],
            num_chunks=len(d["chunks"]),
        )
        for doc_id, d in _store.items()
    ]


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str, x_api_key: Optional[str] = Header(None)):
    _check_api_key(x_api_key)
    if doc_id not in _store:
        raise HTTPException(status_code=404, detail="Documento no encontrado")
    del _store[doc_id]
    _save()
    return {"deleted": doc_id}


@app.post("/query", response_model=QueryOut)
def query(payload: QueryIn, x_api_key: Optional[str] = Header(None)):
    _check_api_key(x_api_key)

    if not _store:
        return QueryOut(query=payload.query, results=[])

    # Armamos el corpus de chunks con referencia a su documento
    all_chunks = []
    chunk_refs = []  # (doc_id, title)
    for doc_id, d in _store.items():
        for chunk in d["chunks"]:
            all_chunks.append(chunk)
            chunk_refs.append((doc_id, d["title"]))

    if not all_chunks:
        return QueryOut(query=payload.query, results=[])

    # TF-IDF sobre corpus + query
    vectorizer = TfidfVectorizer(stop_words=None)
    try:
        tfidf_matrix = vectorizer.fit_transform(all_chunks + [payload.query])
    except ValueError:
        # corpus vacio tras limpieza
        return QueryOut(query=payload.query, results=[])

    query_vec = tfidf_matrix[-1]
    doc_vecs = tfidf_matrix[:-1]
    sims = cosine_similarity(query_vec, doc_vecs).flatten()

    ranked_idx = sims.argsort()[::-1][: payload.top_k]

    results = [
        QueryResultItem(
            document_id=chunk_refs[i][0],
            document_title=chunk_refs[i][1],
            chunk_text=all_chunks[i],
            score=round(float(sims[i]), 4),
        )
        for i in ranked_idx
        if sims[i] > 0
    ]

    return QueryOut(query=payload.query, results=results)
