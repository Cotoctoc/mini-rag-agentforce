"""
Mini RAG API - Retriever externo para Agentforce
--------------------------------------------------
Backend gratuito para probar el patron:
  Agentforce Topic -> Apex Invocable Action -> este API -> respuesta al agente

Endpoints:
  POST /documents          -> sube un documento (texto plano)
  GET  /documents          -> lista documentos cargados
  DELETE /documents/{id}   -> borra un documento
  POST /query               -> busca los fragmentos mas relevantes para una pregunta
  GET  /health              -> healthcheck

Almacenamiento: Postgres (ej. Supabase free tier), persistente entre reinicios
del dyno gratuito de Render (que se duerme tras inactividad y borra el disco
local en cada reinicio).

Busqueda: TF-IDF + similitud coseno (scikit-learn), calculada en memoria en
cada query a partir de lo que hay en la base. Liviano, sin depender de
modelos de embeddings pesados ni APIs externas de pago.
"""

import os
import uuid
import re
from datetime import datetime, timezone
from typing import List, Optional
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.snowball import SnowballStemmer

# --------------------------------------------------------------------------
# Analizador de texto en espanol (stopwords + sinonimos + stemming)
# --------------------------------------------------------------------------
# TF-IDF puro falla cuando la pregunta usa una forma de palabra distinta a
# la del documento (ej. "devolver" en la pregunta vs "devoluciones" en el
# texto). Un stemmer generico tampoco unifica estos casos porque son
# derivaciones de distinta categoria gramatical (verbo vs sustantivo), asi
# que ademas del stemming aplicamos un diccionario de sinonimos acotado a
# los terminos mas comunes del dominio retail.
_stemmer = SnowballStemmer("spanish")

_SPANISH_STOPWORDS = {
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como", "con",
    "contra", "cual", "cuando", "de", "del", "desde", "donde", "durante",
    "e", "el", "ella", "ellas", "ellos", "en", "entre", "era", "erais",
    "eramos", "eran", "eres", "es", "esa", "esas", "ese", "eso", "esos",
    "esta", "estaba", "estado", "estan", "estar", "este", "esto", "estos",
    "fue", "fueron", "ha", "hay", "la", "las", "le", "les", "lo", "los",
    "mas", "me", "mi", "mis", "mucho", "muchos", "muy", "nada", "ni", "no",
    "nos", "nosotros", "o", "os", "otra", "otras", "otro", "otros", "para",
    "pero", "poco", "por", "porque", "que", "quien", "quienes", "se", "sea",
    "sin", "sobre", "su", "sus", "tambien", "tanto", "te", "ti", "tiene",
    "tienen", "todo", "todos", "tu", "tus", "un", "una", "uno", "unos", "y",
    "ya", "yo",
}

_SYNONYM_MAP = {
    "devolver": "devolucion", "devuelvo": "devolucion", "devuelve": "devolucion",
    "devuelto": "devolucion", "devolucion": "devolucion", "devoluciones": "devolucion",
    "reembolso": "devolucion", "reembolsar": "devolucion", "reembolsos": "devolucion",
    "cambiar": "cambio", "cambio": "cambio", "cambios": "cambio", "cambie": "cambio",
    "enviar": "envio", "envio": "envio", "envios": "envio", "entrega": "envio",
    "entregas": "envio", "entregar": "envio", "despacho": "envio", "despachar": "envio",
    "reclamo": "reclamo", "reclamos": "reclamo", "queja": "reclamo", "quejas": "reclamo",
    "reclamar": "reclamo",
    "talla": "talla", "tallas": "talla", "medida": "talla", "medidas": "talla",
    "tamano": "talla", "tamanos": "talla",
    "garantia": "garantia", "garantias": "garantia",
    "retiro": "retiro", "retirar": "retiro", "retiras": "retiro",
    "horario": "horario", "horarios": "horario", "abren": "horario", "cierran": "horario",
    "atienden": "horario",
    "pagar": "pago", "pago": "pago", "pagos": "pago",
}

_TOKEN_RE = re.compile(r"[a-zA-ZñÑáéíóúÁÉÍÓÚ]+")


def spanish_analyzer(text: str) -> List[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    result = []
    for tok in tokens:
        if tok in _SPANISH_STOPWORDS or len(tok) <= 2:
            continue
        result.append(_SYNONYM_MAP.get(tok, _stemmer.stem(tok)))
    return result

# --------------------------------------------------------------------------
# Configuracion
# --------------------------------------------------------------------------
DATABASE_URL = os.environ.get("DATABASE_URL", "")
API_KEY = os.environ.get("API_KEY", "")  # si se define, se exige header X-API-Key
CHUNK_SIZE = 800          # caracteres por chunk
CHUNK_OVERLAP = 150       # solapamiento entre chunks

if not DATABASE_URL:
    raise RuntimeError(
        "Falta la variable de entorno DATABASE_URL. "
        "Configurala con la connection string de tu proyecto Supabase (Postgres)."
    )

app = FastAPI(
    title="Mini RAG API",
    description="Retriever externo simple para Agentforce (Custom Retriever via Apex)",
    version="2.0.0",
)

# --------------------------------------------------------------------------
# Conexion a Postgres
# --------------------------------------------------------------------------
@contextmanager
def get_conn():
    conn = psycopg2.connect(DATABASE_URL, sslmode="require")
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY,
                    title TEXT NOT NULL,
                    source TEXT,
                    created_at TIMESTAMPTZ NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_text TEXT NOT NULL
                );
                """
            )
        conn.commit()


init_db()

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
# Utilidades
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/health")
def health():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents;")
            count = cur.fetchone()[0]
    return {"status": "ok", "documents": count}


@app.post("/documents", response_model=DocumentOut)
def create_document(doc: DocumentIn, x_api_key: Optional[str] = Header(None)):
    _check_api_key(x_api_key)
    chunks = _chunk_text(doc.content)
    if not chunks:
        raise HTTPException(status_code=400, detail="El contenido esta vacio")

    doc_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (id, title, source, created_at) VALUES (%s, %s, %s, %s);",
                (doc_id, doc.title, doc.source, created_at),
            )
            cur.executemany(
                "INSERT INTO chunks (document_id, chunk_text) VALUES (%s, %s);",
                [(doc_id, c) for c in chunks],
            )
        conn.commit()

    return DocumentOut(
        id=doc_id,
        title=doc.title,
        source=doc.source,
        created_at=created_at.isoformat(),
        num_chunks=len(chunks),
    )


@app.get("/documents", response_model=List[DocumentOut])
def list_documents(x_api_key: Optional[str] = Header(None)):
    _check_api_key(x_api_key)
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT d.id, d.title, d.source, d.created_at, COUNT(c.id) AS num_chunks
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                GROUP BY d.id
                ORDER BY d.created_at DESC;
                """
            )
            rows = cur.fetchall()

    return [
        DocumentOut(
            id=str(r["id"]),
            title=r["title"],
            source=r["source"],
            created_at=r["created_at"].isoformat(),
            num_chunks=r["num_chunks"],
        )
        for r in rows
    ]


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str, x_api_key: Optional[str] = Header(None)):
    _check_api_key(x_api_key)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = %s RETURNING id;", (doc_id,))
            deleted = cur.fetchone()
        conn.commit()

    if not deleted:
        raise HTTPException(status_code=404, detail="Documento no encontrado")
    return {"deleted": doc_id}


@app.post("/query", response_model=QueryOut)
def query(payload: QueryIn, x_api_key: Optional[str] = Header(None)):
    _check_api_key(x_api_key)

    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT c.chunk_text, d.id AS document_id, d.title AS document_title
                FROM chunks c
                JOIN documents d ON d.id = c.document_id;
                """
            )
            rows = cur.fetchall()

    if not rows:
        return QueryOut(query=payload.query, results=[])

    # El titulo suele llevar la senal tematica mas fuerte (ej. "Politica de
    # Devoluciones"), asi que lo repetimos para darle mas peso en el vector
    # TF-IDF sin alterar el chunk_text que se devuelve al agente.
    searchable_texts = [
        f"{r['document_title']} {r['document_title']} {r['chunk_text']}" for r in rows
    ]

    vectorizer = TfidfVectorizer(analyzer=spanish_analyzer)
    try:
        tfidf_matrix = vectorizer.fit_transform(searchable_texts + [payload.query])
    except ValueError:
        return QueryOut(query=payload.query, results=[])

    query_vec = tfidf_matrix[-1]
    doc_vecs = tfidf_matrix[:-1]
    sims = cosine_similarity(query_vec, doc_vecs).flatten()

    ranked_idx = sims.argsort()[::-1][: payload.top_k]

    results = [
        QueryResultItem(
            document_id=str(rows[i]["document_id"]),
            document_title=rows[i]["document_title"],
            chunk_text=rows[i]["chunk_text"],
            score=round(float(sims[i]), 4),
        )
        for i in ranked_idx
        if sims[i] > 0
    ]
    return QueryOut(query=payload.query, results=results)
