import csv
import hashlib
import os
import subprocess
from datetime import datetime
from typing import List

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchAny,
    MatchValue,
)

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL_EMBED = os.getenv("OLLAMA_MODEL_EMBED", "nomic-embed-text")
OLLAMA_MODEL_CHAT = os.getenv("OLLAMA_MODEL_CHAT", "qwen2.5:14b")

COLLECTION = os.getenv("QDRANT_COLLECTION", "irongate_chunks")

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

app = FastAPI(title="IronGate Prototype API")
qdrant = QdrantClient(url=QDRANT_URL)


@app.get("/health")
def health():
    return {"ok": True, "time": datetime.utcnow().isoformat()}


class IngestFromMetadataRequest(BaseModel):
    base_dir: str = "/opt/irongate"
    metadata_csv: str = "demo_metadata.csv"
class ChatRequest(BaseModel):
    user_id: str
    role: str
    matter_id: str
    message: str
    top_k: int = 5
def ensure_collection(dim: int = 768) -> None:
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION in existing:
        return
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
def pdf_to_text(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)
    result = subprocess.run(
        ["pdftotext", "-layout", pdf_path, "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pdftotext failed: {result.stderr[:300]}")
    return result.stdout.strip()
def chunk_text(text: str) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks: List[str] = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + CHUNK_SIZE])
        i += (CHUNK_SIZE - CHUNK_OVERLAP)
    return chunks


def stable_point_id(doc_id: str, chunk_index: int) -> int:
    h = hashlib.sha256(f"{doc_id}:{chunk_index}".encode()).hexdigest()
    return int(h[:15], 16)
def ollama_embed(texts: List[str]) -> List[List[float]]:
    vectors: List[List[float]] = []
    for t in texts:
        r = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": OLLAMA_MODEL_EMBED, "prompt": t},
            timeout=180,
        )
        if r.status_code != 200:
            raise RuntimeError(f"Ollama embeddings failed: {r.status_code} {r.text[:200]}")
        vectors.append(r.json()["embedding"])
    return vectors
def ollama_chat(system: str, user: str, context: str) -> str:
    prompt = (
        f"{system}\n\n"
        f"Context (use only if relevant; cite by [Title | chunk N]):\n{context}\n\n"
        f"User:\n{user}\n\nAssistant:"
    )
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL_CHAT, "prompt": prompt, "stream": False, "temperature": 0.2},
        timeout=300,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Ollama generate failed: {r.status_code} {r.text[:200]}")
    return r.json().get("response", "").strip()
def matter_and_role_filter(matter_id: str, role: str) -> Filter:
    return Filter(
        must=[
            FieldCondition(key="matter_id", match=MatchValue(value=matter_id)),
            FieldCondition(key="allowed_roles", match=MatchAny(any=[role])),
        ]
    )


@app.post("/ingest/from_metadata")
def ingest_from_metadata(req: IngestFromMetadataRequest):
    base = req.base_dir.rstrip("/")
    meta_path = os.path.join(base, req.metadata_csv)
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=400, detail=f"metadata_csv not found: {meta_path}")
    ingested = 0
    with open(meta_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = row["doc_id"]
            matter_id = row["matter_id"]
            rel_path = row["path"]
            title = row["title"]
            sensitivity = row["sensitivity"]
            allowed_roles_raw = row["allowed_roles"].strip()
            if allowed_roles_raw.startswith("["):
                import json
                allowed_roles = json.loads(allowed_roles_raw)
            else:
                allowed_roles = [r.strip() for r in allowed_roles_raw.split(",") if r.strip()]

            pdf_path = os.path.join(base, rel_path)
            text = pdf_to_text(pdf_path)
            chunks = chunk_text(text)
            if not chunks:
                continue

            vectors = ollama_embed(chunks)
            dim = len(vectors[0])

            existing = [c.name for c in qdrant.get_collections().collections]
            if COLLECTION not in existing:
                ensure_collection(dim)
            else:
                current_dim = qdrant.get_collection(COLLECTION).config.params.vectors.size
                if current_dim != dim:
                    qdrant.delete_collection(COLLECTION)
                    ensure_collection(dim)
            points = []
            for idx, (ch, vec) in enumerate(zip(chunks, vectors)):
                pid = stable_point_id(doc_id, idx)
                payload = {
                    "doc_id": doc_id,
                    "matter_id": matter_id,
                    "path": rel_path,
                    "title": title,
                    "sensitivity": sensitivity,
                    "allowed_roles": allowed_roles,
                    "chunk_index": idx,
                    "text": ch,
                }
                points.append(PointStruct(id=pid, vector=vec, payload=payload))

            qdrant.upsert(collection_name=COLLECTION, points=points)
            ingested += 1

    return {"ingested_docs": ingested, "collection": COLLECTION}
@app.post("/chat")
def chat(req: ChatRequest):
    query_vec = ollama_embed([req.message])[0]
    flt = matter_and_role_filter(req.matter_id, req.role)

    hits = qdrant.search(
        collection_name=COLLECTION,
        query_vector=query_vec,
        limit=req.top_k,
        query_filter=flt,
        with_payload=True,
    )

    if not hits:
        system = (
            "You are IronGate, a private law-firm assistant. "
            "If you do not have sufficient permitted sources, say so. Do not fabricate."
        )
        answer = ollama_chat(system, req.message, context="(no permitted sources found)")
        return {"answer": answer, "citations": [], "sources_accessed": []}
    citations = []
    context_lines = []
    sources_accessed = []
    for h in hits:
        p = h.payload
        citations.append(
            {
                "title": p["title"],
                "path": p["path"],
                "chunk_index": p["chunk_index"],
                "snippet": p["text"][:350],
            }
        )
        context_lines.append(f"[{p['title']} | chunk {p['chunk_index']}]\n{p['text']}\n")
        sources_accessed.append({"doc_id": p["doc_id"], "chunk_index": p["chunk_index"]})

    system = (
        "You are IronGate, a private law-firm assistant.\n"
        "Rules:\n"
        "1) Use ONLY the provided context for case-specific facts.\n"
        "2) If the context does not contain the answer, say what is missing.\n"
        "3) Cite as [Title | chunk N].\n"
        "4) Keep answers concise and professional."
    )
    answer = ollama_chat(system, req.message, context="\n\n".join(context_lines))
    return {
        "answer": answer,
        "citations": citations,
        "sources_accessed": sources_accessed,
    }

