"""
FastAPI application exposing semantic search over Qdrant using CLIP multilingual model.
"""
import os
from functools import lru_cache
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import SearchParams
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "images")
API_KEY = os.getenv("API_KEY")

app = FastAPI(title="Multilingual Semantic Search")


class SearchResponse(BaseModel):
    id: str
    score: float
    payload: Optional[dict]


@lru_cache(maxsize=1)
def load_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def qdrant_client() -> AsyncQdrantClient:
    return AsyncQdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )


async def verify_api_key(x_api_key: str = Header(...)):
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.post("/search", dependencies=[Depends(verify_api_key)])
async def search(query: str, threshold: float = 0.6, limit: int = 10):
    model = load_model()
    client = qdrant_client()
    vector = model.encode(query, convert_to_numpy=True).tolist()

    hits = await client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=limit,
        search_params=SearchParams(hnsw_ef=128),
    )
    await client.close()

    results: List[SearchResponse] = []
    for hit in hits:
        if hit.score < threshold:
            continue
        results.append(
            SearchResponse(id=str(hit.id), score=hit.score, payload=hit.payload)
        )
    return JSONResponse(content=[r.model_dump() for r in results])


@app.post("/search-by-image", dependencies=[Depends(verify_api_key)])
async def search_by_image(file: UploadFile = File(...), threshold: float = 0.6, limit: int = 10):
    model = load_model()
    client = qdrant_client()
    file.file.seek(0)
    image = Image.open(file.file)
    vector = model.encode(images=[image], convert_to_numpy=True)[0].tolist()

    hits = await client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=limit,
        search_params=SearchParams(hnsw_ef=128),
    )
    await client.close()

    results: List[SearchResponse] = []
    for hit in hits:
        if hit.score < threshold:
            continue
        results.append(
            SearchResponse(id=str(hit.id), score=hit.score, payload=hit.payload)
        )
    return JSONResponse(content=[r.model_dump() for r in results])
