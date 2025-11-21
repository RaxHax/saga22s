"""
Async batch loader for uploading images and metadata to S3 and Qdrant.
"""
import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import aiobotocore.session
from PIL import Image
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer
from tqdm.asyncio import tqdm

MODEL_NAME = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
EMBED_DIM = 512


@dataclass
class IngestConfig:
    data_dir: Path
    bucket: str
    collection: str
    s3_region: Optional[str]
    s3_endpoint: Optional[str]
    concurrency: int
    batch_size: int
    qdrant_url: str
    qdrant_api_key: Optional[str]
    public_base_url: Optional[str]


async def ensure_collection(client: AsyncQdrantClient, collection: str) -> None:
    exists = await client.collection_exists(collection_name=collection)
    if not exists:
        await client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=0),
        )


async def toggle_indexing(client: AsyncQdrantClient, collection: str, m_value: int) -> None:
    await client.update_collection(collection_name=collection, hnsw_config=HnswConfigDiff(m=m_value))


async def list_unprocessed(
    client: AsyncQdrantClient, collection: str, candidate_ids: List[str]
) -> List[str]:
    existing_ids = set()
    chunk_size = 512
    for i in range(0, len(candidate_ids), chunk_size):
        chunk = candidate_ids[i : i + chunk_size]
        existing = await client.retrieve(collection_name=collection, ids=chunk)
        existing_ids.update(point.id for point in existing)
    return [pid for pid in candidate_ids if pid not in existing_ids]


def default_public_url(bucket: str, key: str, region: Optional[str], base: Optional[str]) -> str:
    if base:
        return f"{base.rstrip('/')}/{key}"
    region_segment = f".{region}" if region else ""
    return f"https://{bucket}.s3{region_segment}.amazonaws.com/{key}"


async def upload_image(
    s3_client,
    bucket: str,
    key: str,
    image_path: Path,
) -> None:
    data = await asyncio.get_event_loop().run_in_executor(None, image_path.read_bytes)
    await s3_client.put_object(Bucket=bucket, Key=key, Body=data, ACL="public-read")


async def read_text(text_path: Path) -> Optional[str]:
    if not text_path.exists():
        return None
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, text_path.read_text)


async def gather_batch_payloads(
    s3_client,
    config: IngestConfig,
    files: Iterable[Path],
    model: SentenceTransformer,
) -> List[PointStruct]:
    semaphore = asyncio.Semaphore(config.concurrency)
    tasks = [
        asyncio.create_task(
            load_item_data(semaphore, s3_client, config, image_path, image_path.name)
        )
        for image_path in files
    ]

    items = []
    for coro in asyncio.as_completed(tasks):
        item = await coro
        if item:
            items.append(item)

    if not items:
        return []

    image_embeddings = model.encode(
        images=[item["image"] for item in items],
        convert_to_numpy=True,
        batch_size=len(items),
    )

    text_inputs = [item["text"] for item in items if item.get("text")]
    text_embeddings = None
    if text_inputs:
        text_embeddings = iter(model.encode(text_inputs, convert_to_numpy=True))

    results: List[PointStruct] = []
    for image_embedding, item in zip(image_embeddings, items):
        payload = item["payload"]
        vector = image_embedding

        if item.get("text") and text_embeddings is not None:
            text_embedding = next(text_embeddings)
            vector = (image_embedding + text_embedding) / 2
        results.append(PointStruct(id=item["id"], vector=vector.tolist(), payload=payload))
        if hasattr(item["image"], "close"):
            item["image"].close()

    return results


async def load_item_data(
    semaphore: asyncio.Semaphore,
    s3_client,
    config: IngestConfig,
    image_path: Path,
    key: str,
) -> Optional[dict]:
    async with semaphore:
        await upload_image(s3_client, config.bucket, key, image_path)
        text = await read_text(image_path.with_suffix(".txt"))

        loop = asyncio.get_event_loop()
        image = await loop.run_in_executor(None, Image.open, image_path)
        payload = {
            "image_url": default_public_url(
                config.bucket, key, config.s3_region, config.public_base_url
            ),
            "file_name": image_path.name,
        }
        if text:
            payload["text"] = text

        return {"id": key, "image": image, "text": text, "payload": payload}


async def process(config: IngestConfig) -> None:
    session = aiobotocore.session.get_session()
    model = SentenceTransformer(MODEL_NAME)
    client = AsyncQdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)
    await ensure_collection(client, config.collection)

    files = sorted(config.data_dir.glob("*.png"))
    ids = [p.name for p in files]
    unprocessed_ids = await list_unprocessed(client, config.collection, ids)
    remaining = [p for p in files if p.name in unprocessed_ids]

    await toggle_indexing(client, config.collection, 0)

    connector = dict(region_name=config.s3_region)
    if config.s3_endpoint:
        connector["endpoint_url"] = config.s3_endpoint

    async with session.create_client("s3", **{k: v for k, v in connector.items() if v}) as s3_client:
        for i in tqdm(range(0, len(remaining), config.batch_size)):
            batch_files = remaining[i : i + config.batch_size]
            payloads = await gather_batch_payloads(s3_client, config, batch_files, model)
            if payloads:
                await client.upsert(collection_name=config.collection, points=payloads)

    await toggle_indexing(client, config.collection, 16)
    await client.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Async ingest pipeline for Qdrant + S3")
    parser.add_argument("data_dir", type=Path, help="Directory containing .png images and .txt metadata")
    parser.add_argument("--bucket", required=True, help="Target S3 bucket")
    parser.add_argument("--collection", default="images", help="Qdrant collection name")
    parser.add_argument("--s3-region", dest="s3_region", default=None)
    parser.add_argument("--s3-endpoint", dest="s3_endpoint", default=None)
    parser.add_argument("--public-base-url", dest="public_base_url", default=None)
    parser.add_argument("--concurrency", type=int, default=16, help="Concurrent upload workers")
    parser.add_argument("--batch-size", type=int, default=128, help="Number of items per Qdrant upsert")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--qdrant-api-key", dest="qdrant_api_key", default=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = IngestConfig(
        data_dir=args.data_dir,
        bucket=args.bucket,
        collection=args.collection,
        s3_region=args.s3_region,
        s3_endpoint=args.s3_endpoint,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        public_base_url=args.public_base_url,
    )
    asyncio.run(process(config))


if __name__ == "__main__":
    main()
