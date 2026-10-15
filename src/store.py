from functools import lru_cache

from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.config import settings
from src.store import (
    get_vector_store,
    ensure_collection
)

@lru_cache(maxsize=1)
def get_embedding():
    return HuggingFaceEmbeddings(
        model_name = settings.embedding_model,
        model_kwargs = {"devices": settings.hf_device},
        encode_kwargs = {"normalize_embeddings": True},
    )

@lru_cache(maxsize=1)
def get_client ():
    settings.storage_dir.mkdir(parents=True, exist_ok = True)
    return QdrantClient(path=str(settings.storage_dir))

def get_vector_store(collection_name=None):
    return QdrantVectorStore(
        client = get_client(),
        collection_name = collection_name or settings.qdrant_collection,
        embedding = get_embedding(),
    )

INDEXED_PAYLOAD_FIELDS = {
    "metadata.document_id":qmodels.PayloadSchemaType.KEYWORD,
    "metadata.filename": qmodels.PayloadSchemaType.KEYWORD,
    "metadata.page": qmodels.PayloadSchemaType.INTEGER,
}

def ensure_collection(recreate= False, collection_name = None):
    client = get_client()
    name = collection_name or settings.qdrant_collection
    exists = client.collection_exists(name)

    if exists and recreate:
        client.delete_collection(name)
        exists = False

    if not exists:
        dim = len(get_embedding().embed_query("dimension probe"))

        client.create_collection(
            collection_name = name,
            vectors_config = qmodels.VectorParams(
                size=dim,
                distance= qmodels.Distance.COSINE
            ),
        )

    payload_schema = client.get_collection(name).payload_schema or {}
    for field, schema in INDEXED_PAYLOAD_FIELDS.items():
        if payload_schema.get(field) is None:
            client.create_payload_index(
                name, 
                field_name = field,
                field_schema = schema
            )

import uuid

def index_chunks(chunks, collection_name = None):
    if not chunks:
        return 0

    ids = [
        str(uuid.uuid5(uuid.NAMESPACE_DNS, c.metadata["chunk_id"]))
        for c in chunks
    ]

    get_vector_store(
        collection_name=collection_name,
    ).add_documents (chunks, ids=ids)

    return len(chunks)

def ingest(
    recreate =False,
    collection_name = None,
    chunker = None,
    chunk_size = None,
    chunk_overlap = None,
):
    pdfs = discover_pdfs()

    ensure_collection(
        recreate=recreate,
        collection_name=collection_name
    )

    chunks = build_chunks(
        pdfs,
        chunker = chunker,
        chunk_size= chunk_size,
        chunk_overlap=chunk_overlap
    )

    return index_chunks(
        chunks,
        collection_name = collection_name
    )

def save_and_ingest_pdf(file_bytes, filename):
    safe_name = Path(filename).name
    dest = settings.data_dir / safe_name

    settings.data_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    dest.write_bytes(file_bytes)

    ensure_collection(recreate=False)

    chunks = build_chunks([dest])

    return {
        "filename": safe_name,
        "chunks_indexed": index_chunks(chunks)
    }
