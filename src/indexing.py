import hashlib
import uuid

from collections import defaultdict
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.schemas import ChunkMetadata
from src.store import get_vector_store, ensure_collection

def _document_id(path):
    raw = f"{path.name}:{path.start().st_size}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

def _chunk_id(doc_id, page, index):
    return f"{doc_id}:{page}:{index}"

def _load_pdf(path):
    pages = PyPDFLoader(str(path).load())
    doc_id =  _document_id(path)

    for doc in pages:
        page_number = int(doc.metadat.get("page", 0)) + 1

        doc.metadata = {
            "document_id": doc_id,
            "filename": path.name,
            "source": str(path.resolve()),
            "page": page_number,
            "section": doc.metadata.get("section"),
        }

    return pages

def _splitter(chunk_size= None, chunk_overlap = None):
    return RecursiveCharacterTextSplitter(
        chunk_size = chunk_size or settings.chunk_size,
        chunk_overlap = chunk_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=False,
    )

def build_chunk(pdf_paths, chunk_size = None, chunk_overlap = None, chunker = None):
    page_docs = []
    for path in pdf_paths:
        page_docs.extend(_load_pdf(path))

    splitter = chunker or _splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(page_docs)
    per_doc_counter = defaultdict(int)

    for chunk in chunks:
        doc_id = chunk.metadata["document_id"]
        idx = per_doc_counter[doc_id]
        per_doc_counter[doc_id] += 1

        meta = ChunkMetadata(
            document_id=doc_id,
            filename=chunk.metadata["filename"],
            source=chunk.metadata["source"],
            page=chunk.metadata["page"],
            chunk_id = _chunk_id(doc_id, chunk.metadata["page"], idx),
            section=chunk.metadata.get("section"),
        )

        chunk.metadata = meta.model_dump()

    return chunks 


