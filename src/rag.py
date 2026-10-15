from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from src.config import settings
from src.filters import filters_to_qdrant
from src.schemas import ChunkMetadata, RetrievedChunk, Citation, RagAnswer
from src.store import get_vector_store

PROMPTS_DIR = Path(__file__).parent / "prompts"

def retrieve(query, k=None, filter=None, collection_name=None):
    hits = get_vector_store(collection_name).similarity_search_with_score(
        query=query,
        k=k or settings.top_k,
        filter = filters_to_qdrant(filter),
    )

    return [
        RetrievedChunk(
            text=doc.page_content,
            score=float(score),
            metadata=ChunkMetadata(**doc.metadata)
        )
        for doc, score in hits
    ]

def fetch_all_chunks(filter=None, collection_name = None):
    name = collection_name or settings.qdrant_collection
    result = []

    for page in scroll_all(
        name, 
        scroll_filter=filters_to_qdrant(filters)
    ):
        for point in page:
            payload = point.payload or {}

            meta = payload.get("metadata") or {}
            text = payload.get("page_content") or ""

            if meta and text:
                result.append(
                    RetrievedChunk(
                        text=text,
                        score=0.0,
                        metadata=ChunkMetadata(**meta)
                    )
                )
    return sorted(
        result,
        key=lambda r:(
            r.metadata.filename,
            r.metadata.page,
            int(
                r.metadata.chunk_id.rsplit(":", 1)[-1]
            ),
        )
    )

@lru_cache(maxsize=1)
def _jinja_env():
    return Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)),
        autoescape=False,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )

def render_prompt(template_name, **context):
    return _jinja_env().get_template(template_name).render(**context)

def format_citations(chunks):
    return [
        Citation(
            source_index=i,
            source_marker=f"S{i}",
            filename=c.metadata.filename,
            page=c.metadata.page,
            section=c.metadata.section,
            chunk_id=c.metadata.chunk_id,
        )
        for i, c in enumerate(chunks, start=1)
    ]

def answer(question, k=None, filters=None, collection_name = None):
    chunks= retrieve(
        question,
        k=k,
        filters=filters,
        collection_name=collection_name
    )

    if not chunks:
        return RagAnswer(
            question=question,
            answer="Tôi không có đủ thông tin trong ngữ cảnh được cung cấp để trả lời."
        )

    promt = render_prompt(
        ANSWER_TEMPLATE,
        question=question,
        chunks=chunks
    )

    text = invoke_llm(prompt)

    return RagAnswer(
        question=question,
        answer=text.strip(),
        citations=format_citations(chunks),
        chunks=chunks
    )