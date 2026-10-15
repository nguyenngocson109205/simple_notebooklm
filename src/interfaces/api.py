from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Field

from src.filters import MetadataFilter, filters_to_dict
from src.indexing import list_documents, save_and_ingest_pdf
from src.learning import (
    generate_flashcards,
    generate_quiz,
    summarize as summarize_learning,
)
from src.rag import answer
from src.schemas import (
    FlashcardSet,
    QuizSet,
    RagAnswer,
    Summary,
)


# =====================================================
# RESPONSE SCHEMAS
# =====================================================

class DocumentInfo(BaseModel):
    filename: str
    document_id: str
    pages: int
    chunks: int


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int


# =====================================================
# REQUEST SCHEMAS
# =====================================================

class AskRequest(BaseModel):
    question: str = Field(min_length=1)

    k: int | None = Field(
        default=None,
        ge=1,
        le=64,
    )

    filters: MetadataFilter | None = None


class SummarizeRequest(BaseModel):
    document: str | None = None
    query: str | None = None
    filters: MetadataFilter | None = None

    k: int | None = Field(
        default=None,
        ge=1,
        le=64,
    )


class QuizRequest(BaseModel):
    document: str | None = None
    query: str | None = None
    filters: MetadataFilter | None = None

    count: int | None = Field(
        default=None,
        ge=1,
        le=50,
    )

    k: int | None = Field(
        default=None,
        ge=1,
        le=64,
    )


class FlashcardsRequest(QuizRequest):
    pass


app = FastAPI(
    title= "RAG Learning API",
    description=...,
    version="0.1.0",
)

@app.get("/health")
def health():
    return{
        "status": "ok"
    }

@app.get(
    "/documents",
    response_model=list[DocumentInfo],
)
def documents():
    return list_documents()

@app.post(
    "/upload",
    response_model=UploadResponse
)
async def upload(
    file: UploadFile = File(...)
):
    content = await file.read()

    return save_and_ingest_pdf(
        content,
        file.filename or "",
    )

@app.post(
    "/ask",
    response_model=RagAnswer,
)
def ask(
    req: AskRequest
):
    return answer(
        req.question,
        k=req.k,
        filters=filters_to_dict(
            req.filters
        ),
    )

@app.post(
    "/summarize",
    response_model=Summary,
)

