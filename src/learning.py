import json

from pydantic import ValidationError

from src.config import settings
from src.llm import invoke_llm
from src.rag import (
    fetch_all_chunks,
    format_citations,
    render_prompt,
    retrieve,
)
from src.schemas import (
    Flashcard,
    FlashcardSet,
    QuizItem,
    QuizSet,
    Summary,
)


# =====================================================
# PROMPT TEMPLATE NAMES
# =====================================================

SUMMARY_SINGLE_TEMPLATE = "summary_single.jinja2"
SUMMARY_MAP_TEMPLATE = "summary_map.jinja2"
SUMMARY_REDUCE_TEMPLATE = "summary_reduce.jinja2"

QUIZ_TEMPLATE = "quiz.jinja2"
FLASHCARDS_TEMPLATE = "flashcards.jinja2"


# =====================================================
# 1. XÁC ĐỊNH PHẠM VI NỘI DUNG
# =====================================================

def _resolve_target(
    document=None,
    query=None,
    filters=None,
    k=None,
    retrieval_k=None,
):
    effective_filters = dict(
        filters or {}
    )

    # Nếu user chọn một document cụ thể
    # thì chuyển document thành filter filename.
    if document:
        effective_filters[
            "filename"
        ] = document

    # Nếu có query:
    # dùng semantic search để lấy các chunk liên quan nhất.
    if query:
        chunks = retrieve(
            query,
            k=k or retrieval_k,
            filters=effective_filters,
        )

        return (
            chunks,
            "query",
            query,
        )

    # Nếu không có query nhưng có document/filter:
    # lấy toàn bộ chunk trong phạm vi đó.
    if effective_filters:
        chunks = fetch_all_chunks(
            filters=effective_filters
        )

        scope = (
            "document"
            if document
            else "filter"
        )

        target = ", ".join(
            f"{key}={value}"
            for key, value
            in effective_filters.items()
        )

        return (
            chunks,
            scope,
            target,
        )

    # Không document, không query,
    # không filter -> toàn bộ corpus.
    return (
        fetch_all_chunks(
            filters=None
        ),
        "corpus",
        None,
    )


# =====================================================
# 2. PARSE JSON TỪ OUTPUT CỦA LLM
# =====================================================

def _parse_json(text):
    cleaned = text.strip()

    # LLM đôi khi trả:
    #
    # ```json
    # {...}
    # ```
    #
    # nên cần bỏ code fence.
    if cleaned.startswith("```"):
        cleaned = cleaned.split(
            "\n",
            1
        )[-1]

        cleaned = cleaned.removesuffix(
            "```"
        ).strip()

    obj = json.loads(
        cleaned
    )

    if not isinstance(
        obj,
        (dict, list)
    ):
        raise RuntimeError(
            "Expected JSON object or array."
        )

    return obj


# =====================================================
# 3. VALIDATE SUMMARY JSON
# =====================================================

def _validate_summary_payload(
    payload
):
    if not isinstance(
        payload,
        dict
    ):
        raise RuntimeError(
            "Summary output must be a JSON object."
        )

    summary_text = payload.get(
        "summary"
    )

    key_points = payload.get(
        "key_points",
        []
    )

    if not isinstance(
        summary_text,
        str
    ):
        raise RuntimeError(
            "Summary output is missing "
            "a valid 'summary' field."
        )

    summary_text = (
        summary_text.strip()
    )

    if not summary_text:
        raise RuntimeError(
            "Summary cannot be empty."
        )

    if not isinstance(
        key_points,
        list
    ):
        raise RuntimeError(
            "'key_points' must be a list."
        )

    cleaned_points = []

    for point in key_points:
        if not isinstance(
            point,
            str
        ):
            continue

        point = point.strip()

        if point:
            cleaned_points.append(
                point
            )

    return (
        summary_text,
        cleaned_points,
    )


# =====================================================
# 4. VALIDATE QUIZ / FLASHCARD ITEMS
# =====================================================

def _validate_items(
    payload,
    key,
    model_class,
    dedup_field,
    label,
    valid_markers,
):
    if not isinstance(
        payload,
        dict
    ):
        raise RuntimeError(
            "Expected JSON object."
        )

    raw_items = payload.get(
        key
    )

    if not isinstance(
        raw_items,
        list
    ):
        raise RuntimeError(
            f"Expected '{key}' to be a list."
        )

    items = []
    seen = set()

    for raw in raw_items:

        try:
            item = (
                model_class.model_validate(
                    raw
                )
            )

        except ValidationError:
            continue

        # Field dùng để phát hiện item trùng.
        #
        # Quiz:
        # question
        #
        # Flashcard:
        # front
        norm = str(
            getattr(
                item,
                dedup_field,
                ""
            )
        ).strip().lower()

        # Bỏ item rỗng hoặc duplicate.
        if (
            not norm
            or norm in seen
        ):
            continue

        seen.add(
            norm
        )

        # Chỉ giữ citation marker thực sự tồn tại
        # trong danh sách chunk hiện tại.
        markers = [
            marker
            for marker
            in item.source_markers
            if marker
            in valid_markers
        ]

        item = item.model_copy(
            update={
                "source_markers":
                    markers
            }
        )

        items.append(
            item
        )

    if not items:
        raise RuntimeError(
            f"No valid {label} produced."
        )

    return items


# =====================================================
# 5. SUMMARY
# =====================================================

def summarize(
    document=None,
    query=None,
    filters=None,
    k=None,
):
    chunks, scope, target = (
        _resolve_target(
            document,
            query,
            filters,
            k,
            settings.summarize_retrieval_k,
        )
    )

    if not chunks:
        raise RuntimeError(
            "No chunks found for summarization."
        )

    # =================================================
    # CASE 1:
    # Ít chunk -> summarize một lần
    # =================================================

    if (
        len(chunks)
        <= settings.summarize_batch_size
    ):
        prompt = render_prompt(
            SUMMARY_SINGLE_TEMPLATE,
            chunks=chunks,
        )

        text = invoke_llm(
            prompt
        )

        payload = _parse_json(
            text
        )

        (
            summary_text,
            key_points,
        ) = _validate_summary_payload(
            payload
        )

    # =================================================
    # CASE 2:
    # Nhiều chunk -> Map Reduce
    # =================================================

    else:
        partials = []

        # -------------------------
        # MAP
        # -------------------------

        for start in range(
            0,
            len(chunks),
            settings.summarize_batch_size,
        ):
            batch = chunks[
                start:
                start
                + settings.summarize_batch_size
            ]

            prompt = render_prompt(
                SUMMARY_MAP_TEMPLATE,
                chunks=batch,
            )

            text = invoke_llm(
                prompt
            )

            payload = _parse_json(
                text
            )

            (
                partial_summary,
                partial_key_points,
            ) = _validate_summary_payload(
                payload
            )

            partials.append(
                {
                    "summary":
                        partial_summary,
                    "key_points":
                        partial_key_points,
                }
            )

        # -------------------------
        # REDUCE
        # -------------------------

        reduce_prompt = render_prompt(
            SUMMARY_REDUCE_TEMPLATE,
            partials=partials,
        )

        reduce_text = invoke_llm(
            reduce_prompt
        )

        payload = _parse_json(
            reduce_text
        )

        (
            summary_text,
            key_points,
        ) = _validate_summary_payload(
            payload
        )

    return Summary(
        scope=scope,
        target=target,
        summary=summary_text,
        key_points=key_points,
        citations=format_citations(
            chunks
        ),
        chunks=chunks,
    )


# =====================================================
# 6. GENERATE QUIZ
# =====================================================

def generate_quiz(
    document=None,
    query=None,
    filters=None,
    count=None,
    k=None,
):
    chunks, scope, target = (
        _resolve_target(
            document,
            query,
            filters,
            k,
            settings.generation_retrieval_k,
        )
    )

    if not chunks:
        raise RuntimeError(
            "No chunks found for quiz generation."
        )

    # Nếu user không truyền count
    # thì lấy default trong config.
    n = (
        count
        or settings.quiz_default_count
    )

    # Ví dụ có 3 chunks:
    #
    # S1
    # S2
    # S3
    valid_markers = {
        f"S{i}"
        for i in range(
            1,
            len(chunks) + 1
        )
    }

    prompt = render_prompt(
        QUIZ_TEMPLATE,
        chunks=chunks,
        count=n,
    )

    text = invoke_llm(
        prompt
    )

    payload = _parse_json(
        text
    )

    items = _validate_items(
        payload=payload,
        key="items",
        model_class=QuizItem,
        dedup_field="question",
        label="quiz items",
        valid_markers=valid_markers,
    )

    return QuizSet(
        scope=scope,
        target=target,
        items=items,
        citations=format_citations(
            chunks
        ),
        chunks=chunks,
    )


# =====================================================
# 7. GENERATE FLASHCARDS
# =====================================================

def generate_flashcards(
    document=None,
    query=None,
    filters=None,
    count=None,
    k=None,
):
    chunks, scope, target = (
        _resolve_target(
            document,
            query,
            filters,
            k,
            settings.generation_retrieval_k,
        )
    )

    if not chunks:
        raise RuntimeError(
            "No chunks found for flashcard generation."
        )

    n = (
        count
        or settings.flashcards_default_count
    )

    valid_markers = {
        f"S{i}"
        for i in range(
            1,
            len(chunks) + 1
        )
    }

    prompt = render_prompt(
        FLASHCARDS_TEMPLATE,
        chunks=chunks,
        count=n,
    )

    text = invoke_llm(
        prompt
    )

    payload = _parse_json(
        text
    )

    cards = _validate_items(
        payload=payload,
        key="cards",
        model_class=Flashcard,
        dedup_field="front",
        label="flashcards",
        valid_markers=valid_markers,
    )

    return FlashcardSet(
        scope=scope,
        target=target,
        cards=cards,
        citations=format_citations(
            chunks
        ),
        chunks=chunks,
    )