from functools import lru_cache

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from src.config import settings

### hf_local

def _build_hf_local():
    import torch

    from transformers import(
        AutoModelForCausalLM,
        AutoTokenizer,
        pipeline,
    )

    from langchain_huggingface import (
        ChatHuggingFace,
        HuggingFacePipeline,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        settings.hf_model,
        dtype=torch.bfloat16,
    )

    text_gen = pipeline(
        task="text-generation",
        model = model,
        tokenizer=tokenizer,
        device = settings.device,
        return_full_text = False 
    )

    text_get.generation_config.max_new_token=(
        settings.hf_max_new_tokens
    )

    return ChatHuggingFace(
        llm=HuggingFacePipeline(
            pipeline=text_gen
        )
    )

def _build_openai():
    return ChatOpenAI(
        model = settings.openai_model,
        open_ai_key=settings.openai_api_key,
        temperature=settings.llm_temperature
    )

def _build_vllm():
    return ChatOpenAI(
        model=settings.hf_model,
        openai_api_key=settings.vllm_api_key,
        openai_api_base=settings.vllm_api_base,
        temperature=settings.llm_temperature,
    )


@lru_cache(maxsize=4)
def get_llm(provider=None):

    provider = (
        provider
        or settings.llm_provider
    )

    if provider == "hf_local":
        return _build_hf_local()

    if provider == "openai":
        return _build_openai()

    if provider == "vllm":
        return _build_vllm()

    raise ValueError(
        f"Unknown llm_provider '{provider}'"
    )


def invoke_llm(
    prompt,
    provider=None
):

    llm = get_llm(
        provider=provider
    )

    response = llm.invoke(
        [
            HumanMessage(
                content=prompt
            )
        ]
    )

    if isinstance(
        response.content,
        str
    ):
        return response.content

    return str(
        response.content
    )