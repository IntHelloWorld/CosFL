import json
import sys
from pathlib import Path

import httpx
import tiktoken
from cohere import Client
from llama_index.core import Settings
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.postprocessor.jinaai_rerank import JinaRerank

sys.path.append(Path(__file__).resolve().parents[1].as_posix())
from Utils.path_manager import PathManager

DEFAULT_TIMEOUT = 120


def set_models(path_manager: PathManager):
    # set embedding model
    if path_manager.config.models.embed.series == "openai":
        Settings.embed_model = OpenAIEmbedding(
            model=path_manager.config.models.embed.model,
            api_key=path_manager.config.models.embed.api_key,
            api_base=path_manager.config.models.embed.base_url,
            embed_batch_size=path_manager.config.models.embed.batch_size,
        )
    elif path_manager.config.models.embed.series == "jina":
        Settings.embed_model = JinaEmbedding(
            api_key=path_manager.config.models.embed.api_key,
            model=path_manager.config.models.embed.model,
            embed_batch_size=path_manager.config.models.embed.batch_size,
        )
    elif path_manager.config.models.embed.series == "voyage":
        Settings.embed_model = VoyageEmbedding(
            model_name=path_manager.config.models.embed.model,
            voyage_api_key=path_manager.config.models.embed.api_key,
            embed_batch_size=path_manager.config.models.embed.batch_size,
        )
    else:
        raise ValueError(
            f"Unknown embedding model series: {path_manager.config.models.embed.series}"
        )

    # set summary model
    if path_manager.config.models.summary.series == "openai":
        Settings.llm = OpenAI(
            model=path_manager.config.models.summary.model,
            api_key=path_manager.config.models.summary.api_key,
            api_base=path_manager.config.models.summary.base_url,
        )
    else:
        raise ValueError(
            f"Unknown summary model series: {path_manager.config.models.summary.series}"
        )

    # set reasoning model
    if path_manager.config.models.reason.series == "openai":
        path_manager.reasoning_llm = OpenAI(
            model=path_manager.config.models.reason.model,
            api_key=path_manager.config.models.reason.api_key,
            api_base=path_manager.config.models.reason.base_url,
            timeout=DEFAULT_TIMEOUT,
        )
    else:
        raise ValueError(
            f"Unknown reasoning model series: {path_manager.config.models.reason.series}"
        )


def get_embedding_reranker(path_manager: PathManager):
    embedding_reranker = None
    if path_manager.config.models.rerank.series == "jina":
        embedding_reranker = JinaRerank(
            api_key=path_manager.config.models.rerank.api_key,
            model=path_manager.config.models.rerank.model,
            top_n=path_manager.config.hyper.rerank_top_n,
        )
    elif path_manager.config.models.rerank.series == "cohere":
        embedding_reranker = CohereRerank(
            api_key=path_manager.config.models.rerank.api_key,
            model=path_manager.config.models.rerank.model,
            top_n=path_manager.config.hyper.rerank_top_n,
        )
        # set proxies for reranker._client.httpx_client
        proxies_map = {
            "http://": "http://127.0.0.1:7890",
            "https://": "http://127.0.0.1:7890",
        }
        httpx_client = httpx.Client(proxy=proxies_map)
        cohere_client = Client(
            api_key=path_manager.config.models.rerank.api_key,
            httpx_client=httpx_client,
        )
        embedding_reranker._client = cohere_client
    else:
        raise ValueError(
            f"Unknown rerank model series: {path_manager.config.models.rerank.series}"
        )
    return embedding_reranker


def parse_llm_output(content: str):
    try:
        content1 = content.replace("```json", "").replace("```", "")
        res = json.loads(content1)
        return res
    except Exception as e:
        print(content)
        if e.msg == "Expecting ',' delimiter":
            content2 = content1[: e.pos - 1] + '\\"' + content1[e.pos :]
            return parse_llm_output(content2)
        elif e.msg == "Expecting value":
            last_comma_index = content1.rfind(",", 0, e.pos)
            content2 = (
                content1[:last_comma_index] + content1[last_comma_index + 1 :]
            )
            return parse_llm_output(content2)
        elif e.msg == "Invalid \\escape":
            content2 = content1[: e.pos] + "\\\\" + content1[e.pos + 1 :]
            return parse_llm_output(content2)
        elif e.msg == "Invalid control character at":
            if content1[e.pos] == "\n":
                content2 = content1[: e.pos] + '",' + content1[e.pos + 1 :]
            else:
                content2 = (
                    content1[: e.pos]
                    + "(control character)"
                    + content1[e.pos + 1 :]
                )
            return parse_llm_output(content2)
        else:
            print(e)
            raise ValueError("Invalid JSON response from LLM: \n" + content)


def calculate_in_cost(
    text, model_name="gpt-3.5-turbo", price_per_1m_tokens=0.13699
):
    enc = tiktoken.encoding_for_model(model_name)
    tokens = enc.encode(text)
    token_count = len(tokens)
    cost = (token_count / 1000000) * price_per_1m_tokens
    return token_count, cost


def calculate_out_cost(
    text, model_name="gpt-3.5-turbo", price_per_1m_tokens=0.27397
):
    enc = tiktoken.encoding_for_model(model_name)
    tokens = enc.encode(text)
    token_count = len(tokens)
    cost = (token_count / 1000000) * price_per_1m_tokens
    return token_count, cost
