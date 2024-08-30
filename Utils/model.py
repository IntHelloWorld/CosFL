import sys
from pathlib import Path
from socket import timeout

import httpx
from llama_index.core import Settings
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.dashscope import DashScope
from llama_index.llms.deepinfra import DeepInfraLLM
from llama_index.llms.lmstudio import LMStudio
from llama_index.llms.openai import OpenAI

sys.path.append(Path(__file__).resolve().parents[1].as_posix())
from Utils.path_manager import PathManager

DEFAULT_TIMEOUT = 120

def set_models(path_manager: PathManager):
    # set embedding model
    if "embed" in path_manager.config:
        if path_manager.config["embed"]["embed_series"] == "openai":
            Settings.embed_model = OpenAIEmbedding(
                model=path_manager.config["embed"]["embed_model"],
                api_key=path_manager.config["embed"]["embed_api_key"],
                api_base=path_manager.config["embed"]["embed_base_url"],
                embed_batch_size=path_manager.config["embed"]["embed_batch_size"],
            )
        elif path_manager.config["embed"]["embed_series"] == "jina":
            Settings.embed_model = JinaEmbedding(
                api_key=path_manager.config["embed"]["embed_api_key"],
                model=path_manager.config["embed"]["embed_model"],
                embed_batch_size=path_manager.config["embed"]["embed_batch_size"],
            )
        elif path_manager.config["embed"]["embed_series"] == "voyage":
            Settings.embed_model = VoyageEmbedding(
                model_name=path_manager.config["embed"]["embed_model"],
                voyage_api_key=path_manager.config["embed"]["embed_api_key"],
                embed_batch_size=path_manager.config["embed"]["embed_batch_size"],
            )
        elif path_manager.config["embed"]["embed_series"] == "cohere":
            Settings.embed_model = CohereEmbedding(
                model_name=path_manager.config["embed"]["embed_model"],
                cohere_api_key=path_manager.config["embed"]["embed_api_key"],
                embed_batch_size=path_manager.config["embed"]["embed_batch_size"]
            )
    
    # set summary model
    if "summary" in path_manager.config:
        if path_manager.config["summary"]["summary_series"] == "lmstudio":
            Settings.llm = LMStudio(
                model_name=path_manager.config["summary"]["summary_model"],
                base_url=path_manager.config["summary"]["summary_base_url"],
                timeout=DEFAULT_TIMEOUT
            )
        elif path_manager.config["summary"]["summary_series"] == "openai":
            Settings.llm = OpenAI(
                model=path_manager.config["summary"]["summary_model"],
                api_key=path_manager.config["summary"]["summary_api_key"],
                api_base=path_manager.config["summary"]["summary_base_url"],
            )
    
    # set reasoning model
    if "reason" in path_manager.config:
        if path_manager.config["reason"]["reasoning_series"] == "lmstudio":
            path_manager.reasoning_llm = LMStudio(
                model_name=path_manager.config["reason"]["reasoning_model"],
                base_url=path_manager.config["reason"]["reasoning_base_url"],
                timeout=DEFAULT_TIMEOUT
            )
        elif path_manager.config["reason"]["reasoning_series"] == "openai":
            path_manager.reasoning_llm = OpenAI(
                model=path_manager.config["reason"]["reasoning_model"],
                api_key=path_manager.config["reason"]["reasoning_api_key"],
                api_base=path_manager.config["reason"]["reasoning_base_url"],
                timeout=DEFAULT_TIMEOUT
            )
        elif path_manager.config["reason"]["reasoning_series"] == "dashscope":
            path_manager.reasoning_llm = DashScope(
                model_name=path_manager.config["reason"]["reasoning_model"],
                api_key=path_manager.config["reason"]["reasoning_api_key"],
                timeout=DEFAULT_TIMEOUT,
                max_tokens=path_manager.config["reason"]["reasoning_max_tokens"]
            )
        elif path_manager.config["reason"]["reasoning_series"] == "anthropic":
            path_manager.reasoning_llm = Anthropic(
                model=path_manager.config["reason"]["reasoning_model"],
                api_key=path_manager.config["reason"]["reasoning_api_key"],
                base_url=path_manager.config["reason"]["reasoning_base_url"],
                max_tokens=path_manager.config["reason"]["reasoning_max_tokens"],
                timeout=DEFAULT_TIMEOUT
            )
        elif path_manager.config["reason"]["reasoning_series"] == "deepinfra":
            path_manager.reasoning_llm = DeepInfraLLM(
                model=path_manager.config["reason"]["reasoning_model"],
                api_key=path_manager.config["reason"]["reasoning_api_key"],
                max_tokens=path_manager.config["reason"]["reasoning_max_tokens"],
                timeout=DEFAULT_TIMEOUT
           )