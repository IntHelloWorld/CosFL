import sys
from pathlib import Path

from llama_index.core import Settings
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.lmstudio import LMStudio
from llama_index.llms.openai import OpenAI

sys.path.append(Path(__file__).resolve().parents[1].as_posix())
from Utils.path_manager import PathManager

DEFAULT_TIMEOUT = 120

def set_models(path_manager: PathManager):
    # set embedding model
    if path_manager.embed_series == "openai":
        Settings.embed_model = OpenAIEmbedding(
            model=path_manager.embed_model,
            api_key=path_manager.embed_api_key,
            api_base=path_manager.embed_base_url
        )
    elif path_manager.embed_series == "jina":
        Settings.embed_model = JinaEmbedding(
            api_key=path_manager.embed_api_key,
            model=path_manager.embed_model,
        )
    
    # set summary model
    if path_manager.summary_series == "lmstudio":
        Settings.llm = LMStudio(
            model_name=path_manager.summary_model,
            base_url=path_manager.summary_base_url,
            timeout=DEFAULT_TIMEOUT
        )
    elif path_manager.summary_series == "openai":
        Settings.llm = OpenAI(
            model=path_manager.summary_model,
            api_key=path_manager.summary_api_key,
            api_base=path_manager.summary_base_url
        )
    
    # set reasoning model
    if path_manager.reasoning_series == "lmstudio":
        path_manager.reasoning_llm = LMStudio(
            model_name=path_manager.reasoning_model,
            base_url=path_manager.reasoning_base_url,
            timeout=DEFAULT_TIMEOUT
        )
    elif path_manager.reasoning_series == "openai":
        path_manager.reasoning_llm = OpenAI(
            model=path_manager.reasoning_model,
            api_key=path_manager.reasoning_api_key,
            api_base=path_manager.reasoning_base_url,
            timeout=DEFAULT_TIMEOUT
        )

def set_summary_model(path_manager: PathManager):
    if path_manager.summary_series == "lmstudio":
        Settings.llm = LMStudio(
            model_name=path_manager.summary_model,
            base_url=path_manager.summary_base_url,
            timeout=DEFAULT_TIMEOUT,
            request_timeout=DEFAULT_TIMEOUT
        )
    elif path_manager.summary_series == "openai":
        Settings.llm = OpenAI(
            model=path_manager.summary_model,
            api_key=path_manager.summary_api_key,
            api_base=path_manager.summary_base_url
        )