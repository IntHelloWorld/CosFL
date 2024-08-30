from typing import List

from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.postprocessor.jinaai_rerank import JinaRerank
from llama_index.postprocessor.voyageai_rerank import VoyageAIRerank
from tenacity import retry, stop_after_attempt, wait_fixed

API_URL = "https://api.jina.ai/v1/rerank"

def my_get_content(node: BaseNode):
    return node.metadata["summary"]


@retry(wait=wait_fixed(1), stop=stop_after_attempt(3))
def postprocess_nodes(reranker, nodes: List[NodeWithScore], query_str: str = None) -> List[NodeWithScore]:
    if isinstance(reranker, CohereRerank):
        texts = [my_get_content(node.node) for node in nodes]
        results = reranker._client.rerank(
            model=reranker.model,
            top_n=reranker.top_n,
            query=query_str,
            documents=texts,
        )

        new_nodes = []
        for result in results.results:
            new_node_with_score = NodeWithScore(
                node=nodes[result.index].node, score=result.relevance_score
            )
            new_nodes.append(new_node_with_score)

    elif isinstance(reranker, JinaRerank):
        texts = [my_get_content(node.node) for node in nodes]
        resp = reranker._session.post(  # type: ignore
            API_URL,
            json={
                "query": query_str,
                "documents": texts,
                "model": reranker.model,
                "top_n": reranker.top_n,
            },
        ).json()
        if "results" not in resp:
            raise RuntimeError(resp["detail"])

        results = resp["results"]

        new_nodes = []
        for result in results:
            new_node_with_score = NodeWithScore(
                node=nodes[result["index"]].node, score=result["relevance_score"]
            )
            new_nodes.append(new_node_with_score)
    
    elif isinstance(reranker, VoyageAIRerank):
        texts = [my_get_content(node.node) for node in nodes]
        results = reranker._client.rerank(
            model=reranker.model,
            top_k=reranker.top_k,
            query=query_str,
            documents=texts,
            truncation=reranker.truncation,
        )

        new_nodes = []
        for result in results.results:
            new_node_with_score = NodeWithScore(
                node=nodes[result.index].node, score=result.relevance_score
            )
            new_nodes.append(new_node_with_score)
    
    else:
        raise ValueError("Invalid reranker type")

    return new_nodes