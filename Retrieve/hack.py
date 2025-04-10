from typing import List

from llama_index.core.schema import NodeWithScore
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.postprocessor.jinaai_rerank import JinaRerank
from tenacity import retry, stop_after_attempt, wait_fixed

from Storage.node_utils import get_node_text_for_embedding

API_URL = "https://api.jina.ai/v1/rerank"


@retry(wait=wait_fixed(1), stop=stop_after_attempt(3))
def postprocess_nodes(
    reranker,
    nodes: List[NodeWithScore],
    query_str: str = None,
    use_context: bool = True,
) -> List[NodeWithScore]:
    if isinstance(reranker, CohereRerank):
        texts = [
            get_node_text_for_embedding(node.node, use_context)
            for node in nodes
        ]
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
        texts = [
            get_node_text_for_embedding(node.node, use_context)
            for node in nodes
        ]
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
                node=nodes[result["index"]].node,
                score=result["relevance_score"],
            )
            new_nodes.append(new_node_with_score)

    else:
        raise ValueError("Invalid reranker type")

    return new_nodes
