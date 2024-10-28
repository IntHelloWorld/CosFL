from typing import List

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import BaseNode

from Storage.store import HybridStore


def nodes_filter(nodes: List[BaseNode], node_type: str) -> List[BaseNode]:
    nodes_filtered = [node for node in nodes if node.metadata.get("node_type") == node_type]
    if len(nodes_filtered) == 0:
        raise ValueError(f"No {node_type} nodes found")
    return nodes_filtered

def get_method_functionality_index(store: HybridStore):
    method_nodes = nodes_filter(store.embedded_nodes, "method_node")
    return VectorStoreIndex(method_nodes)


def get_method_description_index(method_nodes: List[BaseNode], store: HybridStore, use_context: bool = True):
    desc_nodes = []
    nods_dict = {n.id_: n for n in store.embedded_nodes}
    for method_node in method_nodes:
        if use_context:
            desc_node_ids = method_node.metadata.get("desc_node_ids", [])
        else:
            desc_node_ids = method_node.metadata.get("desc_node_ids_NC", [])
        # if len(desc_node_ids) == 0:
        #     raise ValueError(f"Method node {method_node.id_} has no description node")
        for desc_node_id in desc_node_ids:
            desc_node = nods_dict.get(desc_node_id, None)
            if desc_node is None:
                raise ValueError(f"Description node {desc_node_id} not found")
            desc_nodes.append(desc_node)
    return VectorStoreIndex(desc_nodes)


def get_context_index(store: HybridStore):
    context_nodes = nodes_filter(store.embedded_nodes, "context_node")
    return VectorStoreIndex(context_nodes)
