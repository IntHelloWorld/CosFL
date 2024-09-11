"""General node utils."""

import hashlib
import sys
from pathlib import Path
from typing import List, Optional, Protocol, Tuple, runtime_checkable

from llama_index.core.schema import BaseNode, NodeRelationship
from tree_sitter import Node as TreeSitterNode

sys.path.append(Path(__file__).resolve().parents[1].as_posix())
from Storage.code_node import CodeNode

CLASS_TYPES = ["class_declaration"]
METHOD_TYPES = ["method_declaration", "constructor_declaration"]


@runtime_checkable
class IdFuncCallable(Protocol):
    def __call__(self, i: int, doc: BaseNode) -> str: ...


def default_id_func(i: int, text: str) -> str:
    sha256 = hashlib.sha256()
    sha256.update(text.encode("utf-8"))
    unique_id = sha256.hexdigest()
    return unique_id


def get_ast_node_type(node: TreeSitterNode) -> str:
    """Get the type of the AST node."""
    if node.type in CLASS_TYPES:
        return "class_node"
    elif node.type in METHOD_TYPES:
        return "method_node"
    else:
        raise ValueError(f"Unknown node type: {node.type}")


def get_method_name(method_declaration: TreeSitterNode) -> str:
    for child in method_declaration.children:
        if child.type == "identifier":
            return bytes.decode(child.text)
    raise ValueError(f"Method declaration does not contain an identifier: {method_declaration}")


def build_nodes_from_splits(
    node_splits: List[TreeSitterNode],
    document: BaseNode,
    ref_doc: Optional[BaseNode] = None,
    id_func: Optional[IdFuncCallable] = None,
) -> List[CodeNode]:
    """Build nodes from splits."""
    ref_doc = ref_doc or document
    id_func = id_func or default_id_func
    nodes: List[CodeNode] = []
    """Calling as_related_node_info() on a document recomputes the hash for the whole text and metadata"""
    """It is not that bad, when creating relationships between the nodes, but is terrible when adding a relationship"""
    """between the node and a document, hence we create the relationship only once here and pass it to the nodes"""
    # relationships = {NodeRelationship.PARENT: ref_doc.as_related_node_info()}
    for i, ast in enumerate(node_splits):
        source = bytes.decode(ast.text)
        comment = ""
        if ast.prev_sibling and "comment" in ast.prev_sibling.type:
            comment = bytes.decode(ast.prev_sibling.text)

        node = CodeNode(
            id_=id_func(i, source),
            text=source,
            embedding=document.embedding,
            excluded_embed_metadata_keys=document.excluded_embed_metadata_keys,
            excluded_llm_metadata_keys=document.excluded_llm_metadata_keys,
            metadata_seperator=document.metadata_seperator,
            metadata_template=document.metadata_template,
            text_template=document.text_template,
        )

        new_node_type = get_ast_node_type(ast)
        node.metadata.update(
            {
                "ast": ast,
                "node_type": new_node_type,
                "source": document.metadata["source"],
                "file_path": document.metadata["file_path"],
            }
        )

        if new_node_type == "method_node":
            node.metadata.update(
                {
                    "start_line": ast.start_point[0],
                    "end_line": ast.end_point[0] + 1,
                    "comment": comment,
                    "method_name": get_method_name(ast),
                }
            )

        nodes.append(node)

    return nodes
