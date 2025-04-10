"""Hierarchical node parser."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.schema import BaseNode, Document, NodeRelationship
from llama_index.core.utils import get_tqdm_iterable
from tree_sitter import Parser as TreeSitterParser

from Storage.splitter import JavaClassSplitter

LANGUAGE = "java"

def get_leaf_nodes(nodes: List[BaseNode]) -> List[BaseNode]:
    """Get leaf nodes."""
    leaf_nodes = []
    for node in nodes:
        if NodeRelationship.CHILD not in node.relationships:
            leaf_nodes.append(node)
    return leaf_nodes


def get_root_nodes(nodes: List[BaseNode]) -> List[BaseNode]:
    """Get root nodes."""
    root_nodes = []
    for node in nodes:
        if NodeRelationship.PARENT not in node.relationships:
            root_nodes.append(node)
    return root_nodes


def get_child_nodes(nodes: List[BaseNode], all_nodes: List[BaseNode]) -> List[BaseNode]:
    """Get child nodes of nodes from given all_nodes."""
    children_ids = []
    for node in nodes:
        if NodeRelationship.CHILD not in node.relationships:
            continue

        children_ids.extend(
            [r.node_id for r in node.relationships[NodeRelationship.CHILD]]
        )

    child_nodes = []
    for candidate_node in all_nodes:
        if candidate_node.node_id not in children_ids:
            continue
        child_nodes.append(candidate_node)

    return child_nodes


def get_deeper_nodes(nodes: List[BaseNode], depth: int = 1) -> List[BaseNode]:
    """Get children of root nodes in given nodes that have given depth."""
    if depth < 0:
        raise ValueError("Depth cannot be a negative number!")
    root_nodes = get_root_nodes(nodes)
    if not root_nodes:
        raise ValueError("There is no root nodes in given nodes!")

    deeper_nodes = root_nodes
    for _ in range(depth):
        deeper_nodes = get_child_nodes(deeper_nodes, nodes)

    return deeper_nodes


class JavaNodeParser(NodeParser):
    """Java file node parser.

    Splits a java file into a recursive hierarchy Nodes using a NodeParser.

    NOTE: this will return a hierarchy of nodes in a flat list, where there will be
    overlap between parent nodes (e.g. with a bigger chunk size), and child nodes
    per parent (e.g. with a smaller chunk size).

    For instance, this may return a list of nodes like:
    - list of top-level nodes with chunk size 2048
    - list of second-level nodes, where each node is a child of a top-level node,
      chunk size 512
    - list of third-level nodes, where each node is a child of a second-level node,
      chunk size 128
    """

    chunk_size: Optional[int] = Field(
        default=None,
        description=(
            "The chunk sizes to use when splitting code."
        ),
    )
    node_parser_map: Dict[str, NodeParser] = Field(
        description="Map of node parser id to node parser.",
    )
    
    parser: TreeSitterParser = Field(
        description="Tree-sitter parser object.",
    )

    @classmethod
    def from_defaults(
        cls,
        chunk_size: Optional[int] = None,
        node_parser_map: Optional[Dict[str, NodeParser]] = None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
    ) -> "JavaNodeParser":
        callback_manager = callback_manager or CallbackManager([])
        from tree_sitter import Parser  # pants: no-infer-dep

        try:
            import tree_sitter_languages  # pants: no-infer-dep
            parser = tree_sitter_languages.get_parser(LANGUAGE)
        except ImportError:
            raise ImportError(
                "Please install tree_sitter_languages to use JavaClassSplitter."
                "Or pass in a parser object."
            )
        except Exception:
            print(
                f"Could not get parser for language {LANGUAGE}. Check "
                "https://github.com/grantjenks/py-tree-sitter-languages#license "
                "for a list of valid languages."
            )
            raise
        if not isinstance(parser, Parser):
            raise ValueError("Parser must be a tree-sitter Parser object.")

        node_parser_map = {
            "java_class_splitter": JavaClassSplitter.from_defaults(),
        }

        return cls(
            chunk_size=chunk_size,
            node_parser_map=node_parser_map,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
            parser=parser,
        )

    @classmethod
    def class_name(cls) -> str:
        return "JavaNodeParser"

    def _recursively_get_nodes_from_nodes(
        self,
        nodes: List[BaseNode],
    ) -> List[BaseNode]:
        """Recursively get nodes from nodes."""

        # first split current nodes into sub-nodes
        sub_nodes = []
        for node in nodes:
            if node.metadata["node_type"] == "method_node":
                cur_sub_nodes = []
            else:
                cur_sub_nodes = self.node_parser_map["java_class_splitter"].get_nodes_from_documents([node])
            # add parent relationship from sub node to parent node
            # add child relationship from parent node to sub node
            # for sub_node in cur_sub_nodes:
            #     _add_parent_child_relationship(parent_node=node,
            #                                    child_node=sub_node)

            sub_nodes.extend(cur_sub_nodes)
            
        if not sub_nodes:
            return []
        # now for each sub-node, recursively split into sub-sub-nodes, and add
        sub_sub_nodes = self._recursively_get_nodes_from_nodes(sub_nodes)

        return sub_nodes + sub_sub_nodes

    def get_nodes_from_documents(
        self,
        documents: Sequence[Document],
        classes: List[str],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Parse document into nodes.

        Args:
            documents (Sequence[Document]): documents to parse
            include_metadata (bool): whether to include metadata in nodes

        """
        with self.callback_manager.event(
            CBEventType.NODE_PARSING, payload={EventPayload.DOCUMENTS: documents}
        ) as event:
            all_nodes: List[BaseNode] = []
            documents_with_progress = get_tqdm_iterable(
                documents, show_progress, "Parsing documents into nodes"
            )

            for doc in documents_with_progress:
                class_name = doc.metadata["file_name"].split(".")[0]
                if class_name not in classes:
                    continue
                
                # parse code into AST
                tree = self.parser.parse(bytes(doc.text, "utf-8"))
                doc.metadata.update({"ast": tree.root_node,
                                     "source": doc.text.split("\n"),
                                     "node_type": "document_node"})
                
                nodes_from_doc = self._recursively_get_nodes_from_nodes([doc])
                all_nodes.extend(nodes_from_doc)
            
            # delete AST from metadata
            excluded_keys = ["ast", "source"]
            for node in all_nodes:
                for key in excluded_keys:
                    node.metadata.pop(key, None)
                    
                    for child in node.relationships.get(NodeRelationship.CHILD, []):
                        child.metadata.pop(key, None)
                        
                    parent_node = node.relationships.get(NodeRelationship.PARENT, [])
                    if parent_node:
                        parent_node.metadata.pop(key, None)

            event.on_end(payload={EventPayload.NODES: all_nodes})

        return all_nodes

    # Unused abstract method
    def _parse_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        return list(nodes)
