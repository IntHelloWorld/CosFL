"""Java class splitter."""

import sys
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.callbacks import CallbackManager, CBEventType, EventPayload
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.schema import BaseNode, Document
from llama_index.core.utils import get_tqdm_iterable
from pydantic import Field
from tree_sitter import Node as TreeSitterNode

sys.path.append(Path(__file__).resolve().parents[1].as_posix())
from preprocess.my_utils import (
    CLASS_TYPES,
    METHOD_TYPES,
    build_nodes_from_splits,
    default_id_func,
)


class ASTSplitter(NodeParser):
    @abstractmethod
    def split_node(self, node) -> List[str]:
        ...

    def _parse_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")
        for node in nodes_with_progress:
            splits = self.split_node(node)

            all_nodes.extend(
                build_nodes_from_splits(splits, node, id_func=self.id_func)
            )

        return all_nodes

class JavaClassSplitter(ASTSplitter):
    """Split a java class using a AST parser.
    """

    _parser: Any = PrivateAttr()

    def __init__(
        self,
        callback_manager: Optional[CallbackManager] = None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        id_func: Optional[Callable[[int, Document], str]] = None,
    ) -> None:
        """Initialize a JavaClassSplitter."""

        callback_manager = callback_manager or CallbackManager([])
        id_func = id_func or default_id_func

        super().__init__(
            callback_manager=callback_manager,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            id_func=id_func,
        )

    @classmethod
    def from_defaults(
        cls,
        callback_manager: Optional[CallbackManager] = None,
    ) -> "JavaClassSplitter":
        """Create a JavaClassSplitter with default values."""
        return cls()

    @classmethod
    def class_name(cls) -> str:
        return "JavaClassSplitter"
    
    def get_source_with_comment(self, node: TreeSitterNode, source: List[str]) -> str:
        if "comment" in node.prev_sibling.type:
            return "\n".join(source[node.prev_sibling.start_point[0]: node.end_point[0] + 1])
        else:
            return "\n".join(source[node.start_point[0]: node.end_point[0] + 1])

    def split_node(self, node: BaseNode) -> List[Tuple[TreeSitterNode, str]]:
        """Split incoming code and return chunks using the AST."""
        with self.callback_manager.event(
            CBEventType.CHUNKING, payload={EventPayload.CHUNKS: [node.text]}
        ) as event:
            ast_node: TreeSitterNode= node.metadata["ast"]
            source: List[str] = node.metadata["source"]
            
            results = []
            
            if ast_node.type == "program":
                child_nodes = ast_node.children
            else:
                child_nodes = ast_node.child_by_field_name("body").children
            
            for child in child_nodes:
                if child.type in CLASS_TYPES:
                    text = self.get_source_with_comment(child, source)
                    results.append((child, text))
                elif child.type in METHOD_TYPES:
                    if child.child_by_field_name("body"):  # check if method has a body
                        if child.child_by_field_name("body").named_child_count:
                            text = self.get_source_with_comment(child, source)
                            results.append((child, text))
            event.on_end(
                payload={EventPayload.CHUNKS: results},
            )

            return results