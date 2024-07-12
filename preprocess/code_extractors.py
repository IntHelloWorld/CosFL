import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from llama_index.core.async_utils import DEFAULT_NUM_WORKERS, run_jobs
from llama_index.core.bridge.pydantic import Field
from llama_index.core.extractors.interface import BaseExtractor
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.service_context_elements.llm_predictor import LLMPredictorType
from llama_index.core.settings import Settings

sys.path.append(Path(__file__).resolve().parents[1].as_posix())
from preprocess.code_node import CodeNode

DEFAULT_SUMMARY_EXTRACT_TEMPLATE = """\
Below is the source code of a method in {language} programming language:
```{language}
{code_str}
```

Please summarize the functionality of the above method in no more than {summary_length} words.\
"""


class CodeSummaryExtractor(BaseExtractor):
    """
    Summary extractor. Node-level extractor with adjacent sharing.
    Extracts `section_summary`, `prev_section_summary`, `next_section_summary`
    metadata fields.

    Args:
        llm (Optional[LLM]): LLM
        summaries (List[str]): list of summaries to extract: 'self', 'prev', 'next'
        prompt_template (str): template for summary extraction
    """

    language: str = Field(description="The program language.")
    llm: LLMPredictorType = Field(description="The LLM to use for generation.")
    prompt_template: str = Field(
        default=DEFAULT_SUMMARY_EXTRACT_TEMPLATE,
        description="Template to use when generating summaries.",
    )

    def __init__(
        self,
        language: str,
        llm: Optional[LLM] = None,
        # TODO: llm_predictor arg is deprecated
        llm_predictor: Optional[LLMPredictorType] = None,
        prompt_template: str = DEFAULT_SUMMARY_EXTRACT_TEMPLATE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        **kwargs: Any,
    ):
        super().__init__(
            language=language,
            llm=llm or llm_predictor or Settings.llm,
            metadata_mode=MetadataMode.LLM,
            prompt_template=prompt_template,
            num_workers=num_workers,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "SummaryExtractor"
    
    def get_dynamic_summary_length(self, code_str: str) -> int:
        """Get the dynamic summary length based on the length of the context."""
        n_lines = code_str.count("\n") + 1
        for i in range(10, 60, 10):
            if n_lines <= i:
                return i * 5
        else:
            return 300

    async def _agenerate_node_summary(self, node: BaseNode) -> str:
        """Generate a summary for a code node."""
        if self.is_text_node_only and not isinstance(node, CodeNode):
            raise ValueError("Only `CodeNode` is allowed for `CodeSummaryExtractor` extractor")

        code_str = node.get_content(metadata_mode=self.metadata_mode)
        summary_length = self.get_dynamic_summary_length(code_str)
        summary = await self.llm.apredict(PromptTemplate(template=self.prompt_template),
                                          language=self.language,
                                          summary_length=summary_length,
                                          code_str=code_str)
        # summary = "test summary"

        return summary.strip()

    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        if not all(isinstance(node, CodeNode) for node in nodes):
            raise ValueError("Only `CodeNode` is allowed for `Summary` extractor")

        node_summaries_jobs = []
        for node in nodes:
            node_summaries_jobs.append(self._agenerate_node_summary(node))

        node_summaries = await run_jobs(
            node_summaries_jobs,
            show_progress=self.show_progress,
            workers=self.num_workers,
        )

        # Extract node-level summary metadata
        metadata_list: List[Dict] = [{"summary": s} for s in node_summaries]
        return metadata_list
