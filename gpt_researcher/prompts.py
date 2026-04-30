import warnings
from datetime import date, datetime, timezone

from langchain_core.documents import Document

from .config import Config
from .utils.enum import ReportSource, ReportType, Tone
from .utils.enum import PromptFamily as PromptFamilyEnum
from typing import Callable, List, Dict, Any


## Prompt Families #############################################################

class PromptFamily:
    """General purpose class for prompt formatting."""

    def __init__(self, config: Config):
        self.cfg = config

    @staticmethod
    def generate_mcp_tool_selection_prompt(query: str, tools_info: List[Dict], max_tools: int = 3) -> str:
        import json
        return f"""You are a research assistant helping to select the most relevant tools for a research query.
RESEARCH QUERY: "{query}"
AVAILABLE TOOLS:
{json.dumps(tools_info, indent=2)}
TASK: Analyze the tools and select EXACTLY {max_tools} tools.
Return a JSON object with this exact format:
{{
  "selected_tools": [
    {{
      "index": 0,
      "name": "tool_name",
      "relevance_score": 9,
      "reason": "..."
    }}
  ],
  "selection_reasoning": "..."
}}
"""

    @staticmethod
    def generate_mcp_research_prompt(query: str, selected_tools: List) -> str:
        tool_names = [getattr(tool, 'name', str(tool)) for tool in selected_tools]
        return f"""You are a research assistant researching: "{query}"
AVAILABLE TOOLS: {tool_names}
Please conduct thorough research and provide findings."""

    @staticmethod
    def generate_image_analysis_prompt(query: str, sections: List[Dict[str, Any]], max_images: int = 3) -> str:
        sections_text = "\n\n".join([f"### Section {i+1}: {s['header']}\n{s['content'][:500]}..." for i, s in enumerate(sections)])
        return f"""Analyze report sections for visual needs. TOPIC: {query}\nSECTIONS:\n{sections_text}\nReturn JSON only."""

    @staticmethod
    def generate_image_prompt_enhancement(base_prompt: str, section_content: str, research_topic: str) -> str:
        return f"""Create illustration for: {research_topic}\nDESCRIPTION: {base_prompt}\nCONTEXT: {section_content[:800]}"""

    @staticmethod
    def generate_search_queries_prompt(question: str, parent_query: str, report_type: str, max_iterations: int = 3, context: List[Dict[str, Any]] = []):
        task = f"{parent_query} - {question}" if report_type in [ReportType.DetailedReport.value, ReportType.SubtopicReport.value] else question
        context_prompt = f"Context: {context}" if context else ""
        dynamic_example = ", ".join([f'"query {i+1}"' for i in range(max_iterations)])
        return f"""Write {max_iterations} google search queries for: "{task}"\nDate: {datetime.now(timezone.utc).strftime('%B %d, %Y')}\n{context_prompt}\nRespond with: [{dynamic_example}]"""

    @staticmethod
    def generate_report_prompt(question: str, context: str, report_source: str, report_format="apa", total_words=1000, tone=None, language="english"):
        if report_source == ReportSource.Web.value:
            reference_prompt = f"""
You MUST write all used source urls at the end of the report as references, and make sure to not add duplicated sources, but only one reference for each.
Every url should be hyperlinked: [url website](url)
Additionally, you MUST include hyperlinks to the relevant URLs wherever they are referenced in the report:
eg: Author, A. A. (Year, Month Date). Title of web page. Website Name. [url website](url)
"""
        else:
            reference_prompt = f"""
You MUST write all used source document names at the end of the report as references, and make sure to not add duplicated sources, but only one reference for each.
"""
        tone_prompt = f"Write the report in a {tone.value} tone." if tone else ""

        return f"""
Information: "{context}"

Using the above information, answer the following query or task: "{question}" in a detailed report.
The report should focus on the integration of Large Language Models (LLMs) and Traditional Chinese Medicine (TCM), specifically analyzing 'Pattern Differentiation' (辨证论治) and clinical evidence from Nature/PubMed.

Please follow all of the following guidelines in your report:
- You MUST determine your own concrete and valid opinion based on the given information. Do NOT defer to general and meaningless conclusions.
- You MUST write the report with markdown syntax and {report_format} format.
- Structure your report with clear markdown headers: use # for the main title, ## for major sections, and ### for subsections.
- Use markdown tables when presenting structured data or comparisons to enhance readability.
- You MUST prioritize the relevance, reliability, and significance of sources from Nature, PubMed, and Lancet.
- Use in-text citation references like this: ([in-text citation](url)).
- Do not forget to add a reference list at the end of the report.

LANGUAGE CRITICAL RULE:
- You MUST write the report in professional academic CHINESE (简体中文), regardless of the input language or source material.
- If a language preference is specified as {language}, ignore it if it is not Chinese. 
- Please translate all technical TCM terms accurately.

{reference_prompt}
{tone_prompt}

Assume that the current date is {date.today()}.
The report should be a minimum of {total_words} words.
"""

    @staticmethod
    def curate_sources(query, sources, max_results=10):
        return f"""Evaluate content for: "{query}" in TCM/AI context. 
- Prioritize Clinical evidence (Nature, PubMed, Lancet).
- Retain 'Pattern Differentiation' (辨证论治) content.
- Return JSON list format only.
SOURCES:
{sources}
"""

    @staticmethod
    def generate_resource_report_prompt(question, context, report_source: str, report_format="apa", tone=None, total_words=1000, language="english"):
        if report_source == ReportSource.Web.value:
            ref = "Include all source urls: [url website](url)"
        else:
            ref = "Include source document names as references."
        return f'"""{context}"""\nBibliography for: "{question}". Lang: {language}. Min words: {total_words}. {ref}'

    @staticmethod
    def generate_custom_report_prompt(query_prompt, context, report_source: str, report_format="apa", tone=None, total_words=1000, language: str = "english"):
        return f'"{context}"\n\n{query_prompt}'

    @staticmethod
    def generate_outline_report_prompt(question, context, report_source: str, report_format="apa", tone=None, total_words=1000, language: str = "english"):
        return f'"""{context}"""\nGenerate Markdown outline for: "{question}". Min words: {total_words}.'

    @staticmethod
    def generate_deep_research_prompt(question, context, report_source, report_format="apa", tone=None, total_words=2000, language="english"):
        if report_source == ReportSource.Web.value:
            ref = "Include source urls: [url website](url)"
        else:
            ref = "Include source document names."
        tp = f"Tone: {tone.value}" if tone else ""
        return f"""Deep research for: "{question}"\nContext: {context}\nMin words: {total_words}. Lang: {language}. {tp}\n{ref}"""

    @staticmethod
    def auto_agent_instructions():
        return "Task involves researching a topic. Categorize by field and emoji."

    @staticmethod
    def generate_summary_prompt(query, data):
        return f'{data}\nSummarize based on: "{query}". Include facts, stats.'

    @staticmethod
    def generate_quick_summary_prompt(query: str, context: str) -> str:
        return f'Synthesize answer for: "{query}"\nResults: {context}'

    @staticmethod
    def pretty_print_docs(docs: list[Document], top_n: int | None = None) -> str:
        return "\n".join(f"Source: {d.metadata.get('source')}\nTitle: {d.metadata.get('title')}\nContent: {d.page_content}" for i, d in enumerate(docs) if top_n is None or i < top_n)

    @staticmethod
    def join_local_web_documents(docs_context: str, web_context: str) -> str:
        return f"Local: {docs_context}\n\nWeb: {web_context}"

    @staticmethod
    def generate_subtopics_prompt() -> str:
        return "Construct list of subtopics for: {task}\nData: {data}\n{format_instructions}"

    @staticmethod
    def generate_subtopic_report_prompt(current_subtopic, existing_headers, relevant_written_contents, main_topic, context, report_format="apa", max_subsections=5, total_words=800, tone=Tone.Objective, language="english"):
        return f"Context: {context}\nSubtopic: {current_subtopic} (Main: {main_topic})\nLang: {language}. Min words: {total_words}."

    @staticmethod
    def generate_draft_titles_prompt(current_subtopic, main_topic, context, max_subsections=5):
        return f"Draft headers for: {current_subtopic} (Main: {main_topic})\nContext: {context}"

    @staticmethod
    def generate_report_introduction(question, research_summary="", language="english", report_format="apa"):
        return f"Intro for: {question}\nSummary: {research_summary}\nLang: {language}."

    @staticmethod
    def generate_report_conclusion(query, report_content, language="english", report_format="apa"):
        return f"Conclusion for: {query}\nReport: {report_content}\nLang: {language}."

# Granite classes and Factory remain unchanged to ensure system compatibility...
class GranitePromptFamily(PromptFamily):
    def _get_granite_class(self) -> type[PromptFamily]:
        if "3.3" in self.cfg.smart_llm: return Granite33PromptFamily
        if "3" in self.cfg.smart_llm: return Granite3PromptFamily
        return PromptFamily
    def pretty_print_docs(self, *args, **kwargs) -> str: return self._get_granite_class().pretty_print_docs(*args, **kwargs)
    def join_local_web_documents(self, *args, **kwargs) -> str: return self._get_granite_class().join_local_web_documents(*args, **kwargs)

class Granite3PromptFamily(PromptFamily):
    _DOCUMENTS_PREFIX = "<|start_of_role|>documents<|end_of_role|>\n"
    _DOCUMENTS_SUFFIX = "\n<|end_of_text|>"
    @classmethod
    def pretty_print_docs(cls, docs, top_n=None):
        all_docs = "\n\n".join([f"Document {doc.metadata.get('source', i)}\nTitle: {doc.metadata.get('title')}\n{doc.page_content}" for i, doc in enumerate(docs) if top_n is None or i < top_n])
        return f"{cls._DOCUMENTS_PREFIX}{all_docs}{cls._DOCUMENTS_SUFFIX}"
    @classmethod
    def join_local_web_documents(cls, docs, web): return f"{cls._DOCUMENTS_PREFIX}{docs}\n\n{web}{cls._DOCUMENTS_SUFFIX}"

class Granite33PromptFamily(PromptFamily):
    _DOCUMENT_TEMPLATE = '<|start_of_role|>document {{"document_id": "{document_id}"}}<|end_of_role|>\n{document_content}<|end_of_text|>\n'
    @staticmethod
    def _get_content(doc): return f"Title: {doc.metadata.get('title')}\n{doc.page_content}".strip()
    @classmethod
    def pretty_print_docs(cls, docs, top_n=None): return "\n".join([cls._DOCUMENT_TEMPLATE.format(document_id=doc.metadata.get("source", i), document_content=cls._get_content(doc)) for i, doc in enumerate(docs) if top_n is None or i < top_n])
    @classmethod
    def join_local_web_documents(cls, docs, web): return f"{docs}\n\n{web}"

# Factory mapping
report_type_mapping = {
    ReportType.ResearchReport.value: "generate_report_prompt",
    ReportType.ResourceReport.value: "generate_resource_report_prompt",
    ReportType.OutlineReport.value: "generate_outline_report_prompt",
    ReportType.CustomReport.value: "generate_custom_report_prompt",
    ReportType.SubtopicReport.value: "generate_subtopic_report_prompt",
    ReportType.DeepResearch.value: "generate_deep_research_prompt",
}

def get_prompt_by_report_type(report_type, prompt_family):
    prompt_by_type = getattr(prompt_family, report_type_mapping.get(report_type, ""), None)
    if not prompt_by_type:
        prompt_by_type = getattr(prompt_family, report_type_mapping.get(ReportType.ResearchReport.value))
    return prompt_by_type

prompt_family_mapping = {
    PromptFamilyEnum.Default.value: PromptFamily,
    PromptFamilyEnum.Granite.value: GranitePromptFamily,
    PromptFamilyEnum.Granite3.value: Granite3PromptFamily,
    PromptFamilyEnum.Granite31.value: Granite3PromptFamily,
    PromptFamilyEnum.Granite32.value: Granite3PromptFamily,
    PromptFamilyEnum.Granite33.value: Granite33PromptFamily,
}

def get_prompt_family(prompt_family_name, config):
    if isinstance(prompt_family_name, PromptFamilyEnum): prompt_family_name = prompt_family_name.value
    family = prompt_family_mapping.get(prompt_family_name, PromptFamily)
    return family(config)