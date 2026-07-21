# ============================================================
# DAY 2 LAB — Research Agent (PATCHED for upstream 502 / rate limits)
# ============================================================
# Changes vs. your skeleton:
#   * Added invoke_with_retry() — exponential backoff on transient
#     upstream errors (502 / ResourceExhausted / 429 / overloaded).
#   * analyze_node, evaluate_node, report_node now route every LLM
#     call through it.
#   * analyze_node sleeps briefly between sources so the per-source
#     loop stops bursting the shared endpoint.
#   * TavilySearch max_results 5 -> 3 (fewer analyze calls per pass).
# Graph structure is UNCHANGED — your wiring was already correct.
# ============================================================

import os
import time
import operator
from datetime import datetime
from typing import Annotated, List, Dict
from typing_extensions import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


# ============================================================
# STEP 1 — THE STATE
# ============================================================
class AgentState(TypedDict):
    topic: str
    search_query: str
    collected_data: List[Dict]
    analyzed_data: List[Dict]
    quality_score: int
    iteration_count: int
    final_report: str
    execution_logs: Annotated[List[str], operator.add]


# ============================================================
# STEP 2 — MODEL, SEARCH TOOL, EMBEDDINGS
# ============================================================
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
    temperature=0,
    base_url=os.getenv("LLM_BASE_URL"),
)

search_tool = TavilySearch(max_results=3)  # was 5 — fewer calls per pass

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = InMemoryVectorStore(embedding=embedding_model)


# ============================================================
# RESILIENCE HELPER — retry transient upstream failures
# ============================================================
# The Nvidia/OpenAI-compatible gateway returns 502 ResourceExhausted
# when its worker request budget is saturated. Those are transient:
# waiting and retrying usually clears them. We retry ONLY on errors
# that look transient; anything else (bad request, auth) re-raises
# immediately so you see the real bug.
_TRANSIENT_MARKERS = (
    "resourceexhausted",
    "502",
    "503",
    "rate limit",
    "429",
    "overloaded",
    "timeout",
    "temporarily",
)


def invoke_with_retry(model, messages, retries=5, base_delay=1.0):
    """Invoke an LLM/runnable, retrying with exponential backoff on transient errors."""
    for attempt in range(retries):
        try:
            return model.invoke(messages)
        except Exception as e:
            msg = str(e).lower()
            transient = any(marker in msg for marker in _TRANSIENT_MARKERS)
            if attempt == retries - 1 or not transient:
                raise
            wait = base_delay * (2 ** attempt)  # 1, 2, 4, 8, 16s
            print(
                f"  [retry] transient upstream error, waiting {wait:.0f}s "
                f"(attempt {attempt + 1}/{retries})"
            )
            time.sleep(wait)


# ============================================================
# STEP 3 — STRUCTURED OUTPUT for the quality score
# ============================================================
class QualityScore(BaseModel):
    """Evaluation of research quality."""
    score: int = Field(ge=1, le=10)
    reasoning: str = Field(description="One-sentence justification")


evaluator = llm.with_structured_output(QualityScore)


# ============================================================
# STEP 4 — NODES
# ============================================================
def collect_node(state: AgentState):
    """Search the web. On retries, CHANGE the query!"""
    iteration = state["iteration_count"] + 1

    refinements = [
        state["topic"],
        f"{state['topic']} latest developments case studies",
        f"{state['topic']} industry analysis best practices",
    ]
    query = refinements[min(iteration - 1, len(refinements) - 1)]

    results = search_tool.invoke({"query": query})["results"]

    return {
        "search_query": query,
        "collected_data": results,
        "iteration_count": iteration,
        "execution_logs": [
            f"Iteration {iteration}: collected {len(results)} sources for '{query}'"
        ],
    }


def store_memory_node(state: AgentState):
    """Save source contents into the vector store."""
    documents = [
        item.get("content", "")
        for item in state["collected_data"]
        if item.get("content")
    ]
    if documents:
        vector_store.add_texts(documents)
    return {"execution_logs": [f"Stored {len(documents)} documents in vector memory."]}


def analyze_node(state: AgentState):
    """LLM-analyze each source, with RAG retrieval from memory."""
    analyzed = []
    for item in state["collected_data"]:
        content = item.get("content", "")
        related = vector_store.similarity_search(content, k=2)
        related_context = "\n".join(d.page_content for d in related)

        response = invoke_with_retry(
            llm,
            [HumanMessage(content=(
                "Analyze the following research content.\n\n"
                f"Content:\n{content}\n\n"
                f"Related prior research from memory:\n{related_context}\n\n"
                "Generate:\n1. Summary\n2. Importance Score (1-10)\n3. Business Impact"
            ))],
        )

        analyzed.append(
            {"source": item.get("url", "Unknown"), "analysis": response.content}
        )
        time.sleep(1)  # throttle: space out calls to the shared endpoint

    return {
        "analyzed_data": analyzed,
        "execution_logs": [f"Analyzed {len(analyzed)} sources."],
    }


def evaluate_node(state: AgentState):
    """Score the research with the STRUCTURED evaluator (Step 3)."""
    result = invoke_with_retry(
        evaluator,
        [HumanMessage(content=(
            "Evaluate the overall quality of this research on a 1-10 scale.\n\n"
            f"Research:\n{state['analyzed_data']}"
        ))],
    )
    return {
        "quality_score": result.score,
        "execution_logs": [f"Quality score = {result.score} ({result.reasoning})"],
    }


def report_node(state: AgentState):
    """Generate the enterprise report from analyzed_data."""
    response = invoke_with_retry(
        llm,
        [HumanMessage(content=(
            "Generate a professional enterprise research report.\n\n"
            f"Topic:\n{state['topic']}\n\n"
            f"Research Analysis:\n{state['analyzed_data']}\n\n"
            "Include:\n- Executive Summary\n- Key Findings\n- Risks\n"
            "- Opportunities\n- Strategic Recommendations"
        ))],
    )
    return {
        "final_report": response.content,
        "execution_logs": ["Final report generated."],
    }


def audit_node(state: AgentState):
    """Log completion stats."""
    return {
        "execution_logs": [
            f"Audit complete. Iterations: {state['iteration_count']}, "
            f"final quality: {state['quality_score']}."
        ]
    }


# ============================================================
# STEP 5 — THE CONDITIONAL EDGE
# ============================================================
MAX_RESEARCH_ITERATIONS = 3
QUALITY_THRESHOLD = 7


def quality_router(state: AgentState) -> str:
    score = state["quality_score"]
    iteration = state["iteration_count"]

    if score >= QUALITY_THRESHOLD:
        return "report"
    if iteration >= MAX_RESEARCH_ITERATIONS:
        return "report"
    return "collect"


# ============================================================
# STEP 6 — WIRE THE GRAPH
# ============================================================
workflow = StateGraph(AgentState)

workflow.add_node("collect", collect_node)
workflow.add_node("store_memory", store_memory_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("evaluate", evaluate_node)
workflow.add_node("report", report_node)
workflow.add_node("audit", audit_node)

workflow.add_edge(START, "collect")

workflow.add_edge("collect", "store_memory")
workflow.add_edge("store_memory", "analyze")
workflow.add_edge("analyze", "evaluate")

workflow.add_conditional_edges(
    "evaluate",
    quality_router,
    {"collect": "collect", "report": "report"},
)

workflow.add_edge("report", "audit")
workflow.add_edge("audit", END)


# ============================================================
# STEP 7 — COMPILE, VISUALIZE, RUN
# ============================================================
if __name__ == "__main__":
    initial_state = {
        "topic": "AI in Saudi healthcare data platforms",
        "search_query": "",
        "collected_data": [],
        "analyzed_data": [],
        "quality_score": 0,
        "iteration_count": 0,
        "final_report": "",
        "execution_logs": [],
    }

    app = workflow.compile(checkpointer=InMemorySaver())

    print("\n--- GRAPH STRUCTURE (Mermaid) ---")
    print(app.get_graph().draw_mermaid())

    config = {"configurable": {"thread_id": "run-1"}}

    final_state = None
    for chunk in app.stream(initial_state, config, stream_mode="values"):
        final_state = chunk

    print("\n================================================")
    print("FINAL ENTERPRISE RESEARCH REPORT")
    print("================================================")
    print(final_state["final_report"])

    print("\n================================================")
    print("EXECUTION LOGS")
    print("================================================")
    for line in final_state["execution_logs"]:
        print(line)
