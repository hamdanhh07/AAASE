# ============================================================
# DAY 3 LAB — SKELETON: From Prototype to Enterprise
# (covers Day 3: multi-agent systems + Day 5: production agents)
# ============================================================
# Fill in every TODO. Each step tells you exactly WHERE in the
# docs to look. Don't copy from the solution file
# (../lab_prototype_to_enterprise.py) until you've tried each
# step — the point of this lab is learning what separates a
# DEMO from a PRODUCT.
#
# The system you're building — a multi-agent report generator
# (Day 3) that you then harden into an enterprise service (Day 5):
#
#   START → research → summarize → write → review
#                        ↑                   │
#                        └─ score < 8 ───────┤  (max 2 revisions)
#                                            └─ score >= 8 → END
#
#   ...then wrapped, layer by layer, in:
#
#   ┌─ Stage 5: FastAPI service (/health, /report) ──────────┐
#   │ ┌─ Stage 4: guardrails + cost budget ────────────────┐ │
#   │ │ ┌─ Stage 3: structured logs, run_id, latency ────┐ │ │
#   │ │ │ ┌─ Stage 2: config from env, secrets in .env ─┐│ │ │
#   │ │ │ │ ┌─ Stage 1: retries, backoff, timeouts ────┐││ │ │
#   │ │ │ │ │        Stage 0: the agent graph          │││ │ │
#   │ │ │ │ └──────────────────────────────────────────┘││ │ │
#   │ │ │ └─────────────────────────────────────────────┘│ │ │
#   │ │ └────────────────────────────────────────────────┘ │ │
#   │ └──────────────────────────────────────────────────────┘ │
#   └──────────────────────────────────────────────────────────┘
#
# Recommended reading BEFORE you start (~30 min):
#   1. Multi-agent concepts (supervisor pattern — today's graph):
#      https://docs.langchain.com/oss/python/langgraph/multi-agent
#   2. Graph API (you know this from Day 2 — skim as refresher):
#      https://docs.langchain.com/oss/python/langgraph/use-graph-api
#   3. Anthropic, "Building effective agents" (when NOT to
#      multi-agent): https://www.anthropic.com/research/building-effective-agents
#
# Model setup: same as Day 2 — OpenAI key, or OpenRouter free
# models (see the big OpenRouter block in
# ../../Day_2/Day Two Lab/Updated_2026/skeleton_research_agent.py,
# Step 2). No key at all? Set MOCK=1 and a fake model is used.
#
# Setup:
#   pip install langchain-openai langgraph python-dotenv fastapi uvicorn
# ============================================================

import json
import logging
import operator
import os
import random
import re
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, List
from typing_extensions import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
#from langchain_tavily import TavilySearch
#from langchain_core.vectorstores import InMemoryVectorStore
#from langchain_huggingface import HuggingFaceEmbeddings

# TODO STEP 0 — import StateGraph, START, END from langgraph.graph
# (same imports as Day 2).

load_dotenv()

STAGE = int(os.getenv("LAB_STAGE", "0"))   # 0..5 — your maturity level
MOCK = os.getenv("MOCK", "0") == "1"


# ============================================================
# STEP 1 — THE STATE (the contract between your agents)
# ============================================================
# Day 3 slides: agents coordinate through a COMMUNICATION
# MECHANISM — here it's shared graph state.
#
# Define a TypedDict with:
#   run_id (str), topic (str), research_notes (str), summary (str),
#   draft (str), review_feedback (str), score (int),
#   revision_count (int), tokens_in (int), tokens_out (int),
#   cost_usd (float), error (str),
#   execution_logs — with the operator.add REDUCER (Day 2!)
#
# ASK YOURSELF: why must revision_count live in STATE and not in
# a Python variable next to the graph? (Hint: checkpointing,
# multiple runs, serving this graph from an API later.)

class ReportState(TypedDict, total=False):
    run_id: str
    topic: str
    research_notes: str
    summary: str
    draft: str
    review_feedback: str
    score: int
    revision_count: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    error: str
    execution_logs: Annotated[List[str], operator.add]
    pass


# ============================================================
# STEP 2 — MODEL (with an offline mock)
# ============================================================
# Create the model exactly as in Day 2 (ChatOpenAI, or OpenRouter
# with base_url + :free model). ONE addition for Stage 1+:
#   pass  timeout=60, max_retries=0  to ChatOpenAI.
# max_retries=0?! Yes — YOU will own retries in Step 5, and two
# competing retry layers multiply (3 SDK x 3 yours = 9 calls).
#
# The FakeChatModel below lets everyone run the lab offline.
# Read it — note how it fails the first review on purpose so the
# revision loop always fires.

class FakeResponse:
    def __init__(self, content):
        self.content = content
        self.usage_metadata = {"input_tokens": 200, "output_tokens": 300}


class FakeChatModel:
    def __init__(self):
        self.review_calls = 0

    def invoke(self, prompt: str):
        time.sleep(0.2)
        p = prompt.lower()
        if "you are a strict reviewer" in p:
            self.review_calls += 1
            score = 6 if self.review_calls == 1 else 9
            return FakeResponse(f"SCORE: {score}\nFEEDBACK: Add a concrete example.")
        if "you are a researcher" in p:
            return FakeResponse("- fact one\n- fact two\n- fact three")
        if "you are a summarizer" in p:
            return FakeResponse("A concise summary of the research notes.")
        return FakeResponse("INTRODUCTION\n...\n\nBODY\n" + "Substantive findings. " * 20
                            + "\n\nCONCLUSION\n...")


# TODO: model = FakeChatModel() if MOCK else ChatOpenAI(...)
if MOCK:
    model = FakeChatModel()
else:
    model = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "nvidia/nemotron-3-ultra-550b-a55b:free"),
        temperature=float(os.getenv("TEMPERATURE", "0")),
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),   # <-- add this line
        timeout=int(os.getenv("REQUEST_TIMEOUT_S", "60")),
        max_retries=0,
    )

# ============================================================
# STEP 3 — ROLE-SPECIALIZED AGENTS (Day 3, slides 28-39)
# ============================================================
# Four nodes, four ROLES: Researcher, Summarizer, Writer, Reviewer.
# Each is a plain function: state in → partial dict out (Day 2 rule).
#
# For now call the model DIRECTLY (model.invoke(prompt).content).
# In Step 5 you will refactor every call to go through ONE
# chokepoint — notice how painful it would be if you had 20 nodes.
#
# WRITER: if state has review_feedback, append it to the prompt
# ("A reviewer said: ... address this feedback"). That single line
# is what turns the loop from "retry" into genuine COLLABORATION.
#
# REVIEWER: force the format "SCORE: <n>\nFEEDBACK: <line>" and
# parse with re.search(r"SCORE:\s*(\d+)", ...). (Day 2 taught the
# better way — with_structured_output. BONUS: use it here too.)

def _usage(state):
    return {"tokens_in": state["tokens_in"],
            "tokens_out": state["tokens_out"],
            "cost_usd": state["cost_usd"]}


def research_node(state: ReportState):
    prompt = ("You are a Researcher. Gather key facts on the topic.\n"
              f"Topic: {state['topic']}\n"
              "Return a concise bulleted list of factual research notes.")
    notes = call_llm(prompt, "research", state)
    return {**_usage(state), "research_notes": notes,
            "execution_logs": ["research: notes gathered"]}


def summarize_node(state: ReportState):
    prompt = ("You are a Summarizer. Condense these research notes.\n"
              f"Notes:\n{state.get('research_notes','')}\n"
              "Return a tight paragraph.")
    summary = call_llm(prompt, "summarize", state)
    return {**_usage(state), "summary": summary,
            "execution_logs": ["summarize: summary written"]}


def write_node(state: ReportState):
    prompt = ("You are a Writer. Write a report with INTRODUCTION, BODY, "
              "CONCLUSION.\n"
              f"Topic: {state['topic']}\n"
              f"Summary: {state.get('summary','')}\n"
              f"Research notes: {state.get('research_notes','')}\n")
    if state.get("review_feedback"):          # <-- this line makes it COLLABORATION
        prompt += (f"\nA reviewer said: {state['review_feedback']}\n"
                   "Revise the report to address this feedback.\n")
    draft = call_llm(prompt, "write", state)
    return {**_usage(state), "draft": draft,
            "execution_logs": ["write: draft produced"]}


def review_node(state: ReportState):
    prompt = ("You are a strict Reviewer.\n"
              f"Topic: {state['topic']}\n"
              f"Draft:\n{state.get('draft','')}\n\n"
              "Respond EXACTLY as:\n"
              "SCORE: <integer 1-10>\nFEEDBACK: <one actionable line>")
    raw = call_llm(prompt, "review", state)
    m = re.search(r"SCORE:\s*(\d+)", raw)
    score = int(m.group(1)) if m else 0
    fb = re.search(r"FEEDBACK:\s*(.+)", raw)
    feedback = fb.group(1).strip() if fb else ""
    result = {**_usage(state), "score": score, "review_feedback": feedback,
              "revision_count": state.get("revision_count", 0) + 1,
              "execution_logs": [f"review: score={score}"]}
    if STAGE >= 3:
        log_event("review_verdict", run_id=state.get("run_id"),
                  score=score, revision=result["revision_count"])
    return result


# ============================================================
# STEP 4 — THE SUPERVISOR DECISION (Day 3: coordination strategy)
# ============================================================
# Router after review:
#   "approve"  score >= QUALITY_THRESHOLD (8)
#   "give_up"  revision_count > MAX_REVISIONS (2)   <- Day 2 lesson:
#                                                      loops MUST terminate
#   "revise"   otherwise → back to write
#
# Then wire the graph:
#   START → research → summarize → write → review
#   add_conditional_edges("review", review_gate,
#       {"approve": END, "give_up": END, "revise": "write"})
#
# WHERE TO LOOK: same conditional-branching docs as Day 2.
# ASK YOURSELF: why does "revise" go to write, not research?
# When WOULD you route back to research instead?

def review_gate(state: ReportState) -> str:
    if state.get("score", 0) >= settings.quality_threshold:
        return "approve"
    if state.get("revision_count", 0) > settings.max_revisions:
        return "give_up"
    return "revise"


workflow = StateGraph(ReportState)
for name, fn in [("research", research_node), ("summarize", summarize_node),
                 ("write", write_node), ("review", review_node)]:
    workflow.add_node(name, fn)
workflow.add_edge(START, "research")
workflow.add_edge("research", "summarize")
workflow.add_edge("summarize", "write")
workflow.add_edge("write", "review")
workflow.add_conditional_edges("review", review_gate,
                               {"approve": END, "give_up": END, "revise": "write"})
graph = workflow.compile()


# TODO: build + compile the graph  →  graph = workflow.compile()


# ============================================================
# ============================================================
#   YOU ARE NOW HERE: a working PROTOTYPE (Stage 0).
#   Everything below is Day 5 — crossing the PoC chasm.
#   Each stage guards its code with  `if STAGE >= n:`  so one
#   file can demonstrate every maturity level.
# ============================================================
# ============================================================


# ============================================================
# STEP 5 — STAGE 1: ROBUSTNESS (Day 5: "Error Handling")
# ============================================================
# Refactor: every node now calls  call_llm(prompt, node, state)
# instead of model.invoke. Implement it with:
#   - up to MAX_RETRIES attempts
#   - exponential backoff WITH JITTER between attempts:
#       delay = 2 ** (attempt - 1) + random.uniform(0, 0.5)
#   - on final failure: raise RuntimeError with node name + error
#   - in generate_report (Step 9): catch it and return a partial
#     result with state["error"] set — degrade, don't crash.
#
# WHERE TO LOOK: https://docs.aws.amazon.com/general/latest/gr/api-retries.html
#   (the canonical backoff+jitter explanation — 5 min read)
# TEST IT: temporarily add
#   if random.random() < 0.3: raise TimeoutError("boom")
# before the invoke and watch retries fire.
# ASK YOURSELF: why jitter? What happens when 100 replicas all
# retry at exactly t=1s, 2s, 4s?

COST_PER_INPUT_TOKEN = 0.00000015
COST_PER_OUTPUT_TOKEN = 0.00000060

def call_llm(prompt: str, node: str, state: ReportState) -> str:
    if STAGE >= 4 and state.get("cost_usd", 0.0) >= settings.cost_budget_usd:
        raise BudgetExceeded(f"budget {settings.cost_budget_usd} hit before '{node}'")

    max_retries = settings.max_retries if STAGE >= 1 else 1   # Stage 0 = no retry
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            resp = model.invoke(prompt)
            latency = time.time() - t0
            usage = getattr(resp, "usage_metadata", None) or {}
            ti, to = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
            delta = ti * COST_PER_INPUT_TOKEN + to * COST_PER_OUTPUT_TOKEN
            state["tokens_in"]  = state.get("tokens_in", 0)  + ti
            state["tokens_out"] = state.get("tokens_out", 0) + to
            state["cost_usd"]   = state.get("cost_usd", 0.0) + delta
            if STAGE >= 3:
                log_event("llm_call", run_id=state.get("run_id"), node=node,
                          attempt=attempt, latency_s=round(latency, 3),
                          tokens_in=ti, tokens_out=to, cost_usd=round(delta, 6))
            return resp.content
        except BudgetExceeded:
            raise
        except Exception as e:
            last_err = e
            if STAGE >= 3:
                log_event("llm_retry", level="WARNING", run_id=state.get("run_id"),
                          node=node, attempt=attempt, error=str(e))
            if attempt == max_retries:
                break
            time.sleep(2 ** (attempt - 1) + random.uniform(0, 0.5))
    raise RuntimeError(f"LLM failed in '{node}' after {max_retries} attempts: {last_err}")

# ============================================================
# STEP 6 — STAGE 2: CONFIG & SECRETS (Day 5: "Security & Governance")
# ============================================================
# Kill every hardcoded number. Build a Settings dataclass:
#   model_name, temperature, request_timeout_s, max_retries,
#   quality_threshold, max_revisions, cost_budget_usd, max_topic_len
# with a  from_env()  classmethod reading os.getenv with defaults.
#
#   settings = Settings.from_env() if STAGE >= 2 else Settings()
#
# WHERE TO LOOK: https://12factor.net/config  (10 min, classic)
# PROVE IT WORKS:  QUALITY_THRESHOLD=10 LAB_STAGE=2 python ...
# → the reviewer can never approve → give_up path fires. No code
# edits. That's the point.

@dataclass
class Settings:
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    request_timeout_s: int = 60
    max_retries: int = 3
    quality_threshold: int = 8
    max_revisions: int = 2
    cost_budget_usd: float = 0.50
    max_topic_len: int = 200

    @classmethod
    def from_env(cls):
        g = os.getenv
        return cls(
            model_name=g("MODEL_NAME", "gpt-4o-mini"),
            temperature=float(g("TEMPERATURE", "0")),
            request_timeout_s=int(g("REQUEST_TIMEOUT_S", "60")),
            max_retries=int(g("MAX_RETRIES", "3")),
            quality_threshold=int(g("QUALITY_THRESHOLD", "8")),
            max_revisions=int(g("MAX_REVISIONS", "2")),
            cost_budget_usd=float(g("COST_BUDGET_USD", "0.50")),
            max_topic_len=int(g("MAX_TOPIC_LEN", "200")),
        )

settings = Settings.from_env() if STAGE >= 2 else Settings()


# ============================================================
# STEP 7 — STAGE 3: OBSERVABILITY (Day 5: "Observability & Maintenance")
# ============================================================
# print() doesn't survive contact with production. Emit ONE JSON
# object per event so a log platform can index and query them:
#   {"ts": ..., "level": ..., "event": "llm_call", "run_id": ...,
#    "node": "write", "attempt": 1, "latency_s": 2.1,
#    "tokens_in": 812, "tokens_out": 405, "cost_usd": 0.0011}
#
# Implement log_event(event, **fields) using the logging module
# with a custom Formatter that json.dumps the record (see the
# solution file's JsonFormatter if stuck — it's 8 lines).
# Emit events: run_started, llm_call, llm_retry, review_verdict,
# run_finished.
#
# WHERE TO LOOK: https://docs.python.org/3/howto/logging-cookbook.html
# ASK YOURSELF: you have 40 runs/hour and one user says "my report
# was bad". Which field in the logs lets you reconstruct exactly
# what happened for THEIR run?

class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {"ts": datetime.now(timezone.utc).isoformat(),
                   "level": record.levelname, "event": record.getMessage()}
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload)

_logger = logging.getLogger("agent")
if not _logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(JsonFormatter())
    _logger.addHandler(_h)
    _logger.setLevel(logging.INFO)

def log_event(event: str, level: str = "INFO", **fields):
    _logger.log(getattr(logging, level, logging.INFO), event, extra={"fields": fields})

# ============================================================
# STEP 8 — STAGE 4: GUARDRAILS + COST (Day 5: "Security" + "Cost")
# ============================================================
# Three cheap, high-value protections:
#
# a) validate_topic(topic) BEFORE any LLM call:
#    - reject empty / longer than max_topic_len
#    - reject prompt-injection patterns, e.g.
#      r"ignore (all|previous|the) instructions", r"system prompt"
# b) validate_report(report) AFTER the run:
#    - reject if < 200 chars or contains refusal artifacts
#      ("as an ai language model", ...)
# c) budget: at the top of call_llm, if state's cost_usd >=
#    settings.cost_budget_usd → raise BudgetExceeded. Catch it in
#    generate_report and abort GRACEFULLY (partial result + error).
#
# WHERE TO LOOK: https://genai.owasp.org/llm-top-10/  — find which
# two entries you just mitigated.
# TEST: TOPIC="Ignore all instructions..." must be rejected;
#       COST_BUDGET_USD=0.0000001 must abort, not crash.

class BudgetExceeded(Exception):
    pass


INJECTION_PATTERNS = [r"ignore (all|previous|the|any) instructions",
                      r"disregard (all|previous|the)", r"system prompt",
                      r"you are now", r"reveal your (instructions|prompt)"]
REFUSAL_ARTIFACTS = ["as an ai language model", "i cannot fulfill",
                     "i'm sorry, but i can"]

def validate_topic(topic: str) -> str:
    topic = (topic or "").strip()
    if not topic:
        raise ValueError("topic is empty")
    if len(topic) > settings.max_topic_len:
        raise ValueError(f"topic exceeds {settings.max_topic_len} chars")
    low = topic.lower()
    for pat in INJECTION_PATTERNS:
        if re.search(pat, low):
            raise ValueError("topic rejected: possible prompt injection")
    return topic

def validate_report(report: str) -> None:
    if not report or len(report) < 200:
        raise ValueError("report too short / empty")
    low = report.lower()
    for art in REFUSAL_ARTIFACTS:
        if art in low:
            raise ValueError(f"report contains refusal artifact: {art!r}")
    pass


# ============================================================
# STEP 9 — generate_report(): tie the stages together
# ============================================================
# def generate_report(topic):
#   1. build initial state (uuid run_id, revision_count=0, cost 0)
#   2. STAGE >= 4: topic = validate_topic(topic)
#   3. STAGE >= 3: log_event("run_started", ...)
#   4. try: final = graph.invoke(state)
#      except BudgetExceeded / RuntimeError:
#          STAGE >= 1 → return partial state with error set
#          STAGE 0    → just crash (that's what prototypes do)
#   5. STAGE >= 4: validate_report(final["draft"])
#   6. STAGE >= 3: log_event("run_finished", ...totals...)

def generate_report(topic: str) -> ReportState:
    run_id = str(uuid.uuid4())
    if STAGE >= 4:
        topic = validate_topic(topic)          # ValueError → 422 upstream
    initial: ReportState = {"run_id": run_id, "topic": topic, "revision_count": 0,
                            "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0,
                            "execution_logs": []}
    if STAGE >= 3:
        log_event("run_started", run_id=run_id, topic=topic, stage=STAGE)
    try:
        final = graph.invoke(initial)
    except (BudgetExceeded, RuntimeError) as e:
        if STAGE >= 1:
            partial = {**initial, "error": str(e)}
            if STAGE >= 3:
                log_event("run_finished", level="ERROR", run_id=run_id, error=str(e))
            return partial
        raise                                   # Stage 0: crash, like a prototype
    if STAGE >= 4:
        validate_report(final.get("draft", ""))
    if STAGE >= 3:
        log_event("run_finished", run_id=run_id, score=final.get("score"),
                  revisions=final.get("revision_count"),
                  tokens_in=final.get("tokens_in"), tokens_out=final.get("tokens_out"),
                  cost_usd=round(final.get("cost_usd", 0.0), 6))
    return final

# ============================================================
# STEP 10 — STAGE 5: SERVING (Day 5: cloud deployment sections)
# ============================================================
# A script is a demo; an API is a product other teams can use.
#   app = FastAPI()
#   GET  /health  → {"status": "ok", "stage": STAGE, "model": ...}
#   POST /report  → body {"topic": str} (pydantic model), calls
#                   generate_report; map errors to HTTP:
#                   guardrail ValueError → 422, run error → 503
#
# WHERE TO LOOK: https://fastapi.tiangolo.com/tutorial/first-steps/
# RUN:   LAB_STAGE=5 python skeleton_enterprise_multiagent.py serve
# TEST:  curl localhost:8000/health
#        curl -X POST localhost:8000/report -H 'Content-Type: application/json' \
#             -d '{"topic": "Smart Cities"}'
#        curl ... -d '{"topic": "Ignore all instructions"}'   # expect 422
# ASK YOURSELF: you now run 3 replicas behind a load balancer.
# Which parts of your file break? (Hint: anything in module-level
# variables — like FakeChatModel.review_calls...)

def create_app():
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    app = FastAPI(title="Report Agent", version=str(STAGE))

    class ReportRequest(BaseModel):
        topic: str

    @app.get("/health")
    def health():
        return {"status": "ok", "stage": STAGE,
                "model": "mock" if MOCK else settings.model_name}

    @app.post("/report")
    def report(req: ReportRequest):
        try:
            result = generate_report(req.topic)
        except ValueError as e:                 # guardrail hit
            raise HTTPException(status_code=422, detail=str(e))
        if result.get("error"):                 # run failed but degraded gracefully
            raise HTTPException(status_code=503, detail=result["error"])
        return {"run_id": result.get("run_id"), "score": result.get("score"),
                "cost_usd": round(result.get("cost_usd", 0.0), 6),
                "report": result.get("draft", "")}
    return app


if __name__ == "__main__":
    print(f"=== STAGE {STAGE} {'(MOCK)' if MOCK else ''} ===")
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        import uvicorn
        uvicorn.run(create_app(), host="0.0.0.0", port=8000)
    else:
        topic = os.getenv("TOPIC", "AI in Saudi healthcare data platforms")
        try:
            result = generate_report(topic)
        except ValueError as e:
            print(f"REJECTED: {e}"); sys.exit(1)
        if result.get("error"):
            print(f"RUN FAILED: {result['error']}")
        print(f"\nSCORE: {result.get('score')}  "
              f"REVISIONS: {result.get('revision_count')}  "
              f"COST: ${result.get('cost_usd', 0.0):.6f}\n")
        print(result.get("draft", "(no draft)"))


# ============================================================
# SELF-CHECK before you look at the solution
# ============================================================
# Day 3 (the agent):
# [ ] Four role agents communicate ONLY through graph state
# [ ] The writer actually USES the reviewer's feedback on revision
# [ ] My loop has both a quality exit AND a revision cap (Day 2!)
# [ ] I can explain when I'd route "revise" → research instead of write
# Day 5 (the chasm):
# [ ] ALL model calls go through call_llm — zero direct invokes left
# [ ] I know why SDK max_retries=0 when I own retries (and why jitter)
# [ ] QUALITY_THRESHOLD=10 changes behavior with zero code edits
# [ ] My logs are one JSON object per line, every one has run_id
# [ ] Injection topic → rejected BEFORE any money is spent
# [ ] Budget exhaustion aborts gracefully mid-run (partial + error)
# [ ] /report returns 422 for guardrail hits, 503 for run failures
# [ ] I can name 3 things STILL missing for real production
#     (auth? rate limiting? queue for long runs? containers? CI/CD?)
#
# Stuck? Debugging order that works:
#   1. MOCK=1 LAB_STAGE=0 — get the bare graph green first
#   2. raise a fake TimeoutError inside call_llm — watch retries
#   3. pipe Stage 3 output through `python -m json.tool` per line
#   4. only THEN open ../lab_prototype_to_enterprise.py
# ============================================================


def save_report(state):
    out_dir = os.getenv("REPORTS_DIR", ".")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"report_{state.get('run_id', 'unknown')}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(state.get("draft", "(no draft)"))
    print(f"saved: {path}")
    return path


if __name__ == "__main__" and os.getenv("REPORTS_DIR"):
    _topic = os.getenv("TOPIC", "AI in Saudi healthcare data platforms")
    save_report(generate_report(_topic))


def save_report(state):
    out_dir = os.getenv("REPORTS_DIR", ".")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"report_{state.get('run_id', 'unknown')}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(state.get("draft", "(no draft)"))
    print(f"saved: {path}")
    return path
