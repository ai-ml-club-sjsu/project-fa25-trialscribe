"""
TrialScribe — LangGraph + Claude + Vector Store + PDF Ingestion
---------------------------------------------------------------
This upgrade adds:
  1) Explicit state machine using LangGraph
  2) Vector store retriever (Chroma + HuggingFace embeddings)
  3) Real PDF ingestion from a directory, persisted to disk

Quick start
-----------
# 1) Install deps
pip install -U \
  langchain langchain-core langchain-community langchain-anthropic \
  langgraph chromadb sentence-transformers pypdf

# 2) Set keys (Claude via Anthropic)
export ANTHROPIC_API_KEY=your_key_here

# 3) Ingest PDFs (put your guidance PDFs under ./pdfs)
python trialscribe_demo.py --ingest --pdf-dir ./pdfs --persist ./chroma

# 4) Run the LangGraph demo
python trialscribe_demo.py --run --persist ./chroma

Notes
-----
- Uses sentence-transformers/all-MiniLM-L6-v2 locally for embeddings (no extra API).
- Chroma persists to a folder so you ingest once and reuse.
- Swap to FAISS easily if you prefer an in-memory index.
"""

from __future__ import annotations
import os
import sys
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from typing_extensions import TypedDict

# LangChain / Anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool

# Vector store + embeddings + loaders
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# LangGraph
from langgraph.graph import StateGraph, END


# ---------------------------------------------------------------------
# Compliance tool — trivial heuristics (extend with real rules later)
# ---------------------------------------------------------------------
@dataclass
class ComplianceIssue:
    rule: str
    message: str


@tool("compliance_check", return_direct=False)
def compliance_check(text: str) -> Dict[str, List[Dict[str, str]]]:
    """Check text for simple compliance heuristics (demo only)."""
    issues: List[ComplianceIssue] = []

    if "TBD" in text or "to be determined" in text.lower():
        issues.append(ComplianceIssue("no_placeholders", "Remove TBD/placeholder language."))

    if "risk" in text.lower() and "mitigation" not in text.lower():
        issues.append(ComplianceIssue("risk_mitigation", "Mention risk mitigation when risks are discussed."))

    if "consent" in text.lower() and "withdraw" not in text.lower():
        issues.append(ComplianceIssue("consent_withdrawal", "State withdrawal rights in consent context."))

    if len(text.split()) < 150:
        issues.append(ComplianceIssue("min_length", "Provide at least ~150 words for sufficient detail."))

    return {
        "ok": len(issues) == 0,
        "issues": [{"rule": i.rule, "message": i.message} for i in issues],
    }


# ---------------------------------------------------------------------
# LLM setup (Claude 3.5 Sonnet via LangChain)
# ---------------------------------------------------------------------

def get_llm() -> ChatAnthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Please set ANTHROPIC_API_KEY environment variable.")
    return ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.2)


# ---------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------
DRAFT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a clinical-trial documentation assistant. Write clear, precise, and compliant text. "
        "Follow retrieved guidance carefully and avoid ambiguous statements."
    )),
    ("human", (
        "TASK: {task}

"
        "CONTEXT (guidance snippets):
{context}

"
        "Write the requested section. Use neutral tone and professional clinical-trial style."
    )),
])

REVISION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a meticulous compliance editor for clinical-trial documents."),
    ("human", (
        "Revise the DRAFT to resolve the following compliance issues. Preserve meaning and structure.

"
        "DRAFT:
{draft}

"
        "COMPLIANCE ISSUES:
{issues}

"
        "Return the full revised text, improved but not overly verbose."
    )),
])


# ---------------------------------------------------------------------
# Vector store & ingestion
# ---------------------------------------------------------------------

def load_pdfs(pdf_dir: str) -> List[Document]:
    docs: List[Document] = []
    if not os.path.isdir(pdf_dir):
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
    for root, _, files in os.walk(pdf_dir):
        for fn in files:
            if fn.lower().endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(root, fn))
                docs.extend(loader.load())
    if not docs:
        raise RuntimeError("No PDFs found to ingest.")
    return docs


def build_vectorstore(pdf_dir: str, persist_dir: str) -> Chroma:
    docs = load_pdfs(pdf_dir)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_dir)
    vs.persist()
    return vs


def load_vectorstore(persist_dir: str) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)


# ---------------------------------------------------------------------
# LangGraph state + nodes
# ---------------------------------------------------------------------
class DocState(TypedDict, total=False):
    task: str
    context: str
    draft: str
    issues: List[Dict[str, str]]
    i: int            # iteration counter
    max_iters: int
    final: str


def fmt_context(docs: List[Document]) -> str:
    lines = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "doc")
        lines.append(f"- [{i}] ({os.path.basename(src)}) {d.page_content[:350].replace('
',' ')}...")
    return "
".join(lines)


# Node: retrieve

def make_retrieve_node(retriever):
    def retrieve(state: DocState) -> DocState:
        task = state["task"]
        docs = retriever.get_relevant_documents(task)
        return {"context": fmt_context(docs)}
    return retrieve


# Node: draft

def draft_node(state: DocState) -> DocState:
    llm = get_llm()
    chain = DRAFT_PROMPT | llm | StrOutputParser()
    draft = chain.invoke({"task": state["task"], "context": state.get("context", "")})
    return {"draft": draft}


# Node: compliance check

def check_node(state: DocState) -> DocState:
    result = compliance_check.invoke({"text": state["draft"]})
    return {"issues": result.get("issues", [])}


# Node: revise

def revise_node(state: DocState) -> DocState:
    llm = get_llm()
    issues_str = "
".join(f"- {i['rule']}: {i['message']}" for i in state.get("issues", []))
    chain = REVISION_PROMPT | llm | StrOutputParser()
    revised = chain.invoke({"draft": state["draft"], "issues": issues_str})
    return {"draft": revised, "i": state.get("i", 0) + 1}


# Router after check

def decide_next(state: DocState) -> str:
    issues = state.get("issues", [])
    if not issues:
        return "end"
    if state.get("i", 0) >= state.get("max_iters", 2):
        return "end"
    return "revise"


# ---------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------

def build_graph(retriever):
    graph = StateGraph(DocState)

    graph.add_node("retrieve", make_retrieve_node(retriever))
    graph.add_node("draft", draft_node)
    graph.add_node("check", check_node)
    graph.add_node("revise", revise_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "draft")
    graph.add_edge("draft", "check")
    graph.add_conditional_edges("check", decide_next, {"revise": "revise", "end": END})
    graph.add_edge("revise", "check")

    return graph.compile()


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Streaming & Visualization helpers
# ---------------------------------------------------------------------

from rich.console import Console
from rich.panel import Panel
from rich.padding import Padding
from rich.text import Text


def stream_run(app, state_init: Dict, events_path: Optional[str] = None):
    """Stream node-by-node updates from the LangGraph app with colored console and optional JSONL."""
    print("
[Streaming] Starting LangGraph run...
")
    console = Console()
    ev_file = open(events_path, "a", encoding="utf-8") if events_path else None
    for event in app.stream(state_init):
        # event is a dict keyed by node name -> state delta
        now = dt.datetime.utcnow().isoformat() + "Z"
        for node, delta in event.items():
            # --- Console (colored) ---
            title = f"[bold cyan]node[/]: [bold]{node}[/]"
            body_lines = []
            if "context" in delta:
                ctx_preview = (delta["context"][:120] + "…") if len(delta["context"]) > 120 else delta["context"]
                body_lines.append(f"[dim]context[/]: {ctx_preview}")
            if "draft" in delta:
                preview = delta["draft"].replace("
", " ")
                preview = (preview[:160] + "…") if len(preview) > 160 else preview
                body_lines.append(f"[magenta]draft[/]: {preview}")
            if "issues" in delta:
                rules = ", ".join([i.get("rule", "?") for i in delta["issues"]])
                color = "red" if rules else "green"
                body_lines.append(f"[{color}]issues[/]: {rules or 'none'}")
            if "i" in delta:
                body_lines.append(f"[yellow]iteration[/]: {delta['i']}")

            panel = Panel.fit(Padding("
".join(body_lines) or "(no changes)", (0, 1)), title=title, border_style="cyan")
            console.print(panel)

            # --- JSONL events ---
            if ev_file:
                try:
                    ev = {"ts": now, "node": node, "delta": delta}
                    ev_file.write(json.dumps(ev, ensure_ascii=False) + "
")
                    ev_file.flush()
                except Exception:
                    pass
    if ev_file:
        ev_file.close()
    print("[Streaming] Done.
")


def export_graph_mermaid(filepath: str = "graph.mmd"):
    """Export a Mermaid diagram of the DAG (static since edges are known)."""
    mermaid = """
flowchart TD
  R[retrieve] --> D[draft]
  D --> C[check]
  C -- no issues / max iters --> E{{END}}
  C -- issues remain --> V[revise]
  V --> C
""".strip()
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(mermaid)
    return filepath


def try_langgraph_png(app, out_png: str = "graph.png") -> bool:
    """Attempt built‑in PNG rendering if the installed LangGraph version supports it."""
    try:
        g = app.get_graph()
        # Some versions provide draw_png; others expose draw_mermaid_png; both may require graphviz
        if hasattr(g, "draw_png"):
            g.draw_png(out_png)
            return True
        if hasattr(g, "draw_mermaid_png"):
            g.draw_mermaid_png(out_png)
            return True
    except Exception as _:
        pass
    return False


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

def run_demo(persist_dir: str, task: Optional[str] = None, stream: bool = False, events_path: Optional[str] = None):
    # Load vector store & retriever
    vs = load_vectorstore(persist_dir)
    retriever = vs.as_retriever(search_kwargs={"k": 4})

    app = build_graph(retriever)

    if not task:
        task = (
            "Write a \"Protocol Synopsis\" paragraph for an interventional Phase II oncology trial. "
            "Mention design, key eligibility, primary endpoint, AE reporting basics, data protection, and informed consent."
        )

    # Initialize state and run
    state: DocState = {"task": task, "i": 0, "max_iters": 2}
    if stream:
        stream_run(app, state, events_path=events_path)
    final_state = app.invoke(state)

    final_text = final_state.get("draft", "")
    print("
=== FINAL OUTPUT ===
")
    print(final_text)


def main():
    parser = argparse.ArgumentParser(description="TrialScribe LangGraph Demo")
    parser.add_argument("--ingest", action="store_true", help="Ingest PDFs into a persistent Chroma store")
    parser.add_argument("--pdf-dir", type=str, default="./pdfs", help="Directory containing guidance PDFs")
    parser.add_argument("--persist", type=str, default="./chroma", help="Chroma persistence directory")
    parser.add_argument("--run", action="store_true", help="Run the LangGraph demo")
    parser.add_argument("--stream", action="store_true", help="Stream node-by-node updates during the run")
    parser.add_argument("--events", type=str, default=None, help="Write JSONL stream events to this file path")
    parser.add_argument("--viz", action="store_true", help="Export a visual DAG (PNG if supported; Mermaid fallback)")
    parser.add_argument("--task", type=str, default=None, help="Custom task")
    args = parser.parse_args()

    try:
        if args.ingest:
            print(f"[Ingest] Loading PDFs from {args.pdf_dir}…")
            build_vectorstore(args.pdf_dir, args.persist)
            print(f"[Ingest] Completed. Persisted vector store at {args.persist}")

        if args.viz:
            # Build a temporary app so we can try built-in PNG export
            vs = load_vectorstore(args.persist)
            app = build_graph(vs.as_retriever(search_kwargs={"k": 4}))
            ok = try_langgraph_png(app, out_png="graph.png")
            if ok:
                print("[Viz] Wrote graph.png")
            else:
                path = export_graph_mermaid("graph.mmd")
                print(f"[Viz] Mermaid diagram saved to {path}. Render with: mmdc -i graph.mmd -o graph.png")

        if args.run:
            run_demo(args.persist, args.task, stream=args.stream, events_path=args.events)

        if not args.ingest and not args.run and not args.viz:
            parser.print_help()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

