#!/usr/bin/env node
/**
 * clinicalTrialTemplateCreate.js
 * LangGraph + Claude + Local Embeddings (Xenova) + PDF ingestion (text-only)
 * Minimal in-memory vector store (no @langchain/community and no Chroma).
 */

import "dotenv/config";
import fs from "fs";
import path from "path";
import process from "process";
import chalk from "chalk";
import { fileURLToPath } from "url";

import { ChatAnthropic } from "@langchain/anthropic";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Embeddings } from "@langchain/core/embeddings";
import { StateGraph, END, Annotation } from "@langchain/langgraph";
import { pipeline } from "@xenova/transformers";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ------------------------- CLI helpers --------------------------
function parseArgs(argv) {
  const args = {};
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith("--")) {
      const key = a.slice(2);
      const val = argv[i + 1] && !argv[i + 1].startsWith("--") ? argv[++i] : true;
      args[key] = val;
    }
  }
  return args;
}
const nowISO = () => new Date().toISOString();

// ------------------------- LLM & Prompts --------------------------

function getLLM() {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) throw new Error("Please set ANTHROPIC_API_KEY in environment or .env");

  // Allow override via env; otherwise pick a widely-available default
  const model =
    process.env.ANTHROPIC_MODEL ||
    "claude-3-haiku-20240307"; // safe default; change if you have access to newer models

  return new ChatAnthropic({
    apiKey,
    model,           // e.g. "claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229", "claude-3-5-sonnet-20241022"
    temperature: 0.2,
  });
}

const DRAFT_PROMPT = ChatPromptTemplate.fromMessages([
  ["system", "You are a clinical-trial documentation assistant. Write clear, precise, compliant text. Follow retrieved guidance carefully and avoid ambiguous statements."],
  ["human", "TASK: {task}\n\nCONTEXT (guidance snippets):\n{context}\n\nWrite the requested section. Use neutral tone and professional clinical-trial style."]
]);

const REVISION_PROMPT = ChatPromptTemplate.fromMessages([
  ["system", "You are a meticulous compliance editor for clinical-trial documents."],
  ["human", "Revise the DRAFT to resolve the following compliance issues. Preserve meaning and structure.\n\nDRAFT:\n{draft}\n\nCOMPLIANCE ISSUES:\n{issues}\n\nReturn the full revised text, improved but not overly verbose."]
]);

// ------------------------- Local embeddings (Xenova) --------------
class XenovaEmbeddings extends Embeddings {
  constructor({ model = "Xenova/all-MiniLM-L6-v2" } = {}) {
    super();
    this.model = model;
    this._pipePromise = null;
  }
  async _pipe() {
    if (!this._pipePromise) this._pipePromise = pipeline("feature-extraction", this.model);
    return this._pipePromise;
  }
  async embedQuery(text) {
    const pipe = await this._pipe();
    const output = await pipe(text, { pooling: "mean", normalize: true });
    return Array.from(output.data);
  }
  async embedDocuments(texts) {
    const pipe = await this._pipe();
    const out = [];
    for (const t of texts) {
      const res = await pipe(t, { pooling: "mean", normalize: true });
      out.push(Array.from(res.data));
    }
    return out;
  }
}

// ------------------------- Minimal in-memory vector store ---------
// Stores: [{ text, metadata, vector: number[] }]
class SimpleVectorStore {
  constructor(embeddings) {
    this.embeddings = embeddings;
    this.items = [];
  }
  static _cosine(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
  }
  async addDocuments(docs) {
    const texts = docs.map(d => d.pageContent);
    const vecs = await this.embeddings.embedDocuments(texts);
    for (let i = 0; i < docs.length; i++) {
      this.items.push({ text: texts[i], metadata: docs[i].metadata || {}, vector: vecs[i] });
    }
  }
  asRetriever({ k = 4 } = {}) {
    return {
      getRelevantDocuments: async (query) => {
        const q = await this.embeddings.embedQuery(query);
        const scored = this.items.map(it => ({ it, score: SimpleVectorStore._cosine(q, it.vector) }));
        scored.sort((a, b) => b.score - a.score);
        const top = scored.slice(0, k).map(s => ({ pageContent: s.it.text, metadata: s.it.metadata, score: s.score }));
        return top;
      }
    };
  }
}

// ------------------------- PDF ingestion (lazy pdfjs) -------------
async function extractTextFromPDF(filePath) {
  // Lazy import so pdfjs never initializes during --run
  const pdfjsLib = await import("pdfjs-dist/legacy/build/pdf.mjs");
  const data = new Uint8Array(fs.readFileSync(filePath));
  const loadingTask = pdfjsLib.getDocument({ data });
  const pdfDoc = await loadingTask.promise;

  let out = "";
  for (let pageNum = 1; pageNum <= pdfDoc.numPages; pageNum++) {
    const page = await pdfDoc.getPage(pageNum);
    const textContent = await page.getTextContent(); // text only
    const strings = textContent.items.map((it) => ("str" in it ? it.str : "")).filter(Boolean);
    out += strings.join(" ") + "\n\n";
  }
  return out.trim();
}

async function loadPDFs(pdfDir) {
  const files = fs.existsSync(pdfDir) ? fs.readdirSync(pdfDir) : [];
  const pdfs = files.filter((f) => f.toLowerCase().endsWith(".pdf"));
  if (pdfs.length === 0) throw new Error("No PDFs found to ingest.");

  const docs = [];
  const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1200, chunkOverlap: 150 });

  for (const fn of pdfs) {
    const full = path.join(pdfDir, fn);
    const text = await extractTextFromPDF(full);
    const chunks = await splitter.splitText(text || "");
    for (const [i, chunk] of chunks.entries()) {
      docs.push({ pageContent: chunk, metadata: { source: full, chunk: i } });
    }
  }
  return docs;
}

// ------------------------- Vector store usage --------------------
let GLOBAL_VS = null;

async function ingestToMemory(pdfDir) {
  const embeddings = new XenovaEmbeddings();
  const docs = await loadPDFs(pdfDir);
  const vs = new SimpleVectorStore(embeddings);
  await vs.addDocuments(docs);
  GLOBAL_VS = vs;
  return vs;
}

async function ensureVectorStore(pdfDirIfNeeded = "./pdfs") {
  if (GLOBAL_VS) return GLOBAL_VS;
  // Build on demand if not ingested yet this process
  return ingestToMemory(pdfDirIfNeeded);
}

// ------------------------- Compliance check ----------------------
function complianceCheck(text) {
  const issues = [];
  if (/\bTBD\b/i.test(text) || /to be determined/i.test(text)) {
    issues.push({ rule: "no_placeholders", message: "Remove TBD/placeholder language." });
  }
  if (/risk/i.test(text) && !/mitigation/i.test(text)) {
    issues.push({ rule: "risk_mitigation", message: "Mention risk mitigation when risks are discussed." });
  }
  if (/consent/i.test(text) && !/withdraw/i.test(text)) {
    issues.push({ rule: "consent_withdrawal", message: "State withdrawal rights in consent context." });
  }
  const wc = (text.trim().match(/\S+/g) || []).length;
  if (wc < 150) issues.push({ rule: "min_length", message: "Provide at least ~150 words for sufficient detail." });
  return { ok: issues.length === 0, issues };
}

// ------------------------- Graph ----------------------
function fmtContext(docs) {
  return docs
    .map((d, idx) => `- [${idx + 1}] (${path.basename(d.metadata?.source || "doc")}) ${String(d.pageContent).replace(/\n/g, " ").slice(0, 350)}...`)
    .join("\n");
}
function makeRetrieveNode(retriever) {
  return async (state) => {
    const docs = await retriever.getRelevantDocuments(state.task);
    return { context: fmtContext(docs) };
  };
}
async function draftNode(state) {
  const llm = getLLM();
  const chain = DRAFT_PROMPT.pipe(llm).pipe(new StringOutputParser());
  const draft = await chain.invoke({ task: state.task, context: state.context || "" });
  return { draft };
}
async function checkNode(state) {
  const res = complianceCheck(state.draft || "");
  return { issues: res.issues };
}
async function reviseNode(state) {
  const llm = getLLM();
  const issuesStr = (state.issues || []).map((i) => `- ${i.rule}: ${i.message}`).join("\n");
  const chain = REVISION_PROMPT.pipe(llm).pipe(new StringOutputParser());
  const revised = await chain.invoke({ draft: state.draft || "", issues: issuesStr });
  return { draft: revised, i: (state.i || 0) + 1 };
}
function decideNext(state) {
  const issues = state.issues || [];
  if (issues.length === 0) return "end";
  if ((state.i || 0) >= (state.maxIters ?? 2)) return "end";
  return "revise";
}

// Define the LangGraph state schema (channels)
const DocState = Annotation.Root({
  task: Annotation(),        // string
  context: Annotation(),     // string
  draft: Annotation(),       // string
  issues: Annotation(),      // array
  i: Annotation({ reducer: (_, next) => next }), // keep latest iteration
  maxIters: Annotation(),    // number
});

function buildGraph(retriever) {
  const graph = new StateGraph(DocState);

  graph.addNode("retrieve", makeRetrieveNode(retriever));
  // OLD: graph.addNode("draft", draftNode);
  graph.addNode("compose", draftNode);

  graph.addNode("check", checkNode);
  graph.addNode("revise", reviseNode);

  graph.setEntryPoint("retrieve");

  // OLD: graph.addEdge("retrieve", "draft");
  graph.addEdge("retrieve", "compose");

  // OLD: graph.addEdge("draft", "check");
  graph.addEdge("compose", "check");

  graph.addConditionalEdges("check", decideNext, { revise: "revise", end: END });
  graph.addEdge("revise", "check");

  return graph.compile();
}


// ------------------------- Streaming ------------------
async function streamRun(app, initState, jsonlPath) {
  console.log("\n" + chalk.bold("[Streaming] Starting LangGraph run...") + "\n");

  // Handle both modern (async iterable) and legacy (event-based) stream APIs
  let stream;
  try {
    stream = app.stream(initState);
  } catch (err) {
    console.error(chalk.red("Streaming not supported in this LangGraph version"), err);
    return;
  }

  const ws = jsonlPath ? fs.createWriteStream(jsonlPath, { flags: "a" }) : null;

  // ✅ Modern async-iterable interface
  if (stream && typeof stream[Symbol.asyncIterator] === "function") {
    for await (const event of stream) {
      handleStreamEvent(event, ws);
    }
  } else {
    // ✅ Legacy synchronous or callback style
    console.log(chalk.yellow("[Streaming] Fallback: stream() not async iterable; running invoke() instead.\n"));
    const finalState = await app.invoke(initState);
    handleStreamEvent({ final: finalState }, ws);
  }

  if (ws) ws.end();
  console.log(chalk.bold("[Streaming] Done."));
}

function handleStreamEvent(event, ws) {
  const ts = nowISO();
  for (const [node, delta] of Object.entries(event)) {
    const parts = [];
    if (delta.context) parts.push(chalk.dim("context: ") + (delta.context.length > 120 ? delta.context.slice(0, 120) + "…" : delta.context));
    if (delta.draft) {
      const prev = delta.draft.replace(/\n/g, " ");
      parts.push(chalk.magenta("draft: ") + (prev.length > 160 ? prev.slice(0, 160) + "…" : prev));
    }
    if (delta.issues) {
      const rules = delta.issues.map((i) => i.rule).join(", ");
      parts.push((rules ? chalk.red : chalk.green)("issues: ") + (rules || "none"));
    }
    if (typeof delta.i !== "undefined") parts.push(chalk.yellow("iteration: ") + delta.i);

    console.log(chalk.cyan("┌" + "─".repeat(40)));
    console.log(" ", chalk.cyan.bold("node: "), chalk.bold(node));
    parts.forEach((ln) => console.log(" ", ln));
    console.log(chalk.cyan("└" + "─".repeat(40)) + "\n");

    if (ws) ws.write(JSON.stringify({ ts, node, delta }) + "\n");
  }
}

// ------------------------- Visualization ----------------
function exportMermaid(filepath = path.join(__dirname, "graph.mmd")) {
  const mermaid =
    "flowchart TD\n  R[retrieve] --> P[compose]\n  P --> C[check]\n  C -- no issues / max iters --> E{{END}}\n  C -- issues remain --> V[revise]\n  V --> C\n";
  fs.writeFileSync(filepath, "%% LangGraph DAG\n" + mermaid + "\n", "utf-8");
  return filepath;
}

// ------------------------- Runner & CLI ----------------
async function runDemo(task, stream = false, eventsPath = null, pdfDir = "./pdfs") {
  const vs = await ensureVectorStore(pdfDir);
  const retriever = vs.asRetriever({ k: 4 });
  const app = buildGraph(retriever);

  const defaultTask =
    'Write a "Protocol Synopsis" paragraph for an interventional Phase II oncology trial. Mention design, key eligibility, primary endpoint, AE reporting basics, data protection, and informed consent.';
  const initState = { task: task || defaultTask, i: 0, maxIters: 2 };

  if (stream) await streamRun(app, initState, eventsPath);

  const finalState = await app.invoke(initState);
  const finalText = finalState.draft || "";
  console.log("\n" + chalk.bold("=== FINAL OUTPUT ===\n"));
  console.log(finalText + "\n");
}

(async function main() {
  const args = parseArgs(process.argv);
  if (!args.ingest && !args.run && !args.viz) {
    console.log(`
Usage:
  node ${path.basename(__filename)} --ingest --pdfDir ./pdfs
  node ${path.basename(__filename)} --run [--task "..."] [--stream] [--events ./runs/stream.jsonl] [--pdfDir ./pdfs]
  node ${path.basename(__filename)} --viz
`);
    process.exit(0);
  }

  try {
    if (args.ingest) {
      const pdfDir = args.pdfDir || path.join(__dirname, "pdfs");
      console.log(chalk.bold(`[Ingest] Loading PDFs from ${pdfDir}…`));
      await ingestToMemory(pdfDir);
      console.log(chalk.green(`[Ingest] Completed (in-memory).`));
    }

    if (args.viz) {
      const p = exportMermaid();
      console.log(chalk.cyan(`[Viz] Mermaid diagram saved to ${p}`));
      console.log(chalk.dim("Render with: mmdc -i graph.mmd -o graph.png (mermaid-cli)"));
    }

    if (args.run) {
      const pdfDir = args.pdfDir || path.join(__dirname, "pdfs");
      await runDemo(args.task || null, !!args.stream, args.events || null, pdfDir);
    }
  } catch (e) {
    console.error(chalk.red("Error:"), e?.stack || e?.message || String(e));
    process.exit(1);
  }
})();

