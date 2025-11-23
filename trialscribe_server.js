#!/usr/bin/env node
// === trialscribe_server.js (complete fixed version) ===

import "dotenv/config";
import fs from "fs";
import path from "path";
import process from "process";
import { fileURLToPath } from "url";

import express from "express";
import cors from "cors";
import multer from "multer";
import { nanoid } from "nanoid";

import { ChatAnthropic } from "@langchain/anthropic";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { Embeddings } from "@langchain/core/embeddings";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { StateGraph, END, Annotation } from "@langchain/langgraph";
import { pipeline } from "@xenova/transformers";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ---------------- Embeddings ----------------
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

// ---------------- Simple in-memory Vector Store ----------------
class SimpleVectorStore {
  constructor(embeddings) {
    this.embeddings = embeddings;
    this.items = [];
  }
  static _cosine(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
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
        return scored.slice(0, k).map(s => ({ pageContent: s.it.text, metadata: s.it.metadata, score: s.score }));
      }
    };
  }
}

// ---------------- PDF ingestion ----------------
async function extractTextFromPDF(filePath) {
  const pdfjsLib = await import("pdfjs-dist/legacy/build/pdf.mjs"); // lazy load
  const data = new Uint8Array(fs.readFileSync(filePath));
  const loadingTask = pdfjsLib.getDocument({ data });
  const pdfDoc = await loadingTask.promise;
  let out = "";
  for (let pageNum = 1; pageNum <= pdfDoc.numPages; pageNum++) {
    const page = await pdfDoc.getPage(pageNum);
    const textContent = await page.getTextContent();
    const strings = textContent.items.map(it => ("str" in it ? it.str : "")).filter(Boolean);
    out += strings.join(" ") + "\n\n";
  }
  return out.trim();
}

async function loadPDFs(dir) {
  const files = fs.existsSync(dir) ? fs.readdirSync(dir) : [];
  const pdfs = files.filter(f => f.toLowerCase().endsWith(".pdf"));
  if (pdfs.length === 0) throw new Error("No PDFs found to ingest.");
  const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1200, chunkOverlap: 150 });
  const docs = [];
  for (const fn of pdfs) {
    const full = path.join(dir, fn);
    const text = await extractTextFromPDF(full);
    const chunks = await splitter.splitText(text || "");
    chunks.forEach((chunk, i) => docs.push({ pageContent: chunk, metadata: { source: full, chunk: i } }));
  }
  return docs;
}

// ---------------- LangGraph pipeline ----------------
function getLLM() {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) throw new Error("Set ANTHROPIC_API_KEY");
  const model = process.env.ANTHROPIC_MODEL || "claude-3-haiku-20240307";
  return new ChatAnthropic({ apiKey, model, temperature: 0.2 });
}

const DRAFT_PROMPT = ChatPromptTemplate.fromMessages([
  ["system", "You are a clinical-trial documentation assistant. Write clear, precise, compliant text. Follow retrieved guidance carefully and avoid ambiguous statements."],
  ["human", "TASK: {task}\n\nCONTEXT (guidance snippets):\n{context}\n\nWrite the requested section. Use neutral tone and professional clinical-trial style."]
]);

const REVISION_PROMPT = ChatPromptTemplate.fromMessages([
  ["system", "You are a meticulous compliance editor for clinical-trial documents."],
  ["human", "Revise the DRAFT to resolve the following compliance issues. Preserve meaning and structure.\n\nDRAFT:\n{draft}\n\nCOMPLIANCE ISSUES:\n{issues}\n\nReturn the full revised text, improved but not overly verbose."]
]);

function complianceCheck(text) {
  const issues = [];
  if (/\bTBD\b/i.test(text) || /to be determined/i.test(text)) issues.push({ rule: "no_placeholders", message: "Remove TBD/placeholder language." });
  if (/risk/i.test(text) && !/mitigation/i.test(text)) issues.push({ rule: "risk_mitigation", message: "Mention risk mitigation when risks are discussed." });
  if (/consent/i.test(text) && !/withdraw/i.test(text)) issues.push({ rule: "consent_withdrawal", message: "State withdrawal rights in consent context." });
  const wc = (text.trim().match(/\S+/g) || []).length; if (wc < 150) issues.push({ rule: "min_length", message: "Provide at least ~150 words for sufficient detail." });
  return { ok: issues.length === 0, issues };
}

function fmtContext(docs) {
  return docs.map((d, i) => `- [${i + 1}] (${path.basename(d.metadata?.source || "doc")}) ${String(d.pageContent).replace(/\n/g, " ").slice(0, 350)}...`).join("\n");
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
  const issuesStr = (state.issues || []).map(i => `- ${i.rule}: ${i.message}`).join("\n");
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

const DocState = Annotation.Root({
  task: Annotation(),
  context: Annotation(),
  draft: Annotation(),
  issues: Annotation(),
  i: Annotation({ reducer: (_, next) => next }),
  maxIters: Annotation(),
});

function buildGraph(retriever) {
  const graph = new StateGraph(DocState);
  graph.addNode("retrieve", makeRetrieveNode(retriever));
  graph.addNode("draft", draftNode);
  graph.addNode("check", checkNode);
  graph.addNode("revise", reviseNode);
  graph.setEntryPoint("retrieve");
  graph.addEdge("retrieve", "draft");
  graph.addEdge("draft", "check");
  graph.addConditionalEdges("check", decideNext, { revise: "revise", end: END });
  graph.addEdge("revise", "check");
  return graph.compile();
}

// ---------------- Server & UI ----------------
const app = express();
app.use(cors());
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true, limit: "10mb" }));

app.get("/health", (_req, res) => res.json({ ok: true }));

const uploadDir = path.join(__dirname, "uploads", "pdfs");
fs.mkdirSync(uploadDir, { recursive: true });
const upload = multer({ dest: uploadDir });

let GLOBAL_VS = null;

async function ensureVS(pdfDir = uploadDir) {
  if (GLOBAL_VS) return GLOBAL_VS;
  const embeddings = new XenovaEmbeddings();
  const vs = new SimpleVectorStore(embeddings);
  try {
    const docs = await loadPDFs(pdfDir);
    await vs.addDocuments(docs);
  } catch (_) {}
  GLOBAL_VS = vs;
  return vs;
}

const jobs = new Map();
function pushEvent(jobId, payload) {
  const job = jobs.get(jobId);
  if (!job) return;
  job.events.push(payload);
  if (job.emitter) job.emitter.write(`data: ${JSON.stringify(payload)}\\n\\n`);
}

// UI page (escaped ${} in script)
app.get("/", (_req, res) => {
  res.type("html").send(`<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TrialScribe UI</title>
  <style>
    body{font-family:system-ui,Arial,sans-serif;margin:0;background:#0b1020;color:#e7ecf3}
    header{padding:16px 24px;border-bottom:1px solid #1c2340;display:flex;gap:12px;align-items:center}
    main{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:16px}
    section{background:#121938;border:1px solid #1c2340;border-radius:12px;padding:16px}
    h2{margin:0 0 8px 0;font-size:18px}
    textarea,input,button{font:inherit}
    textarea{width:100%;min-height:160px;background:#0e1530;color:#e7ecf3;border:1px solid #26305c;border-radius:8px;padding:10px}
    input[type="text"]{width:100%;background:#0e1530;color:#e7ecf3;border:1px solid #26305c;border-radius:8px;padding:8px}
    .row{display:flex;gap:8px;align-items:center}
    .tag{display:inline-block;padding:2px 8px;border-radius:999px;background:#1e274d;color:#9db0ff;border:1px solid #2b3770;font-size:12px}
    .log{white-space:pre-wrap;font-family:ui-monospace,Menlo,Consolas,monospace;background:#0d132b;border:1px solid #1f2750;border-radius:8px;padding:10px;max-height:260px;overflow:auto}
    .ok{color:#7ee787}.warn{color:#f1c40f}.err{color:#ff7676}
    .pill{padding:4px 8px;border-radius:8px;background:#0e1530;border:1px solid #26305c}
    .issues{margin:0;padding-left:18px}
  </style>
</head>
<body>
  <header>
    <div style="font-weight:700;letter-spacing:.4px">TrialScribe</div>
    <span class="tag">LangGraph + Claude</span>
    <span class="tag">Local embeddings</span>
    <span class="tag">PDF ingestion</span>
  </header>
  <main>
    <section>
      <h2>Author</h2>
      <div class="row"><input id="task" type="text" placeholder="Describe the section to draft (e.g., Protocol Synopsis for Phase II oncology)"/></div>
      <div style="height:8px"></div>
      <div class="row"><input id="pdfdir" type="text" placeholder="Optional PDF folder (server)"/></div>
      <div style="height:8px"></div>
      <div class="row">
        <input id="file" type="file" accept="application/pdf" multiple/>
        <button type="button" id="upload">Upload PDFs</button>
        <button type="button" id="ingest">Ingest Server PDFs</button>
        <button type="button" id="run">Run</button>
      </div>
      <div style="height:8px"></div>
      <div class="log" id="stream"></div>
    </section>

    <section>
      <h2>Reviewer</h2>
      <div class="pill"><strong>Final Draft</strong></div>
      <div id="final" class="log" style="min-height:120px"></div>
      <div style="height:8px"></div>
      <div class="pill"><strong>Compliance Issues</strong></div>
      <ul id="issues" class="issues"></ul>
    </section>
  </main>
<script>
const $ = (s) => document.querySelector(s);
const streamEl = document.querySelector('#stream');
const finalEl = document.querySelector('#final');
const issuesEl = document.querySelector('#issues');

function log(txt, cls=''){
  const d = document.createElement('div');
  if(cls) d.className = cls;
  d.textContent = txt;
  streamEl.appendChild(d);
  streamEl.scrollTop = streamEl.scrollHeight;
}

async function postJSON(url, data){
  const res = await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  const text = await res.text();
  if(!res.ok){ throw new Error(\`HTTP \${res.status} \${res.statusText}: \${text}\`); }
  return text ? JSON.parse(text) : {};
}

document.querySelector('#upload').onclick = async () => {
  try{
    const files = document.querySelector('#file').files;
    if(!files.length) return alert('Choose PDF files first');
    const fd = new FormData();
    for(const f of files) fd.append('pdfs', f, f.name);
    log('Uploading PDFs…');
    const res = await fetch('/api/upload', {method:'POST', body:fd});
    const text = await res.text();
    if(!res.ok) throw new Error(\`Upload failed: \${text}\`);
    const data = text ? JSON.parse(text) : {};
    log(\`Uploaded \${data.files?.length ?? 0} file(s).\`, 'ok');
    alert('Uploaded. Click \"Ingest Server PDFs\" to index them.');
  }catch(err){
    console.error(err);
    log(\`Upload error: \${err.message}\`, 'err');
    alert(\`Upload error: \${err.message}\`);
  }
};

document.querySelector('#ingest').onclick = async () => {
  try{
    const dir = document.querySelector('#pdfdir').value || '';
    log(\`Ingesting PDFs from \"\${dir || 'uploads/pdfs'}\"…\`);
    const r = await postJSON('/api/ingest', { dir });
    log(r.message || 'Ingest complete.', 'ok');
    alert(r.message || 'Ingest complete.');
  }catch(err){
    console.error(err);
    log(\`Ingest error: \${err.message}\`, 'err');
    alert(\`Ingest error: \${err.message}\`);
  }
};

document.querySelector('#run').onclick = async () => {
  try{
    streamEl.textContent=''; finalEl.textContent=''; issuesEl.innerHTML='';
    const task = document.querySelector('#task').value || 'Write a \"Protocol Synopsis\" paragraph for a Phase II oncology trial including design, eligibility, primary endpoint, AE reporting, data protection, and informed consent.';
    log('Starting run…');
    const { jobId } = await postJSON('/api/run', { task, stream:true });
    if(!jobId) throw new Error('No jobId returned.');
    const es = new EventSource('/api/stream/' + jobId);
    es.onmessage = (e) => {
      try{
        const evt = JSON.parse(e.data);
        if(evt.type==='node'){
          const { node, delta } = evt;
          if(delta?.context) log(\`[\${node}] context: \${delta.context.slice(0,160)}…\`, 'ok');
          if(delta?.draft)   log(\`[\${node}] draft: \${delta.draft.replace(/\\n/g,' ').slice(0,200)}…\`);
          if(Array.isArray(delta?.issues)){
            const rules = delta.issues.map(i=>i.rule).join(', ') || 'none';
            log(\`[\${node}] issues: \${rules}\`, delta.issues.length ? 'warn' : 'ok');
          }
          if(typeof delta?.i !== 'undefined') log(\`[\${node}] iteration: \${delta.i}\`);
        }
        if(evt.type==='final'){
          const draft = evt.final?.draft || '';
          const issues = Array.isArray(evt.final?.issues) ? evt.final.issues : [];
          finalEl.textContent = draft;
          issuesEl.innerHTML = issues.map(i=> \`• <strong>\${i.rule}</strong> — \${i.message}\`).map(x=>\`<li>\${x}</li>\`).join('');
          log('Final received.', 'ok');
        }
        if(evt.type==='done'){
          es.close();
          log('Run complete.', 'ok');
        }
      }catch(err){
        console.error(err);
        log(\`Stream parse error: \${err.message}\`, 'err');
      }
    };
    es.onerror = () => { es.close(); log('Stream closed (error).', 'warn'); };
  }catch(err){
    console.error(err);
    log(\`Run error: \${err.message}\`, 'err');
    alert(\`Run error: \${err.message}\`);
  }
};
</script>
</body>
</html>`);
});

// Upload PDFs
app.post("/api/upload", upload.array("pdfs", 20), async (req, res) => {
  try {
    console.log("[/api/upload] files:", (req.files || []).map(f => f.originalname));
    res.json({ ok: true, files: (req.files || []).map(f => ({ filename: f.originalname })) });
  } catch (e) {
    console.error("upload error", e);
    res.status(500).json({ ok: false, error: e?.message || String(e) });
  }
});

// Ingest PDFs
app.post("/api/ingest", async (req, res) => {
  try {
    const dir = (req.body?.dir && String(req.body.dir).trim()) || uploadDir;
    console.log("[/api/ingest] dir:", dir);
    const embeddings = new XenovaEmbeddings();
    const vs = new SimpleVectorStore(embeddings);
    const docs = await loadPDFs(dir);
    await vs.addDocuments(docs);
    GLOBAL_VS = vs;
    res.json({ ok: true, message: `Ingested ${docs.length} chunks from ${dir}` });
  } catch (e) {
    console.error("ingest error", e);
    res.status(500).json({ ok: false, error: e?.message || String(e) });
  }
});

// Run and stream
app.post("/api/run", async (req, res) => {
  try {
    const vs = await ensureVS();
    const retriever = vs.asRetriever({ k: 4 });
    const graph = buildGraph(retriever);

    const jobId = nanoid();
    jobs.set(jobId, { events: [], final: null, emitter: null, done: false });

    const task = req.body?.task || "Draft a Protocol Synopsis paragraph.";
    const init = { task, i: 0, maxIters: 2 };

    console.log("[/api/run] jobId:", jobId, "task:", task);

    (async () => {
      let streamInstance = null;
      try { streamInstance = graph.stream(init); } catch (e) { console.log("stream not supported", e?.message); }

      if (streamInstance && typeof streamInstance[Symbol.asyncIterator] === "function") {
        for await (const event of streamInstance) {
          for (const [node, delta] of Object.entries(event)) {
            pushEvent(jobId, { type: "node", node, delta });
          }
        }
        const final = await graph.invoke(init);
        jobs.get(jobId).final = final;
        pushEvent(jobId, { type: "final", final });
        pushEvent(jobId, { type: "done" });
        jobs.get(jobId).done = true;
      } else {
        const final = await graph.invoke(init);
        jobs.get(jobId).final = final;
        pushEvent(jobId, { type: "final", final });
        pushEvent(jobId, { type: "done" });
        jobs.get(jobId).done = true;
      }
    })();

    res.json({ ok: true, jobId });
  } catch (e) {
    console.error("run error", e);
    res.status(500).json({ ok: false, error: e?.message || String(e) });
  }
});

// SSE stream
app.get("/api/stream/:jobId", (req, res) => {
  const { jobId } = req.params;
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders?.();

  const job = jobs.get(jobId);
  if (!job) { res.write(`data: ${JSON.stringify({ type:'error', error:'no such job' })}\n\n`); return res.end(); }

  job.emitter = res;
  for (const evt of job.events) res.write(`data: ${JSON.stringify(evt)}\n\n`);
  if (job.done) { res.write(`data: ${JSON.stringify({ type:'done' })}\n\n`); return res.end(); }

  req.on("close", () => {});
});

// Viz
app.get("/api/viz", (_req, res) => {
  const mmd = `flowchart TD
  R[retrieve] --> D[draft]
  D --> C[check]
  C -- no issues / max iters --> E{{END}}
  C -- issues remain --> V[revise]
  V --> C
`;
  res.type("text/plain").send(mmd);
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`TrialScribe UI running → http://localhost:${PORT}`);
});
