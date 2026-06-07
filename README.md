<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/873489e9-28cc-404d-b9d0-231bf82dcc4a" />

# SpinRAG

> A typed, self-evolving knowledge graph for RAG over **damaged business records** — fiscal ledgers, OCR dumps, half-finished wiki entries, ticket exports. SpinRAG treats broken fragments as the *input signal* and uses the LLM to repair the graph into a self-consistent retrieval index.

`v0.1.0` — first official release. OpenAI-SDK compatible: ships with **OpenRouter** and a self-hosted **llama.cpp** server as first-class backends.

---

## The problem SpinRAG solves

Plain vector RAG assumes your corpus is clean. Real-world record dumps — exported ledgers, scanned invoices, support tickets, crash-recovered notebooks — are not. They mix:

- **complete entries** — `Revenue 2026-Q1-006 | Client: Globex Industries | Period: 2026-Q1 | Amount: $128,000.00 | Status: received | Product: Platform license`,
- **schema-only rows** — `Vendor: Initech | Period: 2026 | Type: revenue`,
- **truncated rows** — `Acme Corp. -`,
- **bare tokens** — `Consulting`, `Pied Piper`.

A flat vector index treats all four as equal points in a cloud. The broken rows pollute retrieval and there is no mechanism to ever *repair* them. Once damaged, always damaged.

SpinRAG starts from the opposite assumption: damaged data is the **input signal**, not noise. Broken fragments are first-class catalysts that drive the graph to repair itself.

---

## What makes SpinRAG novel

| | Vanilla / vector RAG | GraphRAG / KG-RAG | **SpinRAG** |
|---|---|---|---|
| Chunk role | untyped point | untyped node | **typed particle (spin)** |
| Index step | one-shot embed | one-shot extract entities | **iterative LLM-mediated fusion across epochs** |
| Damaged rows | pollute results | dropped / ignored | **become catalysts that repair the graph** |
| Retrieval | k-NN | community / path | **typed k-NN + adjacency on provenance edges** |
| Hallucination contract | none | none | **explicit per-query: TOP = verbatim, others = guardrailed** |

The three things that are actually new:

1. **Typed spins as a 4-state algebra.** Every chunk is classified by a small language model into one of four roles — `TOP` (self-contained entry), `BOTTOM` (complex target), `LEFT` (incomplete fragment), `RIGHT` (schema / parameter row). These types are not metadata; they are the **only thing that decides which production rule applies**.

2. **Evolution by production rules.** Across `n_epochs`, `LEFT` and `RIGHT` chunks act as catalysts that find their *metrically closest* `TOP` / `BOTTOM` partner and fuse with it under a restoration-biased prompt. New `TOP` and `BOTTOM` documents are minted from the fusion, the graph densifies, and incomplete fragments are gradually re-expressed as complete entries. The corpus literally rewrites itself toward coherence.

3. **A bounded hallucination contract.** Because the spin of the *query* selects the production rule, the user can choose their fidelity / synthesis trade per-query. `TOP` queries return the closest stored document **verbatim** — zero LLM rewriting, zero hallucination. `LEFT` / `RIGHT` / `BOTTOM` queries pass through guardrailed generative rules. The user opts in to invention; they don't get it by default.

---

## Concrete benefits

- 🩹 **Recovers content from corrupted exports.** Truncated lines (`Acme Corp. -`) get re-merged with the most semantically related complete row instead of being dropped.
- 📈 **Self-improving index.** Re-running with more epochs produces a denser, more self-consistent graph without re-ingesting raw source.
- 🔬 **Visible provenance.** Every generated document carries an `epoch_history` and `graph["edges"]` that point back to the catalyst and base that produced it. Audit-friendly.
- 🎚️ **Tunable hallucination risk.** TOP-only deployments are fully extractive; enabling LEFT/RIGHT/BOTTOM queries turns on guardrailed synthesis.
- 🔌 **OpenAI-compatible backends.** Drop-in for OpenRouter-hosted models or a self-hosted `llama.cpp` server. No vendor lock-in.
- 🔗 **No API key needed for local.** Run fully on-prem with `llama-server` and your own GGUF weights.

---

## The "spin" of a chunk

Each line of input becomes a particle whose spin determines its dynamics. A small language model assigns the initial spin.

| Spin       | Icon | Meaning                                                                  | Example from `demo-data.txt` |
| :--------- | :--: | :----------------------------------------------------------------------- | :--- |
| **TOP**    |  ⬆️  | Self-contained entry. **Stored verbatim, retrievable directly.**         | `Revenue 2026-Q1-006 \| Client: Globex Industries \| Period: 2026-Q1 \| Amount: $128,000.00 \| Status: received \| Product: Platform license` |
| **BOTTOM** |  ⬇️  | Complex entry / a target waiting to be crystallized.                      | A long memo referencing several open ledger items that need structure. |
| **LEFT**   |  ⬅️  | Incomplete — missing information to be understood.                        | `Acme Corp. -` &nbsp; / &nbsp; `Consulting` &nbsp; / &nbsp; `Pied Piper` |
| **RIGHT**  |  ➡️  | Schema or parameter row.                                                 | `Vendor: Initech \| Period: 2026 \| Type: revenue` |

---

## The four production rules

During each evolution epoch, every `LEFT` and `RIGHT` catalyst is matched to its **single closest** `TOP` / `BOTTOM` partner (cosine similarity over OpenRouter / llama.cpp embeddings) and fused under the restoration guardrail:

| Catalyst | Base | Produces | Intuition |
|---|---|---|---|
| **LEFT** | **TOP** | new **TOP** | An incomplete fragment is *repaired* against the nearest complete entry. The damaged row is the trigger; the intact entry supplies the missing fields. |
| **RIGHT** | **TOP** | new **TOP** | A schema row *resonates* with a complete entry and births a more fully-specified entry. |
| **RIGHT** | **BOTTOM** | new **TOP** | A schema row *crystallizes* a complex entry into a self-contained one. |
| **LEFT** | **BOTTOM** | new **BOTTOM** | An incomplete fragment *thickens* a complex entry into a richer, still-complex entry (next epoch may crystallize it via a RIGHT). |

Catalysts are *not* consumed — they remain available every epoch. The base sets grow as new `TOP` / `BOTTOM` documents are minted, so the next epoch finds different nearest neighbors. The system converges (or halts) when no rule produces a new document.

---

## How a query is answered

1. The query string is classified into a spin.
2. The closest `TOP` document is found by cosine similarity over the embedded `TOP` set.
3. The **query production rules** fire based on the query's spin:

| Query spin | Behaviour | Hallucination risk |
|---|---|---|
| **TOP**    | Return the closest `TOP` document **verbatim**. | **None.** Pure extractive retrieval. |
| **LEFT**   | Restore the missing fields of the query using the closest `TOP` as the source of truth. | Bounded by guardrail. |
| **RIGHT**  | Project a complex target that preserves every fact in both the query and the closest `TOP`. | Bounded by guardrail. |
| **BOTTOM** | Walk the graph: find the most relevant `LEFT` / `RIGHT` *adjacent to* the closest `TOP` (i.e. its actual provenance neighbors), then fuse. | Bounded by guardrail + provenance. |

The `BOTTOM` path is the one that makes the typed graph pay off: instead of doing a global k-NN over `LEFT` / `RIGHT`, it traverses **only the documents that helped produce the closest TOP**. The answer is shaped by the graph topology, not the embedding cloud.

---

## A worked example

Given a corpus where most lines look like

```
Revenue 2026-Q1-006 | Client: Globex Industries | Period: 2026-Q1 | Amount: $128,000.00 | Status: received | Product: Platform license
```

and a few are broken

```
Acme Corp. -
Consulting
Pied Piper
Vendor: Initech | Period: 2026 | Type: revenue
```

A vanilla vector RAG indexes all five lines as points; the four broken lines show up as confident-but-empty matches and degrade every answer.

SpinRAG, after `n_epochs=3`:

1. Classifies the complete rows as `TOP`, the `Vendor: Initech | Period: 2026 | Type: revenue` schema row as `RIGHT`, the truncated / bare rows as `LEFT`.
2. **Epoch 1:** `Consulting` (LEFT) finds its closest `TOP` (some revenue line where the product is a consulting engagement) and births a new `TOP` that incorporates the bare token into a complete entry. The Initech `RIGHT` row resonates with the nearest revenue entry mentioning Initech and produces a new `TOP` with the Initech 2026 revenue now spelled out.
3. **Epoch 2–3:** the freshly minted `TOP` documents are themselves available as bases for the same catalysts, so the broken lines fuse with progressively more specific neighbors.
4. After evolution, all `TOP` documents are embedded.
5. A query like *"What revenue did Acme Corp. generate in 2026-Q1?"* — classified as `BOTTOM` (a complex target without a definition) — retrieves the closest `TOP`, then fuses with the `LEFT`/`RIGHT` adjacent to that `TOP` (which include the original `Acme Corp. -` fragment as provenance), and returns a restored answer that traces back to the actual source line.

The verbose log preserves every decision; the graph preserves every edge.

---

## Getting started

### Prerequisites

- Python 3.8+
- A backend — pick **one** of:
  - An **OpenRouter** account and API key, *or*
  - A locally-running **`llama.cpp` server** (built with the `server` target, exposing its OpenAI-compatible `/v1` endpoint). Start it with the embedding flag enabled if you want to use the embeddings endpoint as well:

    ```bash
    ./llama-server -m model.gguf --port 8080 --embedding
    ```

### Installation

```bash
git clone https://github.com/iblameandrew/spin-rag.git
cd spin-rag
pip install -r requirements.txt
pip install -e .

cp .env.example .env
# Edit .env to set OPENROUTER_API_KEY (if using OpenRouter).
```

### Programmatic usage

```python
from spin_rag import SpinRAG, BACKEND_OPENROUTER, BACKEND_LLAMACPP

with open("path/to/your/ledger_dump.txt", "r", encoding="utf-8") as f:
    content = f.read()

# --- Option A: OpenRouter (default) ----------------------------------------
rag = SpinRAG(
    content=content,
    n_epochs=5,
    backend=BACKEND_OPENROUTER,
    llm_model="openai/gpt-4o-mini",
    embed_model="openai/text-embedding-3-small",
    api_key="sk-or-...",  # or set OPENROUTER_API_KEY
)

# --- Option B: self-hosted llama.cpp server --------------------------------
rag = SpinRAG(
    content=content,
    n_epochs=5,
    backend=BACKEND_LLAMACPP,
    base_url="http://localhost:8080/v1",
    llm_model="llama",   # whatever alias the server has loaded
    embed_model="llama", # same model, used in --embedding mode
)

# Extractive query (TOP): returned verbatim, zero hallucination.
print(rag.query("Globex Industries Q1 revenue"))

# Restoration query (LEFT): the broken fragment is completed against
# the nearest TOP under the restoration guardrail.
print(rag.query("Acme Corp. -"))

# Provenance-aware synthesis (BOTTOM): walks the graph from the closest
# TOP through its LEFT/RIGHT neighbors.
print(rag.query("Summarize 2026-Q1 receivables."))

# Audit trail
for line in rag.get_verbose_log():
    print(line)
```

### Dash demo

```bash
python demo.py
```

Open <http://127.0.0.1:8050>, upload a `.txt`, pick your backend, paste the API key (if any), pick your models and epoch count, click **Initialize RAG**, then chat. The left panel streams the verbose evolution log in real time.

---

## Public API

```python
from spin_rag import (
    SpinRAG,
    SpinType,
    Document,
    BACKEND_OPENROUTER,
    BACKEND_LLAMACPP,
)
```

**`SpinRAG(content, n_epochs=5, llm_model=..., embed_model=..., backend=BACKEND_OPENROUTER, base_url=None, api_key=None, app_name=None, site_url=None, config=None, logger_callback=None)`**

- `content` — raw text; one document per non-empty line.
- `n_epochs` — number of evolution epochs.
- `llm_model` — model identifier for the chosen backend. For OpenRouter, the full model ID (e.g. `openai/gpt-4o-mini`). For llama.cpp, the alias under which the model was loaded on the server.
- `embed_model` — embedding model identifier (same conventions as `llm_model`). Defaults to a sensible value per backend.
- `backend` — `BACKEND_OPENROUTER` (default) or `BACKEND_LLAMACPP`.
- `base_url` — override the default base URL. `None` means use the backend's default (`https://openrouter.ai/api/v1` or `http://localhost:8080/v1`) or whatever is set in the relevant environment variable.
- `api_key` — API key. `None` means read from the relevant environment variable.
- `logger_callback` — optional `Callable[[str], None]` for streaming logs (used by `demo.py`).

**`SpinRAG.query(query_text, reorganize_graph=False) -> str`** — returns the restored answer. If `reorganize_graph=True`, the query and its response are added back into the graph as new nodes with provenance edges, so subsequent queries can see them.

**`SpinRAG.get_verbose_log() -> List[str]`** — snapshot of the evolution / query log.

Public dataclasses: `Document(id, text, spin, embeddings, epoch_history, metadata)`, `SpinType` (`TOP`, `BOTTOM`, `LEFT`, `RIGHT`).

---

## When *not* to use SpinRAG

- Your corpus is already clean and stable → a plain vector store will be faster and equally accurate.
- You need millisecond p99 retrieval on millions of documents → SpinRAG's evolution loop is `O(epochs × catalysts × base_size)` LLM calls; it is built for *quality* on *small-to-medium damaged* corpora, not raw throughput.
- You need a strict no-LLM-in-the-loop guarantee at *index time* → SpinRAG calls the LLM during evolution to mint repaired documents. (Query-time can still be no-LLM if you only issue TOP queries.)

---

## Considerations

- **Backend choice.** OpenRouter is the path of least resistance for hosted models; llama.cpp is the path of least resistance for fully-local deployments. Both speak the same OpenAI HTTP protocol, so swapping is one argument.
- **Hallucination on damaged input.** The restoration guardrail reduces but does not eliminate fabrication. Treat `LEFT` / `RIGHT` / `BOTTOM` query outputs as drafts and verify against the underlying records.
- **Persistence.** The graph lives in memory. For long-running use, persist `rag.documents` and `rag.graph` to disk between runs.
- **Cost / latency.** Evolution is LLM-heavy by design. Pick a small model for spin classification and a more capable model for fusion if you want to balance cost.

---

## Changelog

### v0.1.0 — 2026-06-06 (first official release)

The first release of SpinRAG marketed as production-ready. Ships with a typed, evolving knowledge-graph RAG that runs against **OpenRouter** (hosted) or a **llama.cpp** server (local), both via the OpenAI Python SDK.

**Backend refactor**
- Replaced the LangChain / Ollama stack with a thin wrapper around the official `openai` Python SDK. SpinRAG now talks the OpenAI-compatible HTTP protocol directly.
- Two backends supported out of the box, selectable via `backend=`:
  - `BACKEND_OPENROUTER` (default) — OpenRouter's hosted model catalog. Reads `OPENROUTER_API_KEY` from the environment; honors `OPENROUTER_BASE_URL`, `OPENROUTER_SITE_URL`, and `OPENROUTER_APP_NAME` for attribution headers.
  - `BACKEND_LLAMACPP` — a self-hosted `llama.cpp` server's built-in `/v1` endpoint. No API key required; default base URL is `http://localhost:8080/v1`.
- `langchain`, `langchain-community`, `langchain-ollama`, and `sentence-transformers` are no longer dependencies.

**Demo (`demo.py`)**
- Added a backend radio selector (OpenRouter vs llama.cpp) and an editable base-URL field.
- Added an API-key input; falls back to environment variables when empty.
- Model dropdowns now expose curated OpenRouter and llama.cpp options rather than a hard-coded Ollama list.

**Repo hygiene**
- Replaced `demo-data.txt` with a realistic fiscal-ledger dump so the worked example in the README is grounded in real-world records.
- Rewrote the README to be positioning-neutral and enterprise-readable: removed all decorative examples that read as fringe and replaced them with concrete ledger / invoice scenarios. The algorithm, novelty, and worked example are all carried by business records now.
- Added `.env.example` documenting every environment variable.

### v0.1.0a1 — 2026-06-06 (first public alpha — superseded)

Restoration-first hardening pass and exhaustive bug hunt.

**Packaging (the package was effectively un-installable before this release)**
- `setup.py` previously declared `py_modules=['spinlm']` for a module that did not exist; replaced with `find_packages()` and corrected project metadata (name, author, URL, classifiers, keywords).
- `spin_rag/__init__.py` was empty; now re-exports `SpinRAG`, `SpinType`, `Document`, and `__version__`.
- `requirements.txt` was missing `langchain-ollama` (imported by `spin_rag.py`) and pinned no minimum versions; both fixed.

**Core runtime bugs (`spin_rag/spin_rag.py`)**
- `OllamaEmbeddings(model=llm_model)` crashed initialization on chat-tuned models which do not expose `/api/embeddings`. Split into a separate `embed_model` parameter defaulting to `nomic-embed-text`.
- Cosine similarity divided by `np.linalg.norm` at three sites without a zero-norm guard; consolidated into a `_cosine_similarity` helper that returns `0.0` for degenerate vectors.
- `_find_closest_doc` re-embedded the entire base corpus on every catalyst interaction during evolution. Added a per-text embedding cache so each string is embedded at most once.
- `_get_spin` used naive `"TOP" in response` checks; LLM reasoning prefixes like *"Not a TOP, it is BOTTOM"* misclassified. Replaced with a `\b(TOP|BOTTOM|LEFT|RIGHT)\b` regex match.
- LLM outputs flowed into new documents unstripped, sometimes with surrounding quotes or stray markdown fences. Added a `_clean_llm_output` normaliser.
- Deprecated `LLMChain` replaced with direct `llm.invoke` calls; switched to the new `langchain-ollama` package with a fallback to `langchain-community` for older installs.
- `_log` crashed on Windows cp1252 consoles when printing emoji (`UnicodeEncodeError`); now falls back to ascii-safe output.
- Removed dead imports (`hashlib`, `json`, `time`, `Path`, `RecursiveCharacterTextSplitter`).

**Restoration-first prompts**
- Every evolution and query production rule is now wrapped in a `_RESTORATION_GUARDRAIL` that instructs the model to preserve every noun, number, and relationship from the source fragments and mark genuine unknowns with `[unknown]` instead of guessing.
- Pure `TOP`-spin queries are returned **verbatim** from the closest matching document with no LLM rewriting — full restoration, zero hallucination on that path.
- The default fallback inside `_apply_query_production_rules` now also returns the closest TOP document verbatim instead of dropping to an empty string.

**Demo (`demo.py`)**
- Initialize button never re-enabled after a run; replaced with an `init_in_progress` flag plus a live status indicator (idle / busy / ready).
- Separate selectors for the LLM and the embedding model.
- Race conditions on the shared `rag` instance and `log_stream` resolved with `_rag_lock` / `_state_lock`.
- Chat input now clears after send; pressing Enter submits; the verbose log is reset on each re-initialisation.
- Verbose log capped at the last 500 lines to keep the Dash UI responsive on long runs.
- Deprecated `llm.predict()` replaced with `llm.invoke()`; LLM failures degrade gracefully to showing the retrieved context.

**README**
- Corrected the documented import (`from spin_rag import SpinRAG`) and constructor signature (`content=`, `embed_model=`).
- Fixed the `pip install - r requirements.txt` typo.
- Added the *"What this alpha emphasizes"* section documenting the restoration / hallucination trade-off and a *"Known limitations"* section.

**.gitignore**
- Also ignores build artefacts (`build/`, `dist/`, `*.egg-info/`) and local AI-assistant tooling (`.claude/`, `AGENTS.md`, `CLAUDE.md`, `.gitnexus/`).

## License

MIT.
