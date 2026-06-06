<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/873489e9-28cc-404d-b9d0-231bf82dcc4a" />

# SpinRAG

> **Status: `v0.1.0a1` — first public alpha.** A typed, self-evolving knowledge graph for RAG over **damaged corpora**. Tuned for **document restoration** first: faithful reconstruction is prioritized, and hallucination risk is explicitly bounded by the spin of the query.

---

## The problem SpinRAG solves

Vanilla vector RAG assumes your corpus is clean. Real-world knowledge bases — scraped notes, OCR dumps, half-finished wikis, crash-recovered transcripts, leaked logs — are not. They are a mix of:

- **complete entries** (`Mars Aries × Venus Taurus | The Gilded War-Horn: an artifact that summons a phantom bull made of pure gold…`),
- **truncated entries** (`* Pluto Aquarius -`),
- **bare tokens** (`pluto`),
- and **schema/parameter rows** that describe structure without content.

A flat vector index treats all four as equal points in a cloud. The broken lines pollute retrieval and there is no mechanism to ever *repair* them. Once damaged, always damaged.

SpinRAG starts from the opposite assumption: damaged data is the **input signal**, not noise. Broken fragments are first-class catalysts that *drive the graph to repair itself*.

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

1. **Typed spins as a 4-state algebra.** Every chunk is classified by a small language model into one of four roles — `TOP` (self-contained concept), `BOTTOM` (complex target), `LEFT` (incomplete fragment), `RIGHT` (definition / parameter structure). These types are not metadata; they are the **only thing that decides which production rule applies**.

2. **Evolution by production rules.** Across `n_epochs`, `LEFT` and `RIGHT` chunks act as catalysts that find their *metrically closest* `TOP` / `BOTTOM` partner and fuse with it under a restoration-biased prompt. New `TOP` and `BOTTOM` documents are minted from the fusion, the graph densifies, and incomplete fragments are gradually re-expressed as complete definitions. The corpus literally rewrites itself toward coherence.

3. **A bounded hallucination contract.** Because the spin of the *query* selects the production rule, the user can choose their fidelity / synthesis trade per-query. `TOP` queries return the closest stored document **verbatim** — zero LLM rewriting, zero hallucination. `LEFT` / `RIGHT` / `BOTTOM` queries pass through guardrailed generative rules. The user opts in to invention; they don't get it by default.

---

## Concrete benefits

- 🩹 **Recovers content from corrupted dumps.** Half-finished lines (`* Pluto Aquarius -`) get re-merged with the most semantically related complete entry, instead of being dropped.
- 📈 **Self-improving index.** Re-running with more epochs produces a denser, more self-consistent graph without re-ingesting raw source.
- 🔬 **Visible provenance.** Every generated document carries an `epoch_history` and `graph["edges"]` that point back to the catalyst and base that produced it. Audit-friendly.
- 🎚️ **Tunable hallucination risk.** TOP-only deployments are fully extractive; enabling LEFT/RIGHT/BOTTOM queries turns on guardrailed synthesis.
- 🤖 **Runs on small local models.** Ollama + a 4 B chat model + `nomic-embed-text` is enough. No API keys, no cloud, no data exfiltration.
- 🔗 **LangChain-native.** Drops into existing LangChain pipelines as a retriever-style backend.

---

## The "spin" of a chunk

Each line of input becomes a particle whose spin determines its dynamics. A small language model assigns the initial spin.

| Spin       | Icon | Meaning                                                                  | Example from `demo-data.txt` |
| :--------- | :--: | :----------------------------------------------------------------------- | :--- |
| **TOP**    |  ⬆️  | Self-contained name or concept. **Stored verbatim, retrievable directly.** | `Mars Aries × Sun Leo \| The Arena of Ascended Champions: a mythical battleground where mortals can challenge legendary heroes.` |
| **BOTTOM** |  ⬇️  | Complex / a target waiting to be crystallized.                            | A long passage describing an evolving ritual that needs definitions to lock in. |
| **LEFT**   |  ⬅️  | Incomplete — missing information to be understood.                        | `* Pluto Aquarius -` &nbsp; / &nbsp; `pluto` |
| **RIGHT**  |  ➡️  | Definition or parameter structure.                                        | `Jupiter Aries — Neptune Pisces` (a schema-shaped row with no body) |

---

## The four production rules

During each evolution epoch, every `LEFT` and `RIGHT` catalyst is matched to its **single closest** `TOP` / `BOTTOM` partner (cosine similarity over Ollama embeddings) and fused under the restoration guardrail:

| Catalyst | Base | Produces | Intuition |
|---|---|---|---|
| **LEFT** | **TOP** | new **TOP** | An incomplete fragment is *repaired* against the nearest complete concept. The damaged data is the trigger; the intact concept supplies the missing facts. |
| **RIGHT** | **TOP** | new **TOP** | A definition *resonates* with a concept and births a more fully-specified concept. |
| **RIGHT** | **BOTTOM** | new **TOP** | A definition *crystallizes* a complex target into a self-contained concept. |
| **LEFT** | **BOTTOM** | new **BOTTOM** | An incomplete fragment *thickens* a complex target into a richer, still-complex target (next epoch may crystallize it via a RIGHT). |

Catalysts are *not* consumed — they remain available every epoch. The base sets grow as new `TOP` / `BOTTOM` documents are minted, so the next epoch finds different nearest neighbors. The system converges (or halts) when no rule produces a new document.

---

## How a query is answered

1. The query string is classified into a spin.
2. The closest `TOP` document is found by cosine similarity over the embedded `TOP` set.
3. The **query production rules** fire based on the query's spin:

| Query spin | Behaviour | Hallucination risk |
|---|---|---|
| **TOP**    | Return the closest `TOP` document **verbatim**. | **None.** Pure extractive retrieval. |
| **LEFT**   | Restore the missing parameter structure of the query using the closest `TOP` as the source of truth. | Bounded by guardrail. |
| **RIGHT**  | Project a complex target that preserves every fact in both the query and the closest `TOP`. | Bounded by guardrail. |
| **BOTTOM** | Walk the graph: find the most relevant `LEFT` / `RIGHT` *adjacent to* the closest `TOP` (i.e. its actual provenance neighbors), then fuse. | Bounded by guardrail + provenance. |

The `BOTTOM` path is the one that makes the typed graph pay off: instead of doing a global k-NN over `LEFT` / `RIGHT`, it traverses **only the documents that helped produce the closest TOP**. The answer is shaped by the graph topology, not the embedding cloud.

---

## A worked example

Given a corpus where most lines look like

```
Mars Aries × Venus Libra | The Duelist's Challenge: a spell that creates a pocket dimension where two combatants are forced into a fair, one-on-one fight…
```

and a few are broken

```
Jupiter Aries — Neptune Pisce
* Pluto Aquarius -
pluto
```

A vanilla vector RAG indexes all six lines as points; the three broken lines show up as confident-but-empty matches and degrade every answer.

SpinRAG, after `n_epochs=3`:

1. Classifies the complete rows as `TOP`, the `Jupiter Aries — Neptune Pisce` schema-shaped row as `RIGHT`, the truncated rows as `LEFT`.
2. **Epoch 1:** `pluto` (LEFT) finds its closest `TOP` (some Pluto/Scorpio entry) and births a new `TOP` that incorporates the bare token into a complete definition. `Jupiter Aries — Neptune Pisce` (RIGHT) resonates with the nearest Jupiter/Sagittarius TOP and produces a new TOP with the Jupiter-Aries-to-Neptune-Pisces relationship now spelled out.
3. **Epoch 2–3:** the freshly minted `TOP` documents are themselves available as bases for the same catalysts, so the broken lines fuse with progressively more specific neighbors.
4. After evolution, all `TOP` documents are embedded.
5. A query like *"what does Pluto Aquarius do?"* — classified as `BOTTOM` (a complex target without a definition) — retrieves the closest `TOP`, then fuses with the `LEFT`/`RIGHT` adjacent to that `TOP` (which include the original `* Pluto Aquarius -` fragment as provenance), and returns a restored answer that traces back to the actual source line.

The verbose log preserves every decision; the graph preserves every edge.

---

## Getting started

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) running locally with **two** models pulled:

  ```bash
  # Generation model (any chat/instruct model works)
  ollama pull qwen3:4b-instruct-2507

  # Dedicated embedding model — required because most chat models
  # do NOT expose an embeddings endpoint.
  ollama pull nomic-embed-text
  ```

### Installation

```bash
git clone https://github.com/iblameandrew/spin-rag.git
cd spin-rag
pip install -r requirements.txt
pip install -e .
```

### Programmatic usage

```python
from spin_rag import SpinRAG

with open("path/to/your/knowledge_base.txt", "r", encoding="utf-8") as f:
    content = f.read()

rag = SpinRAG(
    content=content,
    n_epochs=5,
    llm_model="qwen3:4b-instruct-2507",
    embed_model="nomic-embed-text",
)

# Extractive query (TOP): returned verbatim, zero hallucination.
print(rag.query("The Arena of Ascended Champions"))

# Restoration query (LEFT): the broken fragment is completed against
# the nearest TOP under the restoration guardrail.
print(rag.query("* Pluto Aquarius -"))

# Provenance-aware synthesis (BOTTOM): walks the graph from the closest
# TOP through its LEFT/RIGHT neighbors.
print(rag.query("Describe an Aquarius-aspected pluto ritual."))

# Audit trail
for line in rag.get_verbose_log():
    print(line)
```

### Dash demo

```bash
python demo.py
```

Open <http://127.0.0.1:8050>, upload a `.txt`, pick your models and epoch count, click **Initialize RAG**, then chat. The left panel streams the verbose evolution log in real time.

---

## Public API

```python
from spin_rag import SpinRAG, SpinType, Document
```

**`SpinRAG(content, n_epochs=5, llm_model="llama2", embed_model=None, config=None, logger_callback=None)`**

- `content` — raw text; one document per non-empty line.
- `n_epochs` — number of evolution epochs.
- `llm_model` — Ollama generation model.
- `embed_model` — Ollama embedding model. Defaults to `nomic-embed-text` because most chat-tuned models cannot embed.
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

## Known limitations of this alpha

- **No persistence.** The graph lives in memory; restart = re-evolve.
- **No batching.** Each catalyst interaction is a separate LLM call; large corpora are slow.
- **No automated tests.** Behaviour is validated only by the demo flow and manual inspection.
- **Embedding model required.** Reusing a single chat model for both generation and embeddings is no longer supported by default — chat-tuned models typically reject `/api/embeddings`.
- **Hallucination on damaged input.** The restoration guardrail reduces but does not eliminate fabrication. Treat `LEFT` / `RIGHT` / `BOTTOM` query outputs as drafts.

---

## Changelog

### v0.1.0a1 — 2026-06-06 (first public alpha)

Restoration-first hardening pass and exhaustive bug hunt.

**Packaging (the package was effectively un-installable before this release)**
- `setup.py` previously declared `py_modules=['spinlm']` for a module that did not exist; replaced with `find_packages()` and corrected project metadata (name, author, URL, classifiers, keywords).
- `spin_rag/__init__.py` was empty; now re-exports `SpinRAG`, `SpinType`, `Document`, and `__version__`.
- `requirements.txt` was missing `langchain-ollama` (imported by `spin_rag.py`) and pinned no minimum versions; both fixed.

**Core runtime bugs (`spin_rag/spin_rag.py`)**
- `OllamaEmbeddings(model=llm_model)` crashed initialization on chat-tuned models (qwen3-instruct, llama-chat, etc.) which do not expose `/api/embeddings`. Split into a separate `embed_model` parameter defaulting to `nomic-embed-text`.
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
