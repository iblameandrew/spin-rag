<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/873489e9-28cc-404d-b9d0-231bf82dcc4a" />

# SpinRAG

> **Status: `v0.1.0a1` — first public alpha.** Tuned for **document restoration**: faithful reconstruction of damaged, partial, or fragmented knowledge bases. Hallucination risk is explicitly accepted in exchange for high fidelity on intact inputs.

**SpinRAG** is a Retrieval-Augmented Generation (RAG) algorithm designed to move beyond static vector databases and create an evolving knowledge graph that continually re-organizes and forms new perspectives on existing data. The heuristic excels at evolving messy and damaged data into complete, coherent, self-contained definitions.

## What this alpha emphasizes

This release deliberately biases the system toward **restoration over synthesis**:

- Every LLM prompt during evolution is wrapped in a **restoration guardrail** that instructs the model to preserve every noun, number, and relationship from the source fragments and to mark unknowns with `[unknown]` rather than guess.
- Pure `TOP`-spin queries are returned **verbatim** from the closest matching document — no LLM rewriting, no paraphrase.
- Only `LEFT`/`RIGHT`/`BOTTOM`-shaped queries pass through generative rules. These are the documented hallucination-risk paths.

If your corpus is intact, SpinRAG will mostly act as a graph-aware retriever. If your corpus is shredded, SpinRAG will try to glue it back together — and may invent. **You opt into that trade by querying with non-TOP spin.**

## Key Features

- 🧠 **Intuitive data dynamics** — data chunks carry a "spin" that drives attraction, repulsion, and transformation.
- 🌀 **Evolutionary epochs** — the knowledge graph evolves over time into denser, more self-contained definitions.
- 🤖 **Small Language Model core** — uses efficient SLMs via Ollama for spin assignment and rule processing.
- 🔗 **LangChain integration** — fits modern LLM workflows; uses `langchain-ollama` under the hood.
- 🌐 **Interactive demo** — a Dash web UI to visualize the verbose evolution process and chat with your indexed data.

## The "spin" concept

At its heart, SpinRAG treats each piece of data not as a static vector but as a particle with a spin. This spin, determined by an SLM, dictates how it interacts with other data points.

| Spin       | Icon | Description                                                              |
| :--------- | :--: | :----------------------------------------------------------------------- |
| **TOP**    |  ⬆️  | The text is a name / self-contained concept.                             |
| **BOTTOM** |  ⬇️  | The text is complex / a target for further evolution.                    |
| **LEFT**   |  ⬅️  | The text is incomplete and is missing information to be understood.      |
| **RIGHT**  |  ➡️  | The text is a definition / parameter structure.                          |

## Lifecycle

1. **🌱 Initialization** — an input string is split by newlines; each non-empty line becomes a `Document` whose initial `SpinType` is assigned by the SLM.
2. **🌀 Evolution** — for `n_epochs`, `LEFT`/`RIGHT` catalysts react with the metrically closest `TOP`/`BOTTOM` bases, producing new documents under the restoration guardrail.
3. **🧠 Embedding** — after the final epoch, every `TOP` document is embedded with the configured embedding model.
4. **🔍 Querying** — the query is assigned a spin; the closest `TOP` document is found; production rules combine them (for non-`TOP` queries) or the `TOP` document is returned verbatim (for `TOP` queries).

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

print("Initializing SpinRAG...")
rag = SpinRAG(
    content=content,
    n_epochs=5,
    llm_model="qwen3:4b-instruct-2507",
    embed_model="nomic-embed-text",
)
print("Initialization complete.")

query = "What is the main concept of the document?"
response = rag.query(query)

print("\n--- RAG Response ---")
print(response)

print("\n--- Verbose Log ---")
for log_entry in rag.get_verbose_log():
    print(log_entry)
```

### Dash demo

```bash
python demo.py
```

Open <http://127.0.0.1:8050>, upload a `.txt`, pick your models and epoch count, click **Initialize RAG**, then chat.

## Public API

```python
from spin_rag import SpinRAG, SpinType, Document
```

`SpinRAG(content, n_epochs=5, llm_model="llama2", embed_model=None, config=None, logger_callback=None)`

- `content` — raw text; one document per non-empty line.
- `n_epochs` — number of evolution epochs.
- `llm_model` — Ollama generation model.
- `embed_model` — Ollama embedding model. Defaults to `nomic-embed-text` because most chat-tuned models cannot embed.
- `logger_callback` — optional `Callable[[str], None]` for streaming logs (used by `demo.py`).

`SpinRAG.query(query_text, reorganize_graph=False) -> str` — returns the restored answer.

`SpinRAG.get_verbose_log() -> List[str]` — snapshot of the evolution/query log.

## Known limitations of this alpha

- **No persistence.** The graph lives in memory; restart = re-evolve.
- **No batching.** Each catalyst interaction is a separate LLM call; large corpora are slow.
- **No automated tests.** Behaviour is validated only by the demo flow and manual inspection.
- **Embedding model required.** Reusing a single chat model for both generation and embeddings is no longer supported by default — chat-tuned models typically reject `/api/embeddings`.
- **Hallucination on damaged input.** The restoration guardrail reduces but does not eliminate fabrication. Treat `LEFT`/`RIGHT`/`BOTTOM` query outputs as drafts.

## License

MIT.
