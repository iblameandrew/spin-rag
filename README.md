<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/873489e9-28cc-404d-b9d0-231bf82dcc4a" />

**SpinRAG** is a Retrieval-Augmented Generation (RAG) algorithm designed to move beyond static vector databases and create an evolving knowledge graph that continually re-organizes and forms new perspectives on existing data - the heuristic works really well evolving messy and damaged data into complete coherent self contained definitions.

## ✨ Key Features

-   🧠 **Intuitive Data Dynamics**: Models data chunks with "spin" values, creating a system of attraction, repulsion, and transformation.
-   🌀 **Evolutionary Epochs**: The knowledge graph evolves over time, creating denser and more nuanced data relationships.
-   🤖 **Small Language Model Core**: Uses efficient SLMs (via Ollama) for dynamic spin assignment and rule processing.
-   🔗 **LangChain Integration**: Seamlessly fits into modern LLM workflows.
-   🌐 **Interactive Demo**: A Dash-based web UI to visualize the algorithm's verbose processes and chat with your indexed data.

## 💡 The "Spin" Concept Explained

At its heart, SpinRAG treats each piece of data not as a static vector, but as a particle with a "spin." This spin, determined by an SLM, dictates how it interacts with other data points.

#### The Four Spin Types

| Spin   | Icon | Description                                                               |
| :----- | :--: | :------------------------------------------------------------------------ |
| **TOP**    | ⬆️  | The text is a name.                                                       |
| **BOTTOM** | ⬇️  | The text is complex.                                                      |
| **LEFT**   | ⬅️  | The text is incomplete and is missing some information to be understood.  |
| **RIGHT**  | ➡️  | The text is a definition.                                                 |


## ⚙️ How It Works: The Lifecycle

1.  **🌱 Initialization**: An input **string** is processed. Instead of complex chunking, the string is split by newlines, treating each line as a distinct document. Each document is then analyzed by an SLM (e.g., Llama 2) to assign an initial `SpinType`.
2.  **🌀 Evolution**: For a set number of `n_epochs`, the production rules are applied across all documents. This dynamic process generates new documents from the interactions between existing ones, building out the knowledge graph.
3.  **🧠 Embedding**: After the final epoch, all documents with a `TOP` spin are converted into vector embeddings for fast retrieval.
4.  **🔍 Querying**: A user's query is also assigned a spin. The system finds the metrically closest `TOP` document and applies production rules between the query and the retrieved document to generate a contextually rich answer.


## 🚀 Getting Started

### Prerequisites

-   Python 3.8+
-   [Ollama](https://ollama.ai/) installed and running with a model pulled.
    ```bash
    ollama pull qwen3:4b-instruct-2507
    ```

### 1. Installation

Clone the repository and install the required dependencies.

```bash
# Clone the repo
git clone https://github.com/iblameandrew/spin-rag.git
cd spin-rag

# Install the package and its dependencies
pip install - r requirements.txt


### 2. API Usage (Programmatic)

Use `spinlm` directly in your Python projects for a powerful, dynamic RAG backend.
```

```python
from spinlm import SpinRAG

# 1. Initialize the RAG with your data, epochs, and a running Ollama model
print("Initializing SpinRAG...")
rag = SpinRAG(
    file_path="path/to/your/knowledge_base.txt",
    n_epochs=5,
    llm_model="llama2" # Make sure this model is available in Ollama
)
print("Initialization Complete!")

# 2. Query the evolved knowledge graph
query = "What is the main concept of the document?"
response = rag.query(query)

print("\n--- RAG Response ---")
print(response)

# 3. View the verbose log of the process
print("\n--- Verbose Log ---")
for log_entry in rag.get_verbose_log():
    print(log_entry)```


```


