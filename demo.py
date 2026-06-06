"""SpinRAG Dash demo.

A small UI to:
- upload a damaged knowledge base,
- watch the evolution log in real time,
- chat against the restored TOP-spin documents.
"""

from __future__ import annotations

import base64
import threading
from typing import List

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, no_update

try:
    from langchain_ollama import OllamaLLM as _OllamaLLM
except ImportError:  # pragma: no cover
    from langchain_community.llms import Ollama as _OllamaLLM  # type: ignore

from spin_rag import SpinRAG


# --- App initialization -----------------------------------------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "SpinRAG Demo"
server = app.server


# --- Shared mutable state ---------------------------------------------------

_state_lock = threading.Lock()
_rag_lock = threading.Lock()

rag: SpinRAG | None = None
log_stream: List[str] = []
init_in_progress: bool = False

# Cap the displayed log to keep the Dash callback cheap.
_LOG_DISPLAY_LIMIT = 500


def _append_log(message: str) -> None:
    with _state_lock:
        log_stream.append(message)
        if len(log_stream) > _LOG_DISPLAY_LIMIT * 2:
            del log_stream[: len(log_stream) - _LOG_DISPLAY_LIMIT]


def _reset_log() -> None:
    with _state_lock:
        log_stream.clear()


# --- Layout -----------------------------------------------------------------

DEFAULT_MODELS = [
    "qwen3:4b-instruct-2507",
    "qwen3:1.7b",
    "llama3.1:8b",
    "llama2",
]
DEFAULT_EMBED_MODELS = [
    "nomic-embed-text",
    "mxbai-embed-large",
    "all-minilm",
]

app.layout = dbc.Container(
    fluid=True,
    children=[
        dbc.Row(
            [
                dbc.Col(
                    html.H1("🧠 SpinRAG Demo", className="text-center my-4"), width=12
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    width=4,
                    children=[
                        dbc.Card(
                            [
                                dbc.CardHeader("Controls"),
                                dbc.CardBody(
                                    [
                                        dcc.Upload(
                                            id="upload-data",
                                            children=html.Div(
                                                [
                                                    "Drag and Drop or ",
                                                    html.A("Select a Text File"),
                                                ]
                                            ),
                                            style={
                                                "width": "100%",
                                                "height": "60px",
                                                "lineHeight": "60px",
                                                "borderWidth": "1px",
                                                "borderStyle": "dashed",
                                                "borderRadius": "5px",
                                                "textAlign": "center",
                                                "margin": "10px 0",
                                            },
                                            multiple=False,
                                        ),
                                        html.Div(
                                            id="upload-status",
                                            className="small text-muted mb-2",
                                        ),
                                        dbc.Label("LLM model"),
                                        dbc.Select(
                                            id="llm-model-select",
                                            options=[
                                                {"label": m, "value": m}
                                                for m in DEFAULT_MODELS
                                            ],
                                            value=DEFAULT_MODELS[0],
                                        ),
                                        dbc.Label("Embedding model", className="mt-2"),
                                        dbc.Select(
                                            id="embed-model-select",
                                            options=[
                                                {"label": m, "value": m}
                                                for m in DEFAULT_EMBED_MODELS
                                            ],
                                            value=DEFAULT_EMBED_MODELS[0],
                                        ),
                                        dbc.Label("Epochs", className="mt-2"),
                                        dbc.Input(
                                            id="epochs-input",
                                            type="number",
                                            value=3,
                                            min=0,
                                            step=1,
                                            placeholder="Epochs",
                                        ),
                                        dbc.Button(
                                            "Initialize RAG",
                                            id="init-rag-button",
                                            color="primary",
                                            className="mt-3 w-100",
                                        ),
                                        html.Div(
                                            id="init-status", className="small mt-2"
                                        ),
                                    ]
                                ),
                            ]
                        ),
                        dbc.Card(
                            className="mt-4",
                            children=[
                                dbc.CardHeader("SpinRAG Verbose Log"),
                                dbc.CardBody(
                                    id="log-output",
                                    style={"height": "600px", "overflowY": "scroll"},
                                ),
                            ],
                        ),
                    ],
                ),
                dbc.Col(
                    width=8,
                    children=[
                        dbc.Card(
                            [
                                dbc.CardHeader("Chat"),
                                dbc.CardBody(
                                    id="chat-history",
                                    style={"height": "700px", "overflowY": "scroll"},
                                ),
                                dbc.CardFooter(
                                    dbc.InputGroup(
                                        [
                                            dbc.Input(
                                                id="chat-input",
                                                placeholder="Ask a question...",
                                                debounce=True,
                                            ),
                                            dbc.Button(
                                                "Send",
                                                id="send-button",
                                                color="success",
                                            ),
                                        ]
                                    )
                                ),
                            ]
                        )
                    ],
                ),
            ]
        ),
        dcc.Interval(id="log-updater", interval=1000, n_intervals=0),
        dcc.Store(id="chat-store", data=[]),
        dcc.Store(id="file-store", data=None),
    ],
)


# --- Callbacks --------------------------------------------------------------


@app.callback(
    Output("upload-status", "children"),
    Output("file-store", "data"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def cache_uploaded_file(contents, filename):
    if not contents:
        return "", None
    try:
        _, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string).decode("utf-8", errors="replace")
    except Exception as exc:
        return f"Could not decode '{filename}': {exc}", None
    return f"Loaded '{filename}' ({len(decoded)} chars).", decoded


@app.callback(
    Output("log-output", "children"),
    Output("init-rag-button", "disabled"),
    Output("init-status", "children"),
    Input("log-updater", "n_intervals"),
)
def update_logs(_n):
    with _state_lock:
        snapshot = list(log_stream[-_LOG_DISPLAY_LIMIT:])
        busy = init_in_progress
        ready = rag is not None
    log_children = [
        html.P(line, style={"margin": 0, "fontSize": "12px"}) for line in snapshot
    ]
    if busy:
        status = "⏳ Initializing... (this can take a while on first run)"
    elif ready:
        status = "✅ RAG ready."
    else:
        status = "ℹ️ Upload a text file and click Initialize RAG."
    return log_children, busy, status


def _run_rag_initialization(
    content_string: str,
    epochs: int,
    llm_model: str,
    embed_model: str,
) -> None:
    """Worker thread: build a fresh SpinRAG and publish it under the lock."""
    global rag, init_in_progress
    try:
        new_rag = SpinRAG(
            content=content_string,
            n_epochs=epochs,
            llm_model=llm_model,
            embed_model=embed_model,
            logger_callback=_append_log,
        )
        with _rag_lock:
            rag = new_rag
        _append_log("RAG Initialized and Ready!")
    except Exception as exc:
        _append_log(f"❌ RAG initialization failed: {exc}")
    finally:
        with _state_lock:
            init_in_progress = False


@app.callback(
    Output("init-rag-button", "n_clicks"),
    Input("init-rag-button", "n_clicks"),
    State("file-store", "data"),
    State("epochs-input", "value"),
    State("llm-model-select", "value"),
    State("embed-model-select", "value"),
    prevent_initial_call=True,
)
def initialize_rag(n_clicks, content, epochs, llm_model, embed_model):
    global init_in_progress
    if not n_clicks:
        return no_update
    if not content:
        _append_log("⚠️ Please upload a text file before initializing.")
        return no_update
    with _state_lock:
        if init_in_progress:
            _append_log("⚠️ Initialization already in progress; ignoring extra click.")
            return no_update
        init_in_progress = True
    _reset_log()
    _append_log("Starting RAG initialization from memory...")
    try:
        epochs_int = int(epochs) if epochs is not None else 3
    except (TypeError, ValueError):
        epochs_int = 3
    threading.Thread(
        target=_run_rag_initialization,
        args=(content, epochs_int, llm_model, embed_model),
        daemon=True,
    ).start()
    return n_clicks


@app.callback(
    Output("chat-store", "data"),
    Output("chat-input", "value"),
    Input("send-button", "n_clicks"),
    Input("chat-input", "n_submit"),
    State("chat-input", "value"),
    State("chat-store", "data"),
    State("llm-model-select", "value"),
    prevent_initial_call=True,
)
def update_chat(n_clicks, n_submit, user_input, chat_history, model):
    if not (n_clicks or n_submit) or not (user_input and user_input.strip()):
        return no_update, no_update

    chat_history = list(chat_history or [])
    chat_history.append({"user": user_input.strip()})

    with _rag_lock:
        active_rag = rag

    if active_rag is None:
        chat_history.append({"bot": "Please initialize the RAG system first."})
        return chat_history, ""

    try:
        rag_response = active_rag.query(user_input.strip())
    except Exception as exc:
        chat_history.append({"bot": f"RAG error: {exc}"})
        return chat_history, ""

    try:
        llm = _OllamaLLM(model=model)
        prompt = (
            "You are answering with the help of a restored knowledge-base "
            "fragment. Stay faithful to the context; if the answer is not in "
            "the context, say you don't know.\n\n"
            f"Context: {rag_response}\n\n"
            f"Question: {user_input.strip()}"
        )
        llm_response = llm.invoke(prompt)
    except Exception as exc:
        llm_response = f"LLM error: {exc}\n\nRetrieved context:\n{rag_response}"

    chat_history.append({"bot": llm_response})
    return chat_history, ""


@app.callback(
    Output("chat-history", "children"),
    Input("chat-store", "data"),
)
def display_chat(chat_history):
    history = []
    for msg in chat_history or []:
        if "user" in msg:
            history.append(dbc.Alert(msg["user"], color="info", className="text-right"))
        else:
            history.append(dbc.Alert(msg["bot"], color="light"))
    return history


if __name__ == "__main__":
    app.run(debug=True)
