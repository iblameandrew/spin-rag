import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from spin_rag.spin_rag import SpinRAG
from langchain_community.llms import Ollama
import threading
import base64
import io

# --- App Initialization ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server

# --- Global Variables ---
rag = None
log_stream = []

# --- Layout ---
app.layout = dbc.Container(fluid=True, children=[
    dbc.Row([
        dbc.Col(html.H1("🧠 SpinRAG Demo", className="text-center my-4"), width=12)
    ]),
    
    dbc.Row([
        # Left Panel: Controls and Logs
        dbc.Col(width=4, children=[
            dbc.Card([
                dbc.CardHeader("Controls"),
                dbc.CardBody([
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div(['Drag and Drop or ', html.A('Select a Text File')]),
                        style={
                            'width': '100%', 'height': '60px', 'lineHeight': '60px',
                            'borderWidth': '1px', 'borderStyle': 'dashed',
                            'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px 0'
                        },
                        multiple=False
                    ),
                    dbc.Select(id="llm-model-select", options=[{"label": m, "value": m} for m in ["qwen3:1.7b", "qwen3:4b-instruct-2507-q4_K_M", "hf.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Qwen3-30B-A3B-Instruct-2507-UD-IQ1_M.gguf"]], value="qwen3:1.7b"),
                    dbc.Input(id="epochs-input", type="number", value=3, placeholder="Epochs"),
                    dbc.Button("Initialize RAG", id="init-rag-button", color="primary", className="mt-2 w-100"),
                ])
            ]),
            
            dbc.Card(className="mt-4", children=[
                dbc.CardHeader("SpinRAG Verbose Log"),
                dbc.CardBody(id="log-output", style={"height": "600px", "overflowY": "scroll"})
            ])
        ]),

        # Right Panel: Chat Interface
        dbc.Col(width=8, children=[
            dbc.Card([
                dbc.CardHeader("Chat"),
                dbc.CardBody(id="chat-history", style={"height": "700px", "overflowY": "scroll"}),
                dbc.CardFooter(
                    dbc.InputGroup([
                        dbc.Input(id="chat-input", placeholder="Ask a question..."),
                        dbc.Button("Send", id="send-button", color="success")
                    ])
                )
            ])
        ])
    ]),
    dcc.Interval(id='log-updater', interval=1*1000, n_intervals=0),
    dcc.Store(id='chat-store', data=[])
])

# --- Callbacks ---
@app.callback(
    Output("log-output", "children"),
    Input("log-updater", "n_intervals")
)
def update_logs(n):
    global log_stream
    return [html.P(log, style={'margin': 0, 'fontSize': '12px'}) for log in log_stream]

def run_rag_initialization(content_string, epochs, model):
    """Initializes the RAG system and streams logs to the global log_stream."""
    global rag, log_stream
    
    # This function will be called by the SpinRAG instance to log messages
    def stream_log_to_dashboard(message: str):
        log_stream.append(message)

    stream_log_to_dashboard("Starting RAG initialization from memory...")
    
    # Pass the logger function to the SpinRAG constructor
    rag = SpinRAG(content=content_string, n_epochs=epochs, llm_model=model, logger_callback=stream_log_to_dashboard)
    
    # The get_verbose_log and clear_log calls are no longer needed here
    # as logs are streamed in real-time.
    
    stream_log_to_dashboard("RAG Initialized and Ready!")

@app.callback(
    Output("init-rag-button", "disabled"),
    Input("init-rag-button", "n_clicks"),
    State("upload-data", "contents"),
    State("epochs-input", "value"),
    State("llm-model-select", "value"),
    prevent_initial_call=True
)
def initialize_rag(n_clicks, contents, epochs, model):
    if n_clicks and contents:
        # Decode the base64 string from the upload component
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        text_content = decoded.decode('utf-8')
        
        # Start the initialization in a separate thread to not block the UI
        threading.Thread(target=run_rag_initialization, args=(text_content, epochs, model)).start()
        return True
    return False

@app.callback(
    Output('chat-store', 'data'),
    Input('send-button', 'n_clicks'),
    State('chat-input', 'value'),
    State('chat-store', 'data'),
    State('llm-model-select', 'value'),
    prevent_initial_call=True
)
def update_chat(n_clicks, user_input, chat_history, model):
    global rag, log_stream
    if n_clicks and user_input:
        chat_history.append({"user": user_input})
        
        if rag:
            # The rag object already has the logger configured, so rag.query()
            # will automatically stream its logs to the dashboard.
            rag_response = rag.query(user_input)
            
            # These are no longer necessary because logs are streamed via the callback.
            # log_stream.extend(rag.get_verbose_log())
            # rag.clear_log()
            
            llm = Ollama(model=model)
            prompt = f"Using the following context, answer the user's question.\n\nContext: {rag_response}\n\nQuestion: {user_input}"
            llm_response = llm.predict(prompt)
            chat_history.append({"bot": llm_response})
        else:
            chat_history.append({"bot": "Please initialize the RAG system first."})
            
    return chat_history

@app.callback(
    Output('chat-history', 'children'),
    Input('chat-store', 'data')
)
def display_chat(chat_history):
    history = []
    for msg in chat_history:
        if "user" in msg:
            history.append(dbc.Alert(msg["user"], color="info", className="text-right"))
        else:
            history.append(dbc.Alert(msg["bot"], color="light"))
    return history

if __name__ == '__main__':
    app.run(debug=True)