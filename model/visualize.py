import re
import numpy as np
import pandas as pd
import requests
import torch
import networkx as nx
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State
from transformers import AutoTokenizer, AutoModel

from sae import SparseAutoencoder, INPUT_DIM, DICT_SIZE, DEVICE

SPECTER_MODEL = "allenai/specter2_base"
QUERY_TOP_K = 100

OLLAMA_MODEL = "deepseek-r1:latest"
OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
TOP_K_LABEL = 10

EMBEDDINGS_NPY = "../data/embeddings.npy"
EMBEDDING_IDS_NPY = "../data/embedding_ids.npy"
METADATA_CSV = "../data/oc_mini_node_metadata.csv"
EDGELIST_CSV = "../data/oc_mini_edgelist.csv"
CHECKPOINT_PATH = "../data/sae_checkpoint.pt"

MAX_NODES = 100


# ── data loading ────────────────────────────────────────────────────────────

def load_model_and_acts():
    model = SparseAutoencoder(INPUT_DIM, DICT_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    raw = np.load(EMBEDDINGS_NPY).astype(np.float32)
    x = torch.from_numpy(raw)
    x = x - x.mean(dim=0)
    x = x / x.norm(dim=1, keepdim=True).mean()
    with torch.no_grad():
        _, acts = model(x.to(DEVICE))
    return acts.cpu().numpy()


def load_graph_and_meta():
    meta = pd.read_csv(METADATA_CSV).set_index("id")
    edges = pd.read_csv(EDGELIST_CSV)
    G = nx.from_pandas_edgelist(edges, source="source", target="target",
                                create_using=nx.DiGraph())
    return G, meta


# ── feature helpers ──────────────────────────────────────────────────────────

def nodes_for_feature(feature_idx, acts, paper_ids):
    col = acts[:, feature_idx]
    order = np.argsort(col)[::-1]
    active = order[col[order] > 0][:MAX_NODES]
    return [paper_ids[i] for i in active], col


def subgraph_for_nodes(G, node_ids):
    present = [n for n in node_ids if G.has_node(n)]
    neighbors = set(present)
    for n in present:
        neighbors.update(G.predecessors(n))
        neighbors.update(G.successors(n))
    return G.subgraph(list(neighbors)[:MAX_NODES * 3]).copy()


def label_feature(feature_idx, acts, paper_ids, meta):
    col = acts[:, feature_idx]
    order = np.argsort(col)[::-1]
    top_indices = order[col[order] > 0][:TOP_K_LABEL]
    snippets = []
    for i in top_indices:
        pid = paper_ids[i]
        if pid in meta.index:
            row = meta.loc[pid]
            snippets.append(f"{row['title']}: {str(row.get('abstract', ''))}"[:400])
    if not snippets:
        return "unknown", ""
    numbered = "\n\n".join(f"{i+1}. {s}" for i, s in enumerate(snippets))
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{
            "role": "user",
            "content": (
                "Below are titles and abstracts of scientific papers that all "
                "strongly activate the same learned feature in a sparse autoencoder.\n\n"
                f"{numbered}\n\n"
                "In one short phrase (5 words or fewer), what research topic or theme "
                "do these papers share? Reply with only the phrase, no explanation."
            ),
        }],
        "temperature": 0.2,
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    raw_text = resp.json()["choices"][0]["message"]["content"]

    # Extract <think>...</think> reasoning if present
    think_match = re.search(r"<think>(.*?)</think>", raw_text, re.DOTALL)
    reasoning = think_match.group(1).strip() if think_match else ""
    label = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    return label, reasoning


def build_scatter(feature_idx, acts, paper_ids, G, meta):
    col = acts[:, feature_idx]
    act_map = {paper_ids[i]: float(col[i]) for i in range(len(paper_ids))}

    node_acts, active_neighbor_counts, hover = [], [], []
    for pid, activation in act_map.items():
        if activation <= 0 or not G.has_node(pid):
            continue
        neighbors = list(G.predecessors(pid)) + list(G.successors(pid))
        n_active_neighbors = sum(1 for n in neighbors if act_map.get(n, 0) > 0)
        node_acts.append(activation)
        active_neighbor_counts.append(n_active_neighbors)
        title = ""
        if pid in meta.index:
            t = meta.loc[pid, "title"]
            title = t[:80] + "…" if len(t) > 80 else t
        hover.append(f"<b>{pid}</b><br>{title}<br>activation: {activation:.3f}<br>active neighbors: {n_active_neighbors}")

    fig = go.Figure(
        data=go.Scatter(
            x=node_acts,
            y=active_neighbor_counts,
            mode="markers",
            marker=dict(size=7, color=node_acts, colorscale="Viridis",
                        showscale=True, colorbar=dict(title="Activation"),
                        line=dict(width=0.5, color="#333")),
            text=hover,
            hoverinfo="text",
        ),
        layout=go.Layout(
            title="Activation vs. active neighbors",
            xaxis=dict(title="Node activation"),
            yaxis=dict(title="# active neighbors"),
            margin=dict(l=50, r=10, t=40, b=40),
            hovermode="closest",
        ),
    )
    return fig


def embed_query(text, tokenizer, encoder):
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=512, padding=True).to(DEVICE)
    with torch.no_grad():
        out = encoder(**inputs)
    vec = out.last_hidden_state[:, 0, :].cpu().float().numpy()
    return vec[0]


def build_query_histogram(query, embeddings, acts, paper_ids, meta, tokenizer, encoder):
    # Embed query and find top-K similar papers by cosine similarity
    q_vec = embed_query(query, tokenizer, encoder)
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-8)
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    sims = emb_norm @ q_norm
    top_k_idx = np.argsort(sims)[::-1][:QUERY_TOP_K]

    # Aggregate SAE activations across top-K papers (mean over active only)
    top_acts = acts[top_k_idx]                      # [K, DICT_SIZE]
    mean_acts = top_acts.mean(axis=0)               # [DICT_SIZE]

    top20_idx = np.argsort(mean_acts)[::-1][:20]
    active_idx = top20_idx
    active_vals = mean_acts[active_idx]

    fig = go.Figure(
        data=go.Scatter(
            x=active_idx.tolist(),
            y=active_vals.tolist(),
            mode="markers",
            marker=dict(
                symbol="line-ns",
                size=12,
                color=active_vals.tolist(),
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Activation"),
                line=dict(width=1.5, color=active_vals.tolist(), colorscale="Viridis"),
            ),
            hovertext=[f"Feature {i}<br>mean activation: {mean_acts[i]:.4f}"
                       for i in active_idx],
            hoverinfo="text",
        ),
        layout=go.Layout(
            title=f'Feature activations for: "{query[:60]}"',
            xaxis=dict(title="Feature index", range=[-1, DICT_SIZE]),
            yaxis=dict(title="Mean activation (top-100 papers)", rangemode="tozero"),
            margin=dict(l=50, r=10, t=50, b=40),
            transition=dict(duration=400, easing="cubic-in-out"),
        ),
    )

    # Build results table for the top-K papers
    rows = []
    for i in top_k_idx[:10]:
        pid = paper_ids[i]
        sim = float(sims[i])
        title = meta.loc[pid, "title"] if pid in meta.index else str(pid)
        rows.append(html.Tr([
            html.Td(f"{sim:.3f}", style={"padding": "4px 8px", "color": "#666"}),
            html.Td(title[:100], style={"padding": "4px 8px", "fontSize": "12px"}),
        ]))

    table = html.Table(
        [html.Tr([html.Th("Sim", style={"padding": "4px 8px"}),
                  html.Th("Title", style={"padding": "4px 8px"})])] + rows,
        style={"borderCollapse": "collapse", "width": "100%",
               "border": "1px solid #ddd", "fontSize": "12px"},
    )
    return fig, table


def build_figure(feature_idx, acts, paper_ids, G, meta, label):
    active_ids, act_col = nodes_for_feature(feature_idx, acts, paper_ids)
    if not active_ids:
        return go.Figure(), []

    sub = subgraph_for_nodes(G, active_ids)
    if not sub.nodes:
        return go.Figure(), []

    pos = nx.spring_layout(sub, seed=42, k=0.5)
    active_set = set(active_ids)
    act_map = {paper_ids[i]: act_col[i] for i in range(len(paper_ids))}

    edge_x, edge_y = [], []
    for u, v in sub.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

    node_ids_list = list(sub.nodes())
    node_x = [pos[n][0] for n in node_ids_list]
    node_y = [pos[n][1] for n in node_ids_list]
    node_colors = [act_map.get(n, 0.0) for n in node_ids_list]
    node_sizes = [12 if n in active_set else 6 for n in node_ids_list]

    def node_label(n):
        if n in meta.index:
            title = meta.loc[n, "title"]
            title = title[:80] + "…" if len(title) > 80 else title
            return f"<b>{n}</b><br>{title}<br>activation: {act_map.get(n, 0):.3f}"
        return str(n)

    fig = go.Figure(
        data=[
            go.Scatter(x=edge_x, y=edge_y, mode="lines",
                       line=dict(width=0.5, color="#888"), hoverinfo="none"),
            go.Scatter(x=node_x, y=node_y, mode="markers",
                       hoverinfo="text",
                       text=[node_label(n) for n in node_ids_list],
                       marker=dict(showscale=True, colorscale="Viridis",
                                   color=node_colors, size=node_sizes,
                                   colorbar=dict(title="Activation"),
                                   line=dict(width=0.5, color="#333"))),
        ],
        layout=go.Layout(
            title=f'Feature {feature_idx}: "{label}" — {len(active_ids)} active nodes',
            showlegend=False, hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    return fig


# ── Dash app ─────────────────────────────────────────────────────────────────

print("Loading model and computing activations...")
ACTS = load_model_and_acts()
EMBEDDINGS = np.load(EMBEDDINGS_NPY).astype(np.float32)
EMBEDDINGS = EMBEDDINGS - EMBEDDINGS.mean(axis=0)
EMBEDDINGS = EMBEDDINGS / np.linalg.norm(EMBEDDINGS, axis=1, keepdims=True).mean()
PAPER_IDS = np.load(EMBEDDING_IDS_NPY).tolist()
print("Loading graph and metadata...")
G, META = load_graph_and_meta()
print("Loading SPECTER2 for query embedding...")
TOKENIZER = AutoTokenizer.from_pretrained(SPECTER_MODEL)
ENCODER = AutoModel.from_pretrained(SPECTER_MODEL).to(DEVICE).eval()
print(f"Ready. {DICT_SIZE} features, {len(PAPER_IDS)} papers, {G.number_of_edges()} edges.")

app = dash.Dash(__name__)
app.layout = html.Div(style={"fontFamily": "sans-serif", "display": "flex",
                              "flexDirection": "column", "height": "100vh",
                              "padding": "12px", "boxSizing": "border-box"}, children=[
    html.H2("SAE Feature Explorer", style={"margin": "0 0 8px 0"}),
    dcc.Tabs(id="tabs", value="feature-tab", children=[

        dcc.Tab(label="Feature Explorer", value="feature-tab", children=[
            html.Div(style={"display": "flex", "gap": "8px", "margin": "8px 0"}, children=[
                dcc.Input(id="feature-input", type="number", min=0, max=DICT_SIZE - 1,
                          placeholder=f"Feature index (0–{DICT_SIZE - 1})",
                          debounce=True,
                          style={"width": "260px", "padding": "6px", "fontSize": "14px"}),
                html.Button("Explore", id="explore-btn", n_clicks=0,
                            style={"padding": "6px 16px", "fontSize": "14px"}),
                html.Span(id="status-text", style={"alignSelf": "center", "color": "#666",
                                                   "fontSize": "13px"}),
            ]),
            html.Div(style={"display": "flex", "flex": "1", "gap": "12px",
                            "height": "calc(100vh - 140px)"}, children=[
                dcc.Graph(id="graph", style={"flex": "2", "height": "100%"},
                          config={"displayModeBar": False}),
                html.Div(style={"flex": "1", "display": "flex", "flexDirection": "column",
                                "gap": "8px", "overflowY": "auto"}, children=[
                    html.Div(id="label-box", style={
                        "background": "#f0f4ff", "border": "1px solid #bcd",
                        "borderRadius": "6px", "padding": "10px", "fontSize": "15px",
                        "fontWeight": "bold",
                    }),
                    dcc.Graph(id="scatter", config={"displayModeBar": False},
                              style={"minHeight": "260px"}),
                    html.Details([
                        html.Summary("Model reasoning", style={"cursor": "pointer",
                                                                "fontSize": "13px",
                                                                "color": "#555"}),
                        html.Pre(id="reasoning-box", style={
                            "whiteSpace": "pre-wrap", "fontSize": "12px",
                            "background": "#fafafa", "border": "1px solid #ddd",
                            "borderRadius": "4px", "padding": "8px",
                            "maxHeight": "300px", "overflowY": "auto",
                        }),
                    ], open=False),
                ]),
            ]),
        ]),

        dcc.Tab(label="Query Search", value="query-tab", children=[
            html.Div(style={"display": "flex", "gap": "8px", "margin": "8px 0"}, children=[
                dcc.Input(id="query-input", type="text", debounce=True,
                          placeholder="Enter a research query...",
                          style={"flex": "1", "padding": "6px", "fontSize": "14px"}),
                html.Button("Search", id="query-btn", n_clicks=0,
                            style={"padding": "6px 16px", "fontSize": "14px"}),
                html.Span(id="query-status", style={"alignSelf": "center", "color": "#666",
                                                    "fontSize": "13px"}),
            ]),
            html.Div(style={"display": "flex", "gap": "12px",
                            "height": "calc(100vh - 140px)"}, children=[
                dcc.Graph(id="query-hist", style={"flex": "2", "height": "100%"},
                          config={"displayModeBar": False}),
                html.Div(id="query-table", style={
                    "flex": "1", "overflowY": "auto", "fontSize": "12px",
                }),
            ]),
        ]),

    ]),
])


@app.callback(
    Output("graph", "figure"),
    Output("scatter", "figure"),
    Output("label-box", "children"),
    Output("reasoning-box", "children"),
    Output("status-text", "children"),
    Input("explore-btn", "n_clicks"),
    State("feature-input", "value"),
    prevent_initial_call=True,
)
def explore_feature(_, feature_idx):
    if feature_idx is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, "Enter a feature index."
    feature_idx = int(feature_idx)
    label, reasoning = label_feature(feature_idx, ACTS, PAPER_IDS, META)
    network_fig = build_figure(feature_idx, ACTS, PAPER_IDS, G, META, label)
    scatter_fig = build_scatter(feature_idx, ACTS, PAPER_IDS, G, META)
    n_active = int((ACTS[:, feature_idx] > 0).sum())
    status = f"{n_active} active papers"
    return network_fig, scatter_fig, f'Feature {feature_idx}: "{label}"', reasoning, status


@app.callback(
    Output("query-hist", "figure"),
    Output("query-table", "children"),
    Output("query-status", "children"),
    Input("query-btn", "n_clicks"),
    State("query-input", "value"),
    prevent_initial_call=True,
)
def search_query(_, query):
    if not query or not query.strip():
        return dash.no_update, dash.no_update, "Enter a query."
    hist_fig, table = build_query_histogram(
        query.strip(), EMBEDDINGS, ACTS, PAPER_IDS, META, TOKENIZER, ENCODER
    )
    return hist_fig, table, f"Top {QUERY_TOP_K} papers retrieved"


if __name__ == "__main__":
    app.run(debug=False, port=8050)
