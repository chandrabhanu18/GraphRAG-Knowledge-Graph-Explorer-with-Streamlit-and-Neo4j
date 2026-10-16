import os
import sys
import json
import logging
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

# Ensure frontend directory is in sys.path for robust resolution in all execution environments
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from api_client import ingest_pdfs, fetch_graph, query_graphrag

logger = logging.getLogger("graphrag.frontend")
logging.basicConfig(level=logging.INFO)

# Page configuration
st.set_page_config(
    page_title="GraphRAG Knowledge Graph Explorer",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3B82F6, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# Node type color mapping
NODE_TYPE_COLORS = {
    "Algorithm": "#3B82F6",    # Blue
    "Concept": "#10B981",      # Emerald Green
    "Technology": "#8B5CF6",   # Violet
    "Framework": "#F59E0B",    # Amber
    "Library": "#EC4899",      # Pink
    "Dataset": "#06B6D4",      # Cyan
    "Metric": "#EF4444",       # Red
    "Person": "#F97316",       # Orange
    "Organization": "#6366F1", # Indigo
    "Entity": "#64748B",       # Slate
}

# Determine data paths
DATA_PDF_DIR = "/data/pdfs"
if not os.path.exists(DATA_PDF_DIR):
    local_data_dir = os.path.join(os.getcwd(), "data", "pdfs")
    if os.path.exists(local_data_dir):
        DATA_PDF_DIR = local_data_dir
    else:
        try:
            os.makedirs(DATA_PDF_DIR, exist_ok=True)
        except Exception:
            DATA_PDF_DIR = "./data/pdfs"
            os.makedirs(DATA_PDF_DIR, exist_ok=True)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "graph_built" not in st.session_state:
    # Check if backend already has graph populated
    try:
        initial_check = fetch_graph(limit=1)
        if initial_check.get("nodes") and len(initial_check["nodes"]) > 0:
            st.session_state["graph_built"] = True
        else:
            st.session_state["graph_built"] = False
    except Exception:
        st.session_state["graph_built"] = False


# ==========================================
# Sidebar: PDF Upload & Ingestion
# ==========================================
with st.sidebar:
    st.title("📁 Document Ingestion")
    st.markdown("Upload domain documents to extract entities and generate the knowledge graph.")

    uploaded_files = st.file_uploader(
        label="Upload PDF Documents",
        accept_multiple_files=True,
        type=["pdf"]
    )

    build_clicked = st.button(label="Build Graph")

    if build_clicked:
        if uploaded_files:
            # Ensure target directory exists
            target_save_dir = "/data/pdfs"
            try:
                os.makedirs(target_save_dir, exist_ok=True)
            except Exception:
                target_save_dir = os.path.join(os.getcwd(), "data", "pdfs")
                os.makedirs(target_save_dir, exist_ok=True)

            for file in uploaded_files:
                save_path = os.path.join(target_save_dir, file.name)
                with open(save_path, "wb") as f:
                    f.write(file.getbuffer())

            with st.spinner("Building knowledge graph... this may take a minute."):
                try:
                    # Ingest using container path /data/pdfs/
                    ingest_path = "/data/pdfs/"
                    try:
                        res = ingest_pdfs(ingest_path)
                    except Exception:
                        # Fallback for direct local run
                        res = ingest_pdfs(target_save_dir)

                    st.session_state["graph_built"] = True
                    st.success(
                        f"Knowledge graph built successfully! "
                        f"Processed {res.get('files_processed', 0)} files, "
                        f"created {res.get('nodes_created', 0)} nodes and "
                        f"{res.get('edges_created', 0)} edges."
                    )
                except Exception as e:
                    st.error(f"Failed to build graph: {e}")
        else:
            # Check if directory already has PDFs
            existing_pdfs = [
                f for f in os.listdir(DATA_PDF_DIR)
                if f.endswith(".pdf")
            ] if os.path.exists(DATA_PDF_DIR) else []

            if existing_pdfs:
                with st.spinner("Building knowledge graph... this may take a minute."):
                    try:
                        res = ingest_pdfs("/data/pdfs/")
                        st.session_state["graph_built"] = True
                        st.success(
                            f"Knowledge graph built successfully! "
                            f"Processed {res.get('files_processed', 0)} files, "
                            f"created {res.get('nodes_created', 0)} nodes and "
                            f"{res.get('edges_created', 0)} edges."
                        )
                    except Exception:
                        try:
                            res = ingest_pdfs(DATA_PDF_DIR)
                            st.session_state["graph_built"] = True
                            st.success(
                                f"Knowledge graph built successfully! "
                                f"Processed {res.get('files_processed', 0)} files, "
                                f"created {res.get('nodes_created', 0)} nodes and "
                                f"{res.get('edges_created', 0)} edges."
                            )
                        except Exception as e:
                            st.error(f"Failed to build graph: {e}")
            else:
                st.warning("Please upload at least one PDF document before building the graph.")


# ==========================================
# Main Content: Tabs
# ==========================================
st.markdown('<div class="main-header">GraphRAG Knowledge Graph Explorer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Interactive Knowledge Graph Visualizer & Vector-Augmented QA powered by Neo4j and LLMs</div>', unsafe_allow_html=True)

tab_graph, tab_chat = st.tabs(["Knowledge Graph", "Chat with Graph"])


# ------------------------------------------
# Tab 1: Knowledge Graph
# ------------------------------------------
with tab_graph:
    if not st.session_state["graph_built"]:
        st.info("Upload PDFs and click 'Build Graph' in the sidebar to populate the knowledge graph.")
    else:
        try:
            graph_data = fetch_graph(limit=50)
            nodes = graph_data.get("nodes", [])
            edges = graph_data.get("edges", [])

            if not nodes:
                st.warning("The graph database currently has no nodes. Upload documents and click 'Build Graph'.")
            else:
                # Initialize PyVis Network
                net = Network(height="600px", width="100%", directed=True, bgcolor="#0f172a", font_color="#f8fafc")
                
                # Configure physics: Barnes-Hut
                net.barnes_hut(
                    gravity=-3000,
                    central_gravity=0.3,
                    spring_length=120,
                    spring_strength=0.05,
                    damping=0.09
                )

                # Add nodes
                for node in nodes:
                    nid = node["id"]
                    nname = node["name"]
                    ntype = node.get("type", "Concept")
                    ndesc = node.get("description", "")
                    color = NODE_TYPE_COLORS.get(ntype, "#94A3B8")
                    
                    tooltip = f"<b>{nname}</b> ({ntype})<br>{ndesc}"
                    net.add_node(
                        nid,
                        label=nname,
                        title=tooltip,
                        color=color,
                        size=22,
                        shape="dot",
                        borderWidth=2
                    )

                # Add edges
                for edge in edges:
                    net.add_edge(
                        edge["source"],
                        edge["target"],
                        label=edge.get("relation", ""),
                        title=edge.get("relation", ""),
                        color="#64748b",
                        arrows="to"
                    )

                # Render and save HTML to /data/graph_render.html
                render_path = "/data/graph_render.html"
                try:
                    net.save_graph(render_path)
                except Exception:
                    local_render_dir = os.path.join(os.getcwd(), "data")
                    os.makedirs(local_render_dir, exist_ok=True)
                    render_path = os.path.join(local_render_dir, "graph_render.html")
                    net.save_graph(render_path)

                with open(render_path, "r", encoding="utf-8") as f:
                    html_content = f.read()

                components.html(html_content, height=620, scrolling=True)

                # Metrics display below graph
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Nodes", len(nodes))
                col2.metric("Total Edges", len(edges))
                unique_types = set(n.get("type", "Concept") for n in nodes)
                col3.metric("Node Types", len(unique_types))

        except Exception as e:
            st.error(f"Error loading knowledge graph: {e}")


# ------------------------------------------
# Tab 2: Chat with Graph
# ------------------------------------------
with tab_chat:
    # Display conversation history
    for entry in st.session_state["chat_history"]:
        with st.chat_message("user"):
            st.write(entry.get("question", ""))
        
        with st.chat_message("assistant"):
            st.write(entry.get("answer", ""))
            with st.expander("View reasoning"):
                st.markdown("**Anchor Nodes:**")
                st.json(entry.get("anchor_nodes", []))
                st.markdown("**Subgraph Context:**")
                st.code(entry.get("subgraph_context", ""), language="text")

    # Chat input
    user_question = st.chat_input("Ask a question about your documents...")

    if user_question:
        with st.spinner("Searching the knowledge graph..."):
            try:
                response = query_graphrag(user_question)
                st.session_state["chat_history"].append(response)
                st.rerun()
            except Exception as e:
                st.error(f"Error querying GraphRAG: {e}")
