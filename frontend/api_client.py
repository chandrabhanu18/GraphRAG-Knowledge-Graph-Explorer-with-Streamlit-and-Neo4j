import os
import logging
from typing import Dict, Any
import requests

logger = logging.getLogger("graphrag.frontend.api_client")
logging.basicConfig(level=logging.INFO)

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000").rstrip("/")
if BACKEND_URL == "http://backend:8000" and (os.name == "nt" or not os.path.exists("/.dockerenv")):
    BACKEND_URL = "http://127.0.0.1:8000"


def ingest_pdfs(pdf_dir: str) -> Dict[str, Any]:
    """
    Call POST /ingest on the backend to process PDF files in pdf_dir.
    
    Args:
        pdf_dir: Directory path inside container containing PDFs (e.g. /data/pdfs/).
        
    Returns:
        Dict with status, nodes_created, edges_created, files_processed.
    """
    url = f"{BACKEND_URL}/ingest"
    payload = {"pdf_dir": pdf_dir}
    try:
        # Generous timeout for PDF extraction, embeddings, and LLM processing
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to call POST {url}: {e}")
        raise RuntimeError(f"Backend ingestion error: {e}")


def fetch_graph(limit: int = 50) -> Dict[str, Any]:
    """
    Call GET /graph on the backend to retrieve the top connected nodes and edges.
    
    Args:
        limit: Max number of nodes to return (default: 50).
        
    Returns:
        Dict with 'nodes' and 'edges'.
    """
    url = f"{BACKEND_URL}/graph"
    params = {"limit": limit}
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to call GET {url}: {e}")
        raise RuntimeError(f"Backend graph fetch error: {e}")


def query_graphrag(question: str) -> Dict[str, Any]:
    """
    Call POST /graphrag on the backend to run question answering over the knowledge graph.
    
    Args:
        question: User query string.
        
    Returns:
        Dict with 'question', 'anchor_nodes', 'subgraph_context', and 'answer'.
    """
    url = f"{BACKEND_URL}/graphrag"
    payload = {"question": question}
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to call POST {url}: {e}")
        raise RuntimeError(f"Backend GraphRAG query error: {e}")
