import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("backend"))
from main import app

client = TestClient(app)


def test_full_e2e_ingest_graph_and_rag_pipeline():
    """
    Test the full end-to-end workflow:
    1. Ingest actual PDF document from data/pdfs/
    2. Retrieve subgraph and verify nodes and edges
    3. Run GraphRAG query and verify anchors, context, and answer
    """
    # 1. Ingest
    pdf_dir = "data/pdfs/"
    assert os.path.isdir(pdf_dir)
    assert any(f.endswith(".pdf") for f in os.listdir(pdf_dir))

    ingest_res = client.post("/ingest", json={"pdf_dir": pdf_dir})
    assert ingest_res.status_code == 200
    ingest_data = ingest_res.json()

    assert ingest_data["status"] == "success"
    assert ingest_data["files_processed"] >= 1
    assert ingest_data["nodes_created"] > 0
    assert ingest_data["edges_created"] > 0

    # 2. Graph Retrieval
    graph_res = client.get("/graph?limit=50")
    assert graph_res.status_code == 200
    graph_data = graph_res.json()

    assert len(graph_data["nodes"]) > 0
    assert len(graph_data["nodes"]) <= 50
    assert len(graph_data["edges"]) > 0

    for node in graph_data["nodes"]:
        assert "id" in node
        assert "name" in node
        assert "type" in node
        assert "description" in node

    for edge in graph_data["edges"]:
        assert "source" in edge
        assert "target" in edge
        assert "relation" in edge

    # 3. GraphRAG Retrieval & QA
    rag_res = client.post("/graphrag", json={"question": "What is backpropagation and how does it relate to Gradient Descent?"})
    assert rag_res.status_code == 200
    rag_data = rag_res.json()

    assert "question" in rag_data
    assert "anchor_nodes" in rag_data
    assert "subgraph_context" in rag_data
    assert "answer" in rag_data

    assert len(rag_data["anchor_nodes"]) > 0
    assert len(rag_data["answer"]) > 0
    assert len(rag_data["subgraph_context"]) > 0
