import os
import sys
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

# Add backend to sys.path
sys.path.insert(0, os.path.abspath("backend"))
from main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify GET /health returns 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_endpoint():
    """Verify GET / returns 200 with API metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "endpoints" in data


@patch("main.ingest_pipeline")
def test_ingest_endpoint(mock_pipeline):
    """Verify POST /ingest schema, status code, and response format."""
    mock_pipeline.return_value = {
        "status": "success",
        "nodes_created": 14,
        "edges_created": 18,
        "files_processed": 2
    }

    payload = {"pdf_dir": "/data/pdfs/"}
    response = client.post("/ingest", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["nodes_created"] == 14
    assert data["edges_created"] == 18
    assert data["files_processed"] == 2


def test_ingest_endpoint_validation():
    """Verify POST /ingest validates request body."""
    response = client.post("/ingest", json={})
    assert response.status_code == 422


@patch("main.get_subgraph_data")
def test_graph_endpoint(mock_get_subgraph):
    """Verify GET /graph query params, schema, and limits."""
    mock_nodes = [
        {
            "id": f"node-{i}",
            "name": f"Entity {i}",
            "type": "Concept",
            "description": f"Description for entity {i}"
        }
        for i in range(15)
    ]
    mock_edges = [
        {
            "source": f"node-{i}",
            "target": f"node-{i+1}",
            "relation": "CONNECTED_TO"
        }
        for i in range(14)
    ]
    mock_get_subgraph.return_value = {"nodes": mock_nodes, "edges": mock_edges}

    response = client.get("/graph?limit=20")
    assert response.status_code == 200
    data = response.json()

    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) <= 20

    # Verify node schema
    for node in data["nodes"]:
        assert "id" in node
        assert "name" in node
        assert "type" in node
        assert "description" in node

    # Verify edge schema
    for edge in data["edges"]:
        assert "source" in edge
        assert "target" in edge
        assert "relation" in edge


@patch("main.run_graphrag")
def test_graphrag_endpoint(mock_run_graphrag):
    """Verify POST /graphrag schema and response format."""
    mock_run_graphrag.return_value = {
        "question": "What is backpropagation?",
        "anchor_nodes": [
            {
                "name": "Backpropagation",
                "type": "Algorithm",
                "score": 0.95
            }
        ],
        "subgraph_context": "=== Anchor Nodes ===\n- Backpropagation [Algorithm]...",
        "answer": "Backpropagation is a supervised learning algorithm that calculates gradients of the loss function."
    }

    payload = {"question": "What is backpropagation?"}
    response = client.post("/graphrag", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["question"] == "What is backpropagation?"
    assert isinstance(data["anchor_nodes"], list)
    assert len(data["anchor_nodes"]) == 1
    assert data["anchor_nodes"][0]["name"] == "Backpropagation"
    assert data["anchor_nodes"][0]["type"] == "Algorithm"
    assert isinstance(data["anchor_nodes"][0]["score"], float)
    assert isinstance(data["subgraph_context"], str)
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0


def test_graphrag_endpoint_empty_question():
    """Verify POST /graphrag rejects empty questions."""
    response = client.post("/graphrag", json={"question": ""})
    assert response.status_code == 400
