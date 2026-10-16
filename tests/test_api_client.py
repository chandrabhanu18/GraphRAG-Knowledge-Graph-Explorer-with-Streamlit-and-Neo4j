import os
import sys
import inspect
from unittest.mock import patch, MagicMock
import pytest
import requests

# Add frontend to sys.path
sys.path.insert(0, os.path.abspath("frontend"))
import api_client


def test_api_client_functions_defined():
    """Verify api_client defines required functions with correct signatures."""
    assert hasattr(api_client, "ingest_pdfs")
    assert hasattr(api_client, "fetch_graph")
    assert hasattr(api_client, "query_graphrag")

    sig_ingest = inspect.signature(api_client.ingest_pdfs)
    assert "pdf_dir" in sig_ingest.parameters

    sig_graph = inspect.signature(api_client.fetch_graph)
    assert "limit" in sig_graph.parameters
    assert sig_graph.parameters["limit"].default == 50

    sig_rag = inspect.signature(api_client.query_graphrag)
    assert "question" in sig_rag.parameters


def test_api_client_no_direct_db_or_llm():
    """Verify api_client does not contain direct Neo4j or OpenAI imports or calls."""
    with open("frontend/api_client.py", "r") as f:
        code = f.read()

    assert "neo4j" not in code.lower()
    assert "openai" not in code.lower()
    assert "requests" in code


@patch("requests.post")
def test_ingest_pdfs_http_call(mock_post):
    """Verify ingest_pdfs performs expected HTTP POST call and returns dict."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "status": "success",
        "nodes_created": 10,
        "edges_created": 12,
        "files_processed": 1
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    result = api_client.ingest_pdfs("/data/pdfs/")
    assert result["status"] == "success"
    assert result["nodes_created"] == 10
    mock_post.assert_called_once()
    called_url = mock_post.call_args[0][0]
    assert called_url.endswith("/ingest")


@patch("requests.get")
def test_fetch_graph_http_call(mock_get):
    """Verify fetch_graph performs expected HTTP GET call and returns dict."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "nodes": [{"id": "1", "name": "NodeA", "type": "Concept", "description": "Desc"}],
        "edges": [{"source": "1", "target": "2", "relation": "RELATES"}]
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    result = api_client.fetch_graph(limit=25)
    assert len(result["nodes"]) == 1
    assert len(result["edges"]) == 1
    mock_get.assert_called_once()
    called_url = mock_get.call_args[0][0]
    assert called_url.endswith("/graph")
    assert mock_get.call_args[1]["params"]["limit"] == 25


@patch("requests.post")
def test_query_graphrag_http_call(mock_post):
    """Verify query_graphrag performs expected HTTP POST call and returns dict."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "question": "What is backpropagation?",
        "anchor_nodes": [{"name": "Backpropagation", "type": "Algorithm", "score": 0.95}],
        "subgraph_context": "=== Anchor Nodes ===",
        "answer": "Backpropagation calculates loss gradients."
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    result = api_client.query_graphrag("What is backpropagation?")
    assert result["question"] == "What is backpropagation?"
    assert len(result["anchor_nodes"]) == 1
    assert "answer" in result
    mock_post.assert_called_once()
    called_url = mock_post.call_args[0][0]
    assert called_url.endswith("/graphrag")


@pytest.mark.parametrize("error", [
    requests.exceptions.ConnectionError("Connection refused by backend"),
    requests.exceptions.Timeout("Read timeout after 300s"),
    requests.exceptions.HTTPError("500 Server Error: Ingestion failed"),
])
@patch("requests.post")
def test_ingest_pdfs_network_failures(mock_post, error):
    """Verify ingest_pdfs raises RuntimeError on network/HTTP errors."""
    mock_post.side_effect = error
    with pytest.raises(RuntimeError) as exc_info:
        api_client.ingest_pdfs("/data/pdfs/")
    assert "Backend ingestion error" in str(exc_info.value)


@pytest.mark.parametrize("error", [
    requests.exceptions.ConnectionError("Connection refused"),
    requests.exceptions.Timeout("Timeout"),
    requests.exceptions.HTTPError("500 Internal Server Error"),
])
@patch("requests.get")
def test_fetch_graph_network_failures(mock_get, error):
    """Verify fetch_graph raises RuntimeError on network/HTTP errors."""
    mock_get.side_effect = error
    with pytest.raises(RuntimeError) as exc_info:
        api_client.fetch_graph(50)
    assert "Backend graph fetch error" in str(exc_info.value)


@pytest.mark.parametrize("error", [
    requests.exceptions.ConnectionError("Connection failed"),
    requests.exceptions.Timeout("Timeout"),
    requests.exceptions.HTTPError("400 Bad Request: Question cannot be empty."),
])
@patch("requests.post")
def test_query_graphrag_network_failures(mock_post, error):
    """Verify query_graphrag raises RuntimeError on network/HTTP errors."""
    mock_post.side_effect = error
    with pytest.raises(RuntimeError) as exc_info:
        api_client.query_graphrag("Question")
    assert "Backend GraphRAG query error" in str(exc_info.value)

