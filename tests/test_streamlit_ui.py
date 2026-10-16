import os
import sys
from unittest.mock import patch
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = os.path.abspath("frontend/app.py")


def test_streamlit_initial_ui_state():
    """Verify Streamlit UI components when launched for the first time."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    # Verify Sidebar components
    sidebar = at.sidebar
    assert len(sidebar.file_uploader) >= 1
    file_uploader = sidebar.file_uploader[0]
    assert file_uploader.label == "Upload PDF Documents"
    assert ".pdf" in list(file_uploader.proto.type)
    assert file_uploader.proto.multiple_files is True

    assert len(sidebar.button) >= 1
    button = sidebar.button[0]
    assert button.label == "Build Graph"

    # Verify Knowledge Graph tab has the exact required st.info text when graph_built is False
    assert at.session_state["graph_built"] is False
    infos = at.info
    assert len(infos) >= 1
    expected_info = "Upload PDFs and click 'Build Graph' in the sidebar to populate the knowledge graph."
    assert any(expected_info in info.value for info in infos)

    # Verify Chat Input component exists with placeholder
    assert len(at.chat_input) >= 1
    chat_input = at.chat_input[0]
    assert chat_input.placeholder == "Ask a question about your documents..."


@patch("api_client.fetch_graph")
def test_streamlit_graph_view_when_built(mock_fetch_graph):
    """Verify Knowledge Graph tab renders network and metrics when graph_built is True."""
    mock_fetch_graph.return_value = {
        "nodes": [
            {"id": "node-1", "name": "Backpropagation", "type": "Algorithm", "description": "Gradients"},
            {"id": "node-2", "name": "Gradient Descent", "type": "Optimization", "description": "Minimizer"},
        ],
        "edges": [
            {"source": "node-1", "target": "node-2", "relation": "OPTIMIZED_BY"}
        ]
    }

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.session_state["graph_built"] = True
    at.run()

    # Check metrics
    metrics = at.metric
    assert len(metrics) >= 3
    metric_labels = [m.label for m in metrics]
    assert "Total Nodes" in metric_labels
    assert "Total Edges" in metric_labels
    assert "Node Types" in metric_labels

    for m in metrics:
        if m.label == "Total Nodes":
            assert m.value == "2"
        elif m.label == "Total Edges":
            assert m.value == "1"
        elif m.label == "Node Types":
            assert m.value == "2"


@patch("api_client.fetch_graph")
def test_streamlit_graph_fetch_error(mock_fetch_graph):
    """Verify that an error fetching the graph shows an st.error banner."""
    mock_fetch_graph.side_effect = RuntimeError("Database timeout")

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.session_state["graph_built"] = True
    at.run()

    errors = at.error
    assert len(errors) >= 1
    assert any("Error loading knowledge graph" in err.value for err in errors)


@patch("api_client.query_graphrag")
def test_streamlit_chat_interaction(mock_query_graphrag):
    """Verify submitting a question in Chat tab updates chat history and reasoning."""
    mock_query_graphrag.return_value = {
        "question": "What is Backpropagation?",
        "anchor_nodes": [{"name": "Backpropagation", "type": "Algorithm", "score": 0.95}],
        "subgraph_context": "=== Anchor Nodes ===\n- Backpropagation [Algorithm]",
        "answer": "Backpropagation calculates loss gradients through the chain rule."
    }

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    # Submit question to chat_input
    at.chat_input[0].set_value("What is Backpropagation?").run()

    # Verify session state chat history has been updated
    assert len(at.session_state["chat_history"]) == 1
    history_entry = at.session_state["chat_history"][0]
    assert history_entry["question"] == "What is Backpropagation?"
    assert "Backpropagation calculates loss gradients" in history_entry["answer"]

    # Verify expander exists for reasoning
    expanders = at.expander
    assert any("View reasoning" in exp.label for exp in expanders)


@patch("api_client.query_graphrag")
def test_streamlit_chat_error_handling(mock_query_graphrag):
    """Verify error during GraphRAG QA shows an st.error message."""
    mock_query_graphrag.side_effect = RuntimeError("Backend QA timeout")

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    at.chat_input[0].set_value("Failing question").run()
    errors = at.error
    assert len(errors) >= 1
    assert any("Error querying GraphRAG: Backend QA timeout" in err.value for err in errors)
