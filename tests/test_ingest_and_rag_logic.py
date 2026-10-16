import os
import sys
import pytest

# Add backend to sys.path
sys.path.insert(0, os.path.abspath("backend"))
import ingest
import graphrag


def test_pdf_text_extraction():
    """Verify pdfplumber extracts expected text from sample PDF."""
    pdf_path = "data/pdfs/sample_graphrag_overview.pdf"
    assert os.path.exists(pdf_path)

    text = ingest.extract_text_from_pdf(pdf_path)
    assert len(text) > 50
    assert "GraphRAG" in text
    assert "Neo4j" in text
    assert "Backpropagation" in text


def test_chunk_text():
    """Verify text chunking preserves content and creates valid sized chunks."""
    sample_text = (
        "Paragraph 1 discusses GraphRAG architecture and LLM extraction.\n\n"
        "Paragraph 2 discusses Neo4j graph storage and vector embeddings.\n\n"
        "Paragraph 3 discusses Streamlit frontend visualization and PyVis."
    )
    chunks = ingest.chunk_text(sample_text, chunk_size=100, overlap=20)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) > 0


def test_get_embedding_dimension():
    """Verify embedding function produces 1536-dimensional float vector."""
    vec = ingest.get_embedding("Graph Retrieval-Augmented Generation")
    assert isinstance(vec, list)
    assert len(vec) == 1536
    assert all(isinstance(x, (float, int)) for x in vec)


def test_sanitization_helpers():
    """Verify label and relationship sanitization."""
    assert ingest.sanitize_label("Large Language Model") == "LargeLanguageModel"
    assert ingest.sanitize_label("Algorithm-123!") == "Algorithm123"
    assert ingest.sanitize_label("") == "Concept"

    assert ingest.sanitize_rel_type("used in") == "USED_IN"
    assert ingest.sanitize_rel_type("PART-OF") == "PART_OF"
    assert ingest.sanitize_rel_type("") == "RELATED_TO"


def test_fallback_entity_extraction():
    """Verify fallback extraction extracts entities and relationships from domain text."""
    text = (
        "GraphRAG combines Knowledge Graph technologies with Neo4j and Large Language Model inference. "
        "Backpropagation optimizes weights via Gradient Descent."
    )
    result = ingest.fallback_extract_entities_relationships(text)
    assert "entities" in result
    assert "relationships" in result
    assert len(result["entities"]) >= 3
    names = [e["name"] for e in result["entities"]]
    assert any("GraphRAG" in n or "Neo4j" in n or "Backpropagation" in n for n in names)


def test_build_subgraph_context_formatting():
    """Verify subgraph context formatting string adheres to structure."""
    anchors = [
        {"name": "Backpropagation", "type": "Algorithm", "description": "Computes loss gradients", "score": 0.95}
    ]
    relationships = [
        {
            "source_name": "Backpropagation",
            "source_type": "Algorithm",
            "source_desc": "Computes loss gradients",
            "relation": "USED_IN",
            "target_name": "Gradient Descent",
            "target_type": "Algorithm",
            "target_desc": "Iteratively minimizes loss"
        }
    ]
    context = graphrag.build_subgraph_context(anchors, relationships)
    assert "=== Anchor Nodes ===" in context
    assert "Backpropagation [Algorithm]" in context
    assert "=== Connected Subgraph Relationships ===" in context
    assert "USED_IN" in context


def test_synthesize_answer_fallback():
    """Verify answer synthesis generates informative text from anchor nodes."""
    anchors = [
        {"name": "Cosine Similarity", "type": "Metric", "description": "Measures cosine of angle between embeddings."}
    ]
    ans = graphrag.synthesize_answer("What is Cosine Similarity?", "Context here", anchors)
    assert isinstance(ans, str)
    assert "Cosine Similarity" in ans


def test_chunk_text_boundaries():
    """Verify text chunking boundaries for empty, short, and large inputs."""
    assert ingest.chunk_text("") == []
    assert ingest.chunk_text("   \n\n  ") == []
    
    # Text shorter than chunk_size
    short_text = "Short sentence."
    chunks = ingest.chunk_text(short_text, chunk_size=100)
    assert len(chunks) == 1
    assert chunks[0] == short_text

    # Extremely long paragraph without newlines
    long_para = "word " * 500
    chunks = ingest.chunk_text(long_para, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 250


def test_openai_key_validation():
    """Verify OpenAI key validation boundaries."""
    assert ingest.is_valid_openai_key("sk-proj-1234567890abcdef") is True
    assert ingest.is_valid_openai_key("your-key-here") is False
    assert ingest.is_valid_openai_key("placeholder") is False
    assert ingest.is_valid_openai_key("none") is False
    assert ingest.is_valid_openai_key("") is False
    assert ingest.is_valid_openai_key(None) is False


def test_embedding_determinism_and_norm():
    """Verify deterministic fallback embedding properties."""
    vec1 = ingest.generate_fallback_embedding("Sample domain text")
    vec2 = ingest.generate_fallback_embedding("Sample domain text")
    vec3 = ingest.generate_fallback_embedding("Completely different text")

    assert vec1 == vec2
    assert vec1 != vec3
    norm = sum(x * x for x in vec1) ** 0.5
    assert abs(norm - 1.0) < 1e-5


def test_in_memory_graph_store_operations():
    """Verify InMemoryGraphStore node upsert, edge upsert, vector query, and expansion."""
    store = ingest.InMemoryGraphStore()
    sess = store.session()

    # 1. Upsert nodes
    v1 = ingest.generate_fallback_embedding("Artificial Intelligence")
    v2 = ingest.generate_fallback_embedding("Machine Learning")

    r1 = sess.run("MERGE (n:Entity {name: $name})", name="AI", type="Concept", description="Artificial Intelligence", embedding=v1)
    n1_id = r1.single()["id"]

    r2 = sess.run("MERGE (n:Entity {name: $name})", name="ML", type="Field", description="Machine Learning", embedding=v2)
    n2_id = r2.single()["id"]

    assert n1_id in store.nodes
    assert n2_id in store.nodes

    # Upsert same node with longer description - should update description
    sess.run("MERGE (n:Entity {name: $name})", name="AI", type="Concept", description="Artificial Intelligence is broad")
    assert store.nodes[n1_id]["description"] == "Artificial Intelligence is broad"

    # 2. Upsert edge
    sess.run("MERGE (s)-[r:SUBFIELD_OF]->(t)", source="ML", target="AI")
    assert len(store.edges) == 1
    assert store.edges[0]["relation"] == "SUBFIELD_OF"

    # Duplicate edge prevention
    sess.run("MERGE (s)-[r:SUBFIELD_OF]->(t)", source="ML", target="AI")
    assert len(store.edges) == 1

    # 3. Top degree nodes
    top_deg = sess.run("ORDER BY degree DESC ... LIMIT $limit", limit=10)
    top_records = list(top_deg)
    assert len(top_records) == 2

    # 4. Vector query
    q_vec = ingest.generate_fallback_embedding("Machine Learning algorithms")
    v_results = list(sess.run("CALL db.index.vector.queryNodes", top_k=2, embedding=q_vec))
    assert len(v_results) == 2
    assert v_results[0]["score"] >= v_results[1]["score"]

    # 5. Neighborhood expansion
    exp_results = list(sess.run("WHERE s.id IN $anchor_ids OR t.id IN $anchor_ids", anchor_ids=[n1_id], limit=10))
    assert len(exp_results) == 1
    assert exp_results[0]["relation"] == "SUBFIELD_OF"


def test_synthesize_answer_no_anchors_boundary():
    """Verify synthesize_answer handles empty anchor list gracefully."""
    ans = graphrag.synthesize_answer("Unrelated quantum physics question", "Context", [])
    assert "no direct matches were found" in ans.lower()

