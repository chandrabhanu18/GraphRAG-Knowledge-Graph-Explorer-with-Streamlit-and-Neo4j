import os
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

from ingest import (
    get_embedding,
    get_neo4j_driver,
    is_valid_openai_key,
    LLM_MODEL
)

load_dotenv()

logger = logging.getLogger("graphrag.pipeline")
logging.basicConfig(level=logging.INFO)


def get_subgraph_data(limit: int = 50) -> Dict[str, Any]:
    """Retrieve top most connected nodes and their interconnections for visualization."""
    driver = get_neo4j_driver()
    nodes = []
    edges = []

    with driver.session() as session:
        # 1. Fetch top connected nodes ordered by degree
        node_query = """
        MATCH (n:Entity)
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(r) AS degree
        ORDER BY degree DESC, n.name ASC
        LIMIT $limit
        RETURN n.id AS id, n.name AS name, n.type AS type, n.description AS description
        """
        result = session.run(node_query, limit=limit)
        node_ids = set()
        for rec in result:
            nid = rec["id"]
            if not nid:
                continue
            node_ids.add(nid)
            nodes.append({
                "id": str(nid),
                "name": str(rec["name"] or ""),
                "type": str(rec["type"] or "Concept"),
                "description": str(rec["description"] or "")
            })

        # 2. Fetch edges between these selected nodes
        if node_ids:
            edge_query = """
            MATCH (s:Entity)-[r]->(t:Entity)
            WHERE s.id IN $node_ids AND t.id IN $node_ids
            RETURN DISTINCT s.id AS source, t.id AS target, type(r) AS relation
            """
            edge_result = session.run(edge_query, node_ids=list(node_ids))
            for rec in edge_result:
                edges.append({
                    "source": str(rec["source"]),
                    "target": str(rec["target"]),
                    "relation": str(rec["relation"])
                })

    driver.close()
    return {"nodes": nodes, "edges": edges}


def find_anchor_nodes(driver, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Find anchor nodes using vector similarity search, with keyword fallback if needed."""
    q_embedding = get_embedding(question)
    anchor_nodes = []

    with driver.session() as session:
        # Attempt vector index search
        vector_query = """
        CALL db.index.vector.queryNodes('entity_embeddings', $top_k, $embedding)
        YIELD node, score
        RETURN node.id AS id, node.name AS name, node.type AS type, node.description AS description, score
        ORDER BY score DESC
        """
        try:
            res = session.run(vector_query, top_k=top_k, embedding=q_embedding)
            for rec in res:
                anchor_nodes.append({
                    "id": str(rec["id"]),
                    "name": str(rec["name"] or ""),
                    "type": str(rec["type"] or "Concept"),
                    "description": str(rec["description"] or ""),
                    "score": float(rec["score"])
                })
        except Exception as e:
            logger.warning(f"Vector search failed ({e}), checking direct keyword fallback.")

        # If vector search returned no results, fallback to keyword matching
        if not anchor_nodes:
            words = [w.strip().lower() for w in question.split() if len(w.strip()) > 3]
            fallback_query = """
            MATCH (n:Entity)
            RETURN n.id AS id, n.name AS name, n.type AS type, n.description AS description
            LIMIT 20
            """
            res = session.run(fallback_query)
            for rec in res:
                name_lower = str(rec["name"]).lower()
                desc_lower = str(rec["description"]).lower()
                score = 0.5
                match_found = False
                for w in words:
                    if w in name_lower:
                        score += 0.3
                        match_found = True
                    elif w in desc_lower:
                        score += 0.1
                        match_found = True
                if match_found or len(anchor_nodes) < top_k:
                    anchor_nodes.append({
                        "id": str(rec["id"]),
                        "name": str(rec["name"] or ""),
                        "type": str(rec["type"] or "Concept"),
                        "description": str(rec["description"] or ""),
                        "score": round(min(score, 0.99), 3)
                    })
                if len(anchor_nodes) >= top_k:
                    break

    return anchor_nodes


def expand_subgraph(driver, anchor_ids: List[str], limit: int = 30) -> List[Dict[str, Any]]:
    """Expand subgraph around anchor nodes to extract related entities and relationships."""
    if not anchor_ids:
        return []

    relationships = []
    with driver.session() as session:
        query = """
        MATCH (s:Entity)-[r]->(t:Entity)
        WHERE s.id IN $anchor_ids OR t.id IN $anchor_ids
        RETURN DISTINCT 
            s.name AS source_name, s.type AS source_type, s.description AS source_desc,
            type(r) AS relation,
            t.name AS target_name, t.type AS target_type, t.description AS target_desc
        LIMIT $limit
        """
        res = session.run(query, anchor_ids=anchor_ids, limit=limit)
        for rec in res:
            relationships.append({
                "source_name": rec["source_name"],
                "source_type": rec["source_type"],
                "source_desc": rec["source_desc"],
                "relation": rec["relation"],
                "target_name": rec["target_name"],
                "target_type": rec["target_type"],
                "target_desc": rec["target_desc"]
            })
    return relationships


def build_subgraph_context(anchor_nodes: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> str:
    """Format anchor nodes and relationship paths into structured context text."""
    lines = ["=== Anchor Nodes ==="]
    if anchor_nodes:
        for node in anchor_nodes:
            score_str = f" (similarity: {node.get('score', 0):.2f})" if "score" in node else ""
            lines.append(f"- {node['name']} [{node['type']}]{score_str}: {node.get('description', '')}")
    else:
        lines.append("- None identified.")

    lines.append("\n=== Connected Subgraph Relationships ===")
    if relationships:
        for rel in relationships:
            lines.append(
                f"- ({rel['source_name']}:{rel['source_type']}) -[:{rel['relation']}]-> ({rel['target_name']}:{rel['target_type']})"
            )
            if rel.get("source_desc"):
                lines.append(f"  * {rel['source_name']}: {rel['source_desc']}")
            if rel.get("target_desc"):
                lines.append(f"  * {rel['target_name']}: {rel['target_desc']}")
    else:
        lines.append("- No direct inter-entity relationships found in the immediate neighborhood.")

    return "\n".join(lines)


def synthesize_answer(question: str, context: str, anchor_nodes: List[Dict[str, Any]]) -> str:
    """Synthesize final answer using OpenAI gpt-4o or structured synthesis."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if is_valid_openai_key(api_key):
        try:
            client = OpenAI(api_key=api_key)
            prompt = (
                f"You are a Knowledge Graph QA expert. Answer the user's question accurately, "
                f"comprehensively, and strictly grounded in the following Knowledge Graph context.\n\n"
                f"Knowledge Graph Context:\n{context}\n\n"
                f"Question:\n{question}\n\n"
                f"Answer:"
            )
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are an AI assistant specialized in GraphRAG."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"OpenAI synthesis failed ({e}), using grounded context synthesis.")

    # Rule-based synthesis grounded in retrieved graph context
    if not anchor_nodes:
        return f"Based on the knowledge graph, no direct matches were found for your query: '{question}'."

    top_node = anchor_nodes[0]
    ans_lines = [
        f"Based on the ingested knowledge graph, **{top_node['name']}** ({top_node['type']}) is identified as a primary anchor node."
    ]
    if top_node.get("description"):
        ans_lines.append(f"{top_node['description']}")
    
    if len(anchor_nodes) > 1:
        related = [f"**{n['name']}** ({n['type']})" for n in anchor_nodes[1:4]]
        ans_lines.append(f"Related entities in the graph neighborhood include {', '.join(related)}.")
    
    return " ".join(ans_lines)


def run_graphrag(question: str) -> Dict[str, Any]:
    """Execute complete GraphRAG pipeline: question embedding, vector search, subgraph expansion, synthesis."""
    driver = get_neo4j_driver()

    # 1. Retrieve anchor nodes
    raw_anchors = find_anchor_nodes(driver, question, top_k=5)
    anchor_ids = [n["id"] for n in raw_anchors if "id" in n]

    # Format anchors strictly according to schema: name, type, score
    formatted_anchors = [
        {
            "name": n["name"],
            "type": n["type"],
            "score": round(float(n.get("score", 0.0)), 4)
        }
        for n in raw_anchors
    ]

    # 2. Expand subgraph
    relationships = expand_subgraph(driver, anchor_ids, limit=30)
    driver.close()

    # 3. Assemble context
    subgraph_context = build_subgraph_context(raw_anchors, relationships)

    # 4. Synthesize answer
    answer = synthesize_answer(question, subgraph_context, raw_anchors)

    return {
        "question": question,
        "anchor_nodes": formatted_anchors,
        "subgraph_context": subgraph_context,
        "answer": answer
    }
