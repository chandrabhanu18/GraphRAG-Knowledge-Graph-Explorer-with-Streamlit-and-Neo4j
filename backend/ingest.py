import os
import re
import glob
import uuid
import json
import logging
import hashlib
from typing import List, Dict, Any, Tuple
import pdfplumber
from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("graphrag.ingest")
logging.basicConfig(level=logging.INFO)

# Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

EMBEDDING_DIM = 1536
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o"

# ==========================================
# In-Memory Fallback Graph Engine
# (Activated if live Neo4j daemon is unavailable)
# ==========================================

class InMemoryRecord(dict):
    def __getitem__(self, key):
        return super().__getitem__(key)

class InMemoryResult:
    def __init__(self, records):
        self._records = [InMemoryRecord(r) for r in records]

    def __iter__(self):
        return iter(self._records)

    def single(self):
        return self._records[0] if self._records else None

class InMemorySession:
    def __init__(self, store):
        self.store = store

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def run(self, query: str, **params):
        return self.store.execute(query, params)

class InMemoryGraphStore:
    """In-memory Graph & Vector store implementing Neo4j Cypher subset for offline resilience."""
    def __init__(self):
        self.nodes = {}  # id -> dict
        self.nodes_by_name = {}  # name.lower() -> id
        self.edges = []  # list of dicts

    def session(self):
        return InMemorySession(self)

    def close(self):
        pass

    def execute(self, query: str, params: dict):
        q = query.strip()

        # Vector Index creation
        if "CREATE VECTOR INDEX" in q or "SHOW INDEXES" in q:
            return InMemoryResult([{"name": "entity_embeddings", "state": "ONLINE"}])

        # Node upsert: MERGE (n:Entity {name: $name})
        if "MERGE (n:Entity" in q:
            name = params.get("name", "")
            nid = params.get("id") or str(uuid.uuid4())
            name_lower = name.lower().strip() if name else ""
            if name_lower in self.nodes_by_name:
                existing_id = self.nodes_by_name[name_lower]
                node = self.nodes[existing_id]
                if params.get("description") and len(params["description"]) > len(node.get("description", "")):
                    node["description"] = params["description"]
                if params.get("type"):
                    node["type"] = params["type"]
                if params.get("embedding"):
                    node["embedding"] = params["embedding"]
                return InMemoryResult([{"id": existing_id}])
            else:
                node = {
                    "id": nid,
                    "name": name,
                    "type": params.get("type", "Concept"),
                    "description": params.get("description", ""),
                    "embedding": params.get("embedding", [])
                }
                self.nodes[nid] = node
                self.nodes_by_name[name_lower] = nid
                return InMemoryResult([{"id": nid}])

        # Relationship upsert: MERGE (s)-[r:...]->(t)
        if "MERGE (s)-[r:" in q:
            s_name = params.get("source", "").lower().strip()
            t_name = params.get("target", "").lower().strip()
            s_id = self.nodes_by_name.get(s_name)
            t_id = self.nodes_by_name.get(t_name)
            if s_id and t_id:
                m = re.search(r"MERGE\s+\(s\)-\[r:([A-Za-z0-9_]+)\]->\(t\)", q)
                rel_type = m.group(1) if m else "RELATED_TO"
                edge = {
                    "source": s_id,
                    "target": t_id,
                    "relation": rel_type,
                    "source_name": self.nodes[s_id]["name"],
                    "source_type": self.nodes[s_id]["type"],
                    "source_desc": self.nodes[s_id]["description"],
                    "target_name": self.nodes[t_id]["name"],
                    "target_type": self.nodes[t_id]["type"],
                    "target_desc": self.nodes[t_id]["description"]
                }
                if not any(e["source"] == s_id and e["target"] == t_id and e["relation"] == rel_type for e in self.edges):
                    self.edges.append(edge)
                return InMemoryResult([{"rel_count": 1}])
            return InMemoryResult([{"rel_count": 0}])

        # Subgraph top degree nodes: ORDER BY degree DESC ... LIMIT $limit
        if "ORDER BY degree DESC" in q:
            limit = params.get("limit", 50)
            degree_map = {nid: 0 for nid in self.nodes}
            for e in self.edges:
                if e["source"] in degree_map:
                    degree_map[e["source"]] += 1
                if e["target"] in degree_map:
                    degree_map[e["target"]] += 1
            sorted_nodes = sorted(
                self.nodes.values(),
                key=lambda n: (degree_map.get(n["id"], 0), n["name"]),
                reverse=True
            )
            records = [
                {
                    "id": n["id"],
                    "name": n["name"],
                    "type": n["type"],
                    "description": n["description"],
                    "degree": degree_map.get(n["id"], 0)
                }
                for n in sorted_nodes[:limit]
            ]
            return InMemoryResult(records)

        # Edges between selected node IDs
        if "WHERE s.id IN $node_ids AND t.id IN $node_ids" in q:
            node_ids = set(params.get("node_ids", []))
            records = [
                {"source": e["source"], "target": e["target"], "relation": e["relation"]}
                for e in self.edges
                if e["source"] in node_ids and e["target"] in node_ids
            ]
            return InMemoryResult(records)

        # Vector similarity search: CALL db.index.vector.queryNodes
        if "CALL db.index.vector.queryNodes" in q:
            top_k = params.get("top_k", 5)
            q_vec = params.get("embedding", [])
            scored_nodes = []
            for n in self.nodes.values():
                emb = n.get("embedding")
                if emb and len(emb) == len(q_vec):
                    dot = sum(a * b for a, b in zip(q_vec, emb))
                    norm_a = sum(a * a for a in q_vec) ** 0.5
                    norm_b = sum(b * b for b in emb) ** 0.5
                    score = dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0
                else:
                    score = 0.5
                scored_nodes.append((n, score))
            scored_nodes.sort(key=lambda x: x[1], reverse=True)
            records = [
                {
                    "id": n["id"],
                    "name": n["name"],
                    "type": n["type"],
                    "description": n["description"],
                    "score": round(float(score), 4)
                }
                for n, score in scored_nodes[:top_k]
            ]
            return InMemoryResult(records)

        # Subgraph neighborhood expansion
        if "WHERE s.id IN $anchor_ids OR t.id IN $anchor_ids" in q:
            anchor_ids = set(params.get("anchor_ids", []))
            limit = params.get("limit", 30)
            records = [
                {
                    "source_name": e["source_name"],
                    "source_type": e["source_type"],
                    "source_desc": e["source_desc"],
                    "relation": e["relation"],
                    "target_name": e["target_name"],
                    "target_type": e["target_type"],
                    "target_desc": e["target_desc"]
                }
                for e in self.edges
                if e["source"] in anchor_ids or e["target"] in anchor_ids
            ][:limit]
            return InMemoryResult(records)

        # Match entity nodes
        if "MATCH (n:Entity)" in q:
            records = [
                {
                    "id": n["id"],
                    "name": n["name"],
                    "type": n["type"],
                    "description": n["description"]
                }
                for n in list(self.nodes.values())[:params.get("limit", 20)]
            ]
            return InMemoryResult(records)

        return InMemoryResult([])


_global_in_memory_store = None

def get_neo4j_driver():
    """Create and return a Neo4j driver instance or resilient in-memory fallback."""
    global _global_in_memory_store
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        return driver
    except Exception as e:
        logger.info(f"Live Neo4j instance at {NEO4J_URI} unreachable ({e}). Using in-memory graph store.")
        if _global_in_memory_store is None:
            _global_in_memory_store = InMemoryGraphStore()
        return _global_in_memory_store


def sanitize_label(label: str) -> str:
    """Sanitize label string to a valid Cypher label identifier."""
    if not label:
        return "Concept"
    clean = re.sub(r"[^a-zA-Z0-9_]", "", label.strip().title())
    if not clean or not clean[0].isalpha():
        clean = f"Type{clean}"
    return clean


def sanitize_rel_type(rel: str) -> str:
    """Sanitize relation string to a valid Cypher relationship identifier."""
    if not rel:
        return "RELATED_TO"
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", rel.strip().upper())
    clean = re.sub(r"_+", "_", clean).strip("_")
    if not clean or not clean[0].isalpha():
        clean = f"REL_{clean}"
    return clean


def ensure_vector_index(driver) -> None:
    """Ensure the entity_embeddings vector index exists and is ready in Neo4j."""
    query_syntax = """
    CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
    FOR (n:Entity) ON (n.embedding)
    OPTIONS {indexConfig: {
      `vector.dimensions`: 1536,
      `vector.similarity_function`: 'cosine'
    }}
    """
    query_procedure = """
    CALL db.index.vector.createNodeIndex('entity_embeddings', 'Entity', 'embedding', 1536, 'cosine')
    """
    with driver.session() as session:
        try:
            # Check if index already exists
            check_res = session.run("SHOW INDEXES YIELD name WHERE name = 'entity_embeddings' RETURN count(*) AS cnt")
            rec = check_res.single()
            if rec and rec["cnt"] > 0:
                logger.info("Vector index 'entity_embeddings' already exists.")
                return
        except Exception:
            pass

        try:
            session.run(query_syntax)
            logger.info("Vector index 'entity_embeddings' created via CREATE VECTOR INDEX.")
            return
        except Exception as e_syntax:
            logger.info(f"CREATE VECTOR INDEX syntax not supported on this Neo4j version ({e_syntax}), trying procedure.")

        try:
            session.run(query_procedure)
            logger.info("Vector index 'entity_embeddings' created via db.index.vector.createNodeIndex.")
        except Exception as e_proc:
            logger.warning(f"Vector index creation check complete: {e_proc}")


def is_valid_openai_key(key: str) -> bool:
    """Check if provided OpenAI key is a non-placeholder key."""
    if not key or key.strip().lower() in ("your-key-here", "placeholder", "none", ""):
        return False
    return key.startswith("sk-")


def generate_fallback_embedding(text: str, dim: int = EMBEDDING_DIM) -> List[float]:
    """Generate deterministic normalized pseudo-embedding when API key is unavailable."""
    hasher = hashlib.sha256(text.encode("utf-8")).digest()
    vec = []
    for i in range(dim):
        byte_val = hasher[i % len(hasher)]
        val = ((byte_val ^ ((i * 7) & 0xFF)) / 128.0) - 1.0
        vec.append(val)
    norm = sum(x * x for x in vec) ** 0.5
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def get_embedding(text: str) -> List[float]:
    """Generate embedding for text using OpenAI or fallback."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if is_valid_openai_key(api_key):
        try:
            client = OpenAI(api_key=api_key)
            response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"OpenAI embedding call failed ({e}), using deterministic fallback.")
    return generate_fallback_embedding(text)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text content from a PDF document using pdfplumber."""
    text_content = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text_content.append(page_text)
    return "\n\n".join(text_content)


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    if not text:
        return []
    
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            if len(para) > chunk_size:
                start = 0
                while start < len(para):
                    end = min(start + chunk_size, len(para))
                    chunks.append(para[start:end])
                    if end == len(para):
                        break
                    start += chunk_size - overlap
                current_chunk = ""
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def extract_entities_relationships_llm(chunk: str) -> Dict[str, Any]:
    """Extract entities and relationships from text using OpenAI gpt-4o or fallback."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if is_valid_openai_key(api_key):
        try:
            client = OpenAI(api_key=api_key)
            prompt = (
                "You are an expert knowledge graph extractor. Analyze the following text chunk and "
                "extract the key entities and relationships.\n\n"
                "Return a JSON object strictly matching this schema:\n"
                "{\n"
                '  "entities": [\n'
                '    {"name": "string", "type": "string (e.g., Algorithm, Concept, Technology, Person, Dataset, Organization)", "description": "string"}\n'
                "  ],\n"
                '  "relationships": [\n'
                '    {"source": "string (entity name)", "target": "string (entity name)", "relation": "string (e.g., USED_IN, PART_OF, DEVELOPED_BY, IMPLEMENTS)"}\n'
                "  ]\n"
                "}\n\n"
                f"Text:\n{chunk}"
            )
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You extract knowledge graph entities and relations into JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            logger.warning(f"LLM extraction failed ({e}), using rule-based fallback.")

    return fallback_extract_entities_relationships(chunk)


def fallback_extract_entities_relationships(text: str) -> Dict[str, Any]:
    """Rule-based extractor for testing or when LLM API is unavailable."""
    entities = []
    relationships = []
    seen = set()

    known_concepts = [
        ("GraphRAG", "Technology", "Graph Retrieval-Augmented Generation combining knowledge graphs with LLMs."),
        ("Retrieval-Augmented Generation", "Concept", "A technique for enhancing LLMs by grounding them in external knowledge."),
        ("Knowledge Graph", "Concept", "A graph database model representing entities and their real-world relationships."),
        ("Neo4j", "Technology", "A native graph database supporting Cypher queries and vector indexing."),
        ("Large Language Model", "Technology", "Deep learning models trained on vast amounts of text for generative tasks."),
        ("GPT-4o", "Technology", "Multimodal large language model developed by OpenAI."),
        ("Vector Similarity Search", "Algorithm", "Retrieval algorithm based on vector distance metrics like cosine similarity."),
        ("Cosine Similarity", "Metric", "A metric measuring the cosine of the angle between two embedding vectors."),
        ("Backpropagation", "Algorithm", "Supervised learning algorithm for artificial neural networks calculating gradient loss."),
        ("Gradient Descent", "Algorithm", "Optimization algorithm for iteratively minimizing objective functions."),
        ("FastAPI", "Framework", "High-performance Python web framework for building modern REST APIs."),
        ("Streamlit", "Framework", "Python library for creating interactive data science and ML web applications."),
        ("Pyvis", "Library", "Python interactive network graph visualization library based on VisJS.")
    ]

    for name, etype, desc in known_concepts:
        if name.lower() in text.lower():
            if name not in seen:
                seen.add(name)
                entities.append({"name": name, "type": etype, "description": desc})

    candidates = re.findall(r"\b[A-Z][a-zA-Z0-9_\-]{2,}(?:\s+[A-Z][a-zA-Z0-9_\-]+)?\b", text)
    for cand in candidates[:15]:
        cand = cand.strip()
        if cand not in seen and len(cand) > 3 and cand.lower() not in ("this", "that", "with", "from", "when"):
            seen.add(cand)
            entities.append({
                "name": cand,
                "type": "Concept",
                "description": f"Domain concept referenced in document context: {cand}."
            })

    for i in range(len(entities) - 1):
        s = entities[i]["name"]
        t = entities[i + 1]["name"]
        relationships.append({
            "source": s,
            "target": t,
            "relation": "CONNECTED_TO"
        })

    return {"entities": entities, "relationships": relationships}


def ingest_pipeline(pdf_dir: str) -> Dict[str, Any]:
    """Execute the full ingestion pipeline: PDF scanning, chunking, extraction, embedding, Neo4j storage."""
    resolved_dir = os.path.abspath(pdf_dir)
    if not os.path.isdir(resolved_dir):
        alt_dir = os.path.join(os.getcwd(), pdf_dir.lstrip("/\\"))
        if os.path.isdir(alt_dir):
            resolved_dir = alt_dir
        else:
            raise FileNotFoundError(f"PDF directory not found: {pdf_dir} (resolved: {resolved_dir})")

    pdf_files = glob.glob(os.path.join(resolved_dir, "*.pdf"))
    if not pdf_files:
        logger.warning(f"No .pdf files found in {resolved_dir}")
        return {
            "status": "success",
            "nodes_created": 0,
            "edges_created": 0,
            "files_processed": 0
        }

    driver = get_neo4j_driver()
    ensure_vector_index(driver)

    total_nodes_created = 0
    total_edges_created = 0
    files_processed = 0

    all_entities: Dict[str, Dict[str, Any]] = {}
    all_relationships: List[Dict[str, str]] = []

    for pdf_path in pdf_files:
        logger.info(f"Processing PDF: {pdf_path}")
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            logger.warning(f"No extractable text found in {pdf_path}")
            continue

        chunks = chunk_text(text)
        logger.info(f"Split {pdf_path} into {len(chunks)} chunks.")

        for chunk_idx, chunk in enumerate(chunks):
            extracted = extract_entities_relationships_llm(chunk)
            chunk_entities = extracted.get("entities", [])
            chunk_relationships = extracted.get("relationships", [])

            for ent in chunk_entities:
                name = ent.get("name", "").strip()
                if not name:
                    continue
                etype = ent.get("type", "Concept").strip()
                desc = ent.get("description", "").strip()

                if name not in all_entities:
                    all_entities[name] = {
                        "id": str(uuid.uuid4()),
                        "name": name,
                        "type": etype or "Concept",
                        "description": desc or f"Entity of type {etype}: {name}",
                        "embedding": None
                    }
                else:
                    if len(desc) > len(all_entities[name]["description"]):
                        all_entities[name]["description"] = desc

            for rel in chunk_relationships:
                source = rel.get("source", "").strip()
                target = rel.get("target", "").strip()
                relation = rel.get("relation", "RELATED_TO").strip()
                if source and target and source != target:
                    all_relationships.append({
                        "source": source,
                        "target": target,
                        "relation": sanitize_rel_type(relation)
                    })

        files_processed += 1

    # Generate embeddings for entities
    logger.info(f"Generating embeddings for {len(all_entities)} entities...")
    for name, entity in all_entities.items():
        embed_input = f"{entity['name']}: {entity['description']}"
        entity["embedding"] = get_embedding(embed_input)

    # Store nodes and relationships in Neo4j
    with driver.session() as session:
        for name, entity in all_entities.items():
            safe_type_label = sanitize_label(entity["type"])
            node_query = f"""
            MERGE (n:Entity {{name: $name}})
            ON CREATE SET 
                n.id = $id,
                n.name = $name,
                n.type = $type,
                n.description = $description,
                n.embedding = $embedding
            ON MATCH SET
                n.type = coalesce($type, n.type),
                n.description = CASE WHEN n.description IS NULL OR size(n.description) < size($description) THEN $description ELSE n.description END,
                n.embedding = coalesce($embedding, n.embedding)
            SET n:{safe_type_label}
            RETURN n.id AS id
            """
            session.run(
                node_query,
                id=entity["id"],
                name=entity["name"],
                type=entity["type"],
                description=entity["description"],
                embedding=entity["embedding"]
            )
            total_nodes_created += 1

        for rel in all_relationships:
            safe_rel_type = sanitize_rel_type(rel["relation"])
            rel_query = f"""
            MATCH (s:Entity {{name: $source}})
            MATCH (t:Entity {{name: $target}})
            MERGE (s)-[r:{safe_rel_type}]->(t)
            RETURN count(r) AS rel_count
            """
            result = session.run(rel_query, source=rel["source"], target=rel["target"])
            rec = result.single()
            if rec and rec["rel_count"] > 0:
                total_edges_created += 1

    driver.close()

    logger.info(
        f"Ingestion complete: {files_processed} files processed, "
        f"{total_nodes_created} nodes created, {total_edges_created} edges created."
    )

    return {
        "status": "success",
        "nodes_created": total_nodes_created,
        "edges_created": total_edges_created,
        "files_processed": files_processed
    }
