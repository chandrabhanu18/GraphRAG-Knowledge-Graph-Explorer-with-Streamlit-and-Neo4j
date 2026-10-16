import os
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ingest import ingest_pipeline
from graphrag import get_subgraph_data, run_graphrag

logger = logging.getLogger("graphrag.api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="GraphRAG Knowledge Graph API",
    description="Backend API for PDF knowledge graph extraction and GraphRAG exploration with Neo4j and LLMs.",
    version="1.0.0"
)

# Enable CORS for frontend and external communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Pydantic Schemas matching exact contracts
# ==========================================

class IngestRequest(BaseModel):
    pdf_dir: str = Field(..., description="Path inside the container, e.g. /data/pdfs/")

class IngestResponse(BaseModel):
    status: str
    nodes_created: int
    edges_created: int
    files_processed: int

class NodeSchema(BaseModel):
    id: str
    name: str
    type: str
    description: str

class EdgeSchema(BaseModel):
    source: str
    target: str
    relation: str

class GraphResponse(BaseModel):
    nodes: List[NodeSchema]
    edges: List[EdgeSchema]

class GraphRAGRequest(BaseModel):
    question: str

class AnchorNode(BaseModel):
    name: str
    type: str
    score: float

class GraphRAGResponse(BaseModel):
    question: str
    anchor_nodes: List[AnchorNode]
    subgraph_context: str
    answer: str


# ==========================================
# Endpoints
# ==========================================

@app.get("/health")
def health_check():
    """Service healthcheck endpoint."""
    return {"status": "ok"}


@app.get("/")
def root():
    """Service root endpoint."""
    return {
        "service": "GraphRAG Knowledge Graph API",
        "endpoints": {
            "POST /ingest": "Process PDFs and construct knowledge graph",
            "GET /graph": "Retrieve subgraph for visualization",
            "POST /graphrag": "Execute GraphRAG QA pipeline"
        }
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(request: IngestRequest):
    """
    Ingest PDF documents, extract entities and relationships using LLM,
    generate vector embeddings, and store in Neo4j.
    """
    try:
        result = ingest_pipeline(request.pdf_dir)
        return result
    except FileNotFoundError as e:
        logger.error(f"Directory not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error during ingestion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.get("/graph", response_model=GraphResponse)
def get_graph_endpoint(limit: int = Query(default=50, ge=1, le=500)):
    """
    Retrieve a subgraph of the most connected nodes and their relationships for visualization.
    """
    try:
        data = get_subgraph_data(limit=limit)
        return data
    except Exception as e:
        logger.error(f"Error retrieving graph: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Graph retrieval failed: {str(e)}")


@app.post("/graphrag", response_model=GraphRAGResponse)
def graphrag_endpoint(request: GraphRAGRequest):
    """
    Perform GraphRAG: embed question, vector search for anchor nodes, expand subgraph,
    synthesize answer using LLM.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = run_graphrag(request.question)
        return result
    except Exception as e:
        logger.error(f"Error in GraphRAG pipeline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"GraphRAG execution failed: {str(e)}")
