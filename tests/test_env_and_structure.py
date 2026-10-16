import os
import pytest
import yaml

def test_project_structure():
    """Verify all required files and directories exist as per req-structure."""
    required_paths = [
        "docker-compose.yml",
        ".env.example",
        "README.md",
        "backend",
        "backend/Dockerfile",
        "backend/main.py",
        "backend/ingest.py",
        "backend/graphrag.py",
        "backend/requirements.txt",
        "frontend",
        "frontend/Dockerfile",
        "frontend/app.py",
        "frontend/api_client.py",
        "frontend/requirements.txt",
        "data",
        "data/pdfs"
    ]
    for path in required_paths:
        assert os.path.exists(path), f"Required path does not exist: {path}"


def test_env_example_content():
    """Verify .env.example contains all required keys with placeholders."""
    assert os.path.exists(".env.example")
    with open(".env.example", "r") as f:
        content = f.read()

    required_keys = [
        "OPENAI_API_KEY",
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
        "BACKEND_URL"
    ]

    for key in required_keys:
        assert f"{key}=" in content, f"Key {key} not found in .env.example"

    # Verify values are placeholders and not actual secrets
    assert "your-key-here" in content
    assert "bolt://neo4j:7687" in content
    assert "http://backend:8000" in content


def test_docker_compose_config():
    """Verify docker-compose.yml defines neo4j, backend, and frontend with required configs."""
    assert os.path.exists("docker-compose.yml")
    with open("docker-compose.yml", "r") as f:
        config = yaml.safe_load(f)

    services = config.get("services", {})
    assert "neo4j" in services, "neo4j service missing in docker-compose.yml"
    assert "backend" in services, "backend service missing in docker-compose.yml"
    assert "frontend" in services, "frontend service missing in docker-compose.yml"

    # neo4j service checks
    neo4j = services["neo4j"]
    assert "neo4j:5.13" in neo4j.get("image", "")
    assert "7474:7474" in neo4j.get("ports", [])
    assert "7687:7687" in neo4j.get("ports", [])
    assert "healthcheck" in neo4j

    # backend service checks
    backend = services["backend"]
    assert "8000:8000" in backend.get("ports", [])
    assert "neo4j" in backend.get("depends_on", {})
    assert backend["depends_on"]["neo4j"]["condition"] == "service_healthy"

    # frontend service checks
    frontend = services["frontend"]
    assert "8501:8501" in frontend.get("ports", [])
    assert "backend" in frontend.get("depends_on", {})
    assert frontend["depends_on"]["backend"]["condition"] == "service_healthy"
    env_vars = frontend.get("environment", [])
    backend_url_set = any("BACKEND_URL=http://backend:8000" in str(e) for e in env_vars)
    assert backend_url_set, "BACKEND_URL not configured correctly in frontend service"
