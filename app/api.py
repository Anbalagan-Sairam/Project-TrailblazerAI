import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.rag_engine import RAGEngine

# -----------------------------
# Load environment
# -----------------------------
load_dotenv()
AWS_REGION = os.environ.get("AWS_REGION")
BEDROCK_EMBED_MODEL = os.environ.get("BEDROCK_EMBED_MODEL")
BEDROCK_LLM_MODEL = os.environ.get("BEDROCK_LLM_MODEL")

# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI(
    title="TrailblazeAI RAG API",
    description="RAG system with FastAPI endpoints",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# -----------------------------
# Request schema
# -----------------------------
class QueryRequest(BaseModel):
    query: str

# -----------------------------
# Global RAG engine (initialized once on startup)
# -----------------------------
rag_engine: RAGEngine = None

@app.on_event("startup")
def startup_event():
    global rag_engine
    print("[INFO] Initializing RAG Engine on startup...")
    rag_engine = RAGEngine(
        aws_region=AWS_REGION,
        embedding_model_id=BEDROCK_EMBED_MODEL,
        llm_model_id=BEDROCK_LLM_MODEL,
        k=3
    )
    print("[INFO] RAG Engine ready!")

# -----------------------------
# Health check
# -----------------------------
@app.get("/health")
def health():
    return {"status": "healthy"}

# -----------------------------
# Ready check (RAG loaded)
# -----------------------------
@app.get("/ready")
def ready():
    if rag_engine is None:
        return {"ready": False}
    return {"ready": True}

# -----------------------------
# Query endpoint
# -----------------------------
@app.post("/query")
def query(request: QueryRequest):
    if rag_engine is None:
        return {"error": "RAG engine not ready"}
    
    result = rag_engine.query_llm(request.query)
    return {
        "question": request.query,
        "answer": result["answer"],
        "retrieved_chunks": result["retrieved_chunks"]
    }