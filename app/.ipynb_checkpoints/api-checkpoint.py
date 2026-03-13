# app/api.py

import os
import sys
from dotenv import load_dotenv

# -----------------------------
# Load environment variables first
# -----------------------------
load_dotenv()  # ensures PINECONE_API_KEY, BEDROCK_* are loaded

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.config import AppConfig
from app.rag_engine import RAGEngine
import boto3
from botocore.exceptions import ClientError

# -----------------------------
# Load and validate config
# -----------------------------
try:
    config = AppConfig()
except Exception as e:
    print(f"[FATAL] Configuration validation failed: {e}")
    sys.exit(1)

# -----------------------------
# Bedrock connectivity check
# -----------------------------
try:
    client = boto3.client("bedrock", region_name=config.aws_region)
    client.list_foundation_models()
except ClientError as e:
    print(f"[FATAL] Unable to access Bedrock in region '{config.aws_region}': {e}")
    sys.exit(1)

# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI(
    title="TrailblazeAI RAG API",
    description="RAG system with FastAPI endpoints and startup validation",
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
# Global RAG engine
# -----------------------------
rag_engine: RAGEngine = None

@app.on_event("startup")
def startup_event():
    global rag_engine
    print("[INFO] Initializing RAG Engine on startup...")
    try:
        # Pinecone RAGEngine now reads API key / env automatically
        rag_engine = RAGEngine(top_k=config.model_k)
    except Exception as e:
        print(f"[FATAL] Failed to initialize RAG Engine: {e}")
        sys.exit(1)
    print("[INFO] RAG Engine ready!")

# -----------------------------
# Health endpoint
# -----------------------------
@app.get("/health")
def health():
    return {"status": "healthy"}

# -----------------------------
# Ready endpoint
# -----------------------------
@app.get("/ready")
def ready():
    return {"ready": rag_engine is not None}

# -----------------------------
# Query endpoint
# -----------------------------
@app.post("/query")
def query(request: QueryRequest):
    if rag_engine is None:
        return {"error": "RAG engine not ready"}
    
    try:
        result = rag_engine.query(request.query)
        return {
            "question": request.query,
            "answer": result["answer"],
            "retrieved_chunks": result["retrieved_chunks"]
        }
    except Exception as e:
        print(f"[ERROR] Query failed: {e}")
        return {"error": str(e)}