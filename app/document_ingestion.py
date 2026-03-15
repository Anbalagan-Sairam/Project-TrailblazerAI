#document_ingestion.py
import os
import json
import boto3
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion

# Load environment variables
load_dotenv()

AWS_REGION          = os.environ["AWS_REGION"]
BEDROCK_EMBED_MODEL = os.environ["BEDROCK_EMBED_MODEL"]
PINECONE_API_KEY    = os.environ["PINECONE_API_KEY"]
PINECONE_ENV        = os.environ["PINECONE_ENV"]
INDEX_NAME          = os.environ["PINECONE_INDEX_NAME"]

# Titan embed text produces 1536-dim text embeddings
EMBED_DIM = 1536

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# If the index exists, delete to start fresh
if pc.has_index(INDEX_NAME):
    print(f"Deleting existing index {INDEX_NAME} …")
    pc.delete_index(INDEX_NAME)

print(f"Creating index {INDEX_NAME} with dimension={EMBED_DIM}")
pc.create_index(
    name=INDEX_NAME,
    dimension=EMBED_DIM,
    metric="cosine",
    spec=ServerlessSpec(
        cloud=CloudProvider.AWS,
        region=AwsRegion[AWS_REGION.replace('-', '_').upper()]
    )
)

index = pc.Index(INDEX_NAME)

# Load and read PDFs from data/
DATA_DIR = "data"
files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]

docs = []
for fname in files:
    reader = PdfReader(os.path.join(DATA_DIR, fname))
    text = ""
    for p, page in enumerate(reader.pages):
        text += f"\n--- PAGE {p+1} ---\n" + (page.extract_text() or "")
    docs.append((fname, text))

print(f"Loaded PDFs: {files}")

# Simple chunking function
def chunk_text(txt, size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(txt):
        chunks.append(txt[start:start+size])
        start += size - overlap
    return chunks

# Build a list of chunk records
vectors = []
for fname, text in docs:
    for idx, chunk in enumerate(chunk_text(text)):
        # We keep filename, index, and raw chunk text
        vectors.append((f"{fname}_{idx}", fname, idx, chunk))

print(f"Created {len(vectors)} text chunks")

# Initialize AWS Bedrock embedder
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

def get_embedding(text):
    """
    Calls AWS Bedrock embed model to produce a 1536-dim embedding.
    """
    body = json.dumps({"inputText": text}).encode("utf-8")
    resp = bedrock.invoke_model(
        modelId=BEDROCK_EMBED_MODEL,
        contentType="application/json",
        accept="application/json",
        body=body
    )
    out = json.loads(resp["body"].read().decode("utf-8"))
    return out["embedding"]

# Batch upsert function to keep within Pinecone's ~2MB request limit
def upsert_in_batches(ix, vecs, batch_size=80):
    """
    Upsert chunks in batches to avoid hitting Pinecone's
    2MB per-request size limit.
    """
    for i in range(0, len(vecs), batch_size):
        batch = vecs[i:i+batch_size]
        to_upsert = []
        for vid, fname, idx, chunk in batch:
            emb = get_embedding(chunk)

            # **Store chunk text in metadata for observability**
            metadata = {
                "document_id": fname,
                "chunk_index": idx,
                "chunk_text": chunk,
            }

            to_upsert.append((vid, emb, metadata))

        print(f"Upserting batch {i} → {i+len(batch)} …")
        ix.upsert(vectors=to_upsert)

# Run ingestion
upsert_in_batches(index, vectors, batch_size=80)
print("Ingestion complete!")