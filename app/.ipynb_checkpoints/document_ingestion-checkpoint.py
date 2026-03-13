# app/document_ingestion.py
import os
from pathlib import Path
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_EMBED_MODEL = os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v1")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "trailblazeai")
DATA_PATH = Path("data").resolve()


def load_documents_with_metadata(folder: Path):
    docs = []
    for file_path in folder.rglob("*"):
        if file_path.suffix.lower() not in [".txt", ".pdf"]:
            continue

        # category/subcategory
        rel_path = file_path.relative_to(DATA_PATH)
        parts = rel_path.parts
        category = parts[0] if len(parts) > 0 else "uncategorized"
        subcategory = parts[1] if len(parts) > 1 else "general"

        # read text
        content = ""
        if file_path.suffix.lower() == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
        elif file_path.suffix.lower() == ".pdf":
            reader = PdfReader(file_path)
            for page in reader.pages:
                content += (page.extract_text() or "") + "\n"
            content = content.strip()

        if content:
            docs.append({
                "text": content,
                "category": category,
                "subcategory": subcategory,
                "source": str(file_path)
            })
    print(f"[INFO] Loaded {len(docs)} documents with metadata")
    return docs


def split_documents(docs):
    documents = [Document(page_content=d["text"]) for d in docs]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    enriched_chunks = []
    for i, chunk in enumerate(chunks):
        original = docs[i % len(docs)]
        enriched_chunks.append({
            "id": f"doc-{i}",
            "content": chunk.page_content,
            "metadata": {
                "category": original["category"],
                "subcategory": original["subcategory"],
                "source": original["source"]
            }
        })
    print(f"[INFO] Split into {len(enriched_chunks)} chunks with metadata")
    return enriched_chunks


def build_vectorstore():
    print("=== Starting ingestion ===")
    embed_fn = BedrockEmbeddings(model_id=BEDROCK_EMBED_MODEL, region_name=AWS_REGION)
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing = [n for n in pc.list_indexes().names()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=AWS_REGION)
        )
        print(f"[INFO] Created Pinecone index: {INDEX_NAME}")
    index = pc.Index(INDEX_NAME)

    docs = load_documents_with_metadata(DATA_PATH)
    if not docs:
        print("[WARN] No documents found.")
        return

    chunks = split_documents(docs)

    batch = []
    for chunk in chunks:
        vector = embed_fn.embed_query(chunk["content"])
        batch.append((chunk["id"], vector, chunk["metadata"]))
        if len(batch) >= 100:
            index.upsert(vectors=batch)
            batch = []
    if batch:
        index.upsert(vectors=batch)

    print(f"[INFO] Upserted {len(chunks)} vectors into '{INDEX_NAME}'")
    print("=== Ingestion done ===")


if __name__ == "__main__":
    build_vectorstore()