# app/rag_engine.py

import os
from langchain.schema import Document
from langchain_aws import BedrockEmbeddings, ChatBedrock
from pinecone import Pinecone, ServerlessSpec

# -----------------------------
# Environment variables
# -----------------------------
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "trailblazeai")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
BEDROCK_EMBED_MODEL = os.environ.get("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v1")
LLM_MODEL_ID = os.environ.get("BEDROCK_LLM_MODEL", "amazon.titan-chat")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV", "aws")  # optional

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is missing from environment!")

# -----------------------------
# RAG Engine
# -----------------------------
class RAGEngine:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k

        # Initialize Pinecone client
        self.pc = Pinecone(api_key=PINECONE_API_KEY)

        # Create index if it doesn't exist
        existing_indexes = [n for n in self.pc.list_indexes().names()]
        if INDEX_NAME not in existing_indexes:
            self.pc.create_index(
                name=INDEX_NAME,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=AWS_REGION)
            )

        self.index = self.pc.Index(INDEX_NAME)

        # Embedding function
        self.embed_fn = BedrockEmbeddings(model_id=BEDROCK_EMBED_MODEL, region_name=AWS_REGION)

        # LLM
        self.llm = ChatBedrock(model_id=LLM_MODEL_ID, region_name=AWS_REGION)

    def query(self, query_text: str):
        # Embed query
        query_vector = self.embed_fn.embed_query(query_text)

        # Query Pinecone index
        response = self.index.query(
            vector=query_vector,
            top_k=self.top_k,
            include_metadata=True
        )

        # Extract docs from metadata
        retrieved_docs = []
        for match in response.get("matches", []):
            text = match["metadata"].get("text", "")
            if text:
                retrieved_docs.append(Document(page_content=text))

        # Build context
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        system_prompt = (
            "You are a smart assistant that answers user queries using the provided context. "
            "Always provide clear, concise, and informative answers. "
            "Answer based on context only; do not hallucinate."
        )

        response_llm = self.llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query_text}"}
        ])

        return {
            "answer": response_llm.content,
            "retrieved_chunks": [doc.page_content for doc in retrieved_docs]
        }


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    engine = RAGEngine(top_k=5)
    query = "Summarize Starbucks 2025 annual report"
    result = engine.query(query)
    print("Answer:", result["answer"])
    print("\nRetrieved Chunks:")
    for i, chunk in enumerate(result["retrieved_chunks"], 1):
        print(f"{i}. {chunk[:300].replace(chr(10), ' ')}...")