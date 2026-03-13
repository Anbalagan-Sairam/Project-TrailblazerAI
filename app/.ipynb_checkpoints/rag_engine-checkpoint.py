# app/rag_engine.py

import os
from pinecone import Pinecone
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA

INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
AWS_REGION = os.environ.get("AWS_REGION")
BEDROCK_EMBED_MODEL = os.environ.get("BEDROCK_EMBED_MODEL")
BEDROCK_LLM_MODEL = os.environ.get("BEDROCK_LLM_MODEL")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

class RAGEngine:

    def __init__(self, top_k: int = 5):

        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)

        embeddings = BedrockEmbeddings(
            model_id=BEDROCK_EMBED_MODEL,
            region_name=AWS_REGION
        )

        # IMPORTANT: tell LangChain where chunk text lives
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="chunk_text"
        )

        retriever = vectorstore.as_retriever(
            search_kwargs={"k": top_k}
        )

        llm = ChatBedrock(
            model_id=BEDROCK_LLM_MODEL,
            region_name=AWS_REGION
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

    def query(self, query_text: str):

        result = self.qa_chain.invoke({"query": query_text})

        docs = result["source_documents"]

        return {
            "answer": result["result"],
            "retrieved_chunks": [doc.page_content for doc in docs]
        }