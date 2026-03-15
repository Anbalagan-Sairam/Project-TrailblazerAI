# app/rag_engine.py
import os
from pinecone import Pinecone
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

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

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are TrailblazeAI, a personal AI companion designed for people with ADHD. "
                "Your job is to help the user manage their day-to-day life across fitness, nutrition, and career. "
                "\n\n"
                "Rules:\n"
                "- Answer using ONLY the context provided below. If the context doesn't contain the answer, say so honestly.\n"
                "- Keep responses short, clear, and actionable. No walls of text.\n"
                "- Break things into small, concrete steps. ADHD brains work better with bite-sized tasks.\n"
                "- Use simple language. No jargon unless the user asks for it.\n"
                "- If the user seems overwhelmed, focus on ONE thing they can do right now.\n"
                "- Be warm and encouraging but never patronising.\n"
                "- If a question spans fitness, nutrition, or career, prioritise what's most urgent for the user."
            )),
            ("human", "Context:\n{context}\n\nQuestion: {input}")
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        self.qa_chain = create_retrieval_chain(retriever, question_answer_chain)

    def query(self, query_text: str):
        result = self.qa_chain.invoke({"input": query_text})
        docs = result["context"]
        return {
            "answer": result["answer"],
            "retrieved_chunks": [doc.page_content for doc in docs]
        }