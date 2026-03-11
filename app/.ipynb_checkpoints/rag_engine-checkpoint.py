import os
from langchain_aws import ChatBedrock
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings

VECTORSTORE_PATH = "vectorstore"

class RAGEngine:
    def __init__(self, aws_region, embedding_model_id, llm_model_id, k=3):
        self.aws_region = aws_region
        self.embedding_model_id = embedding_model_id
        self.llm_model_id = llm_model_id
        self.k = k

        # Load vectorstore
        self.vectorstore = self._load_vectorstore()
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k}
        )

        # Initialize LLM
        self.llm = self._init_llm()

    def _load_vectorstore(self):
        embedding_function = BedrockEmbeddings(
            model_id=self.embedding_model_id,
            region_name=self.aws_region
        )
        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            embedding_function,
            allow_dangerous_deserialization=True
        )
        return vectorstore

    def _init_llm(self):
        llm = ChatBedrock(
            model_id=self.llm_model_id,
            region_name=self.aws_region
        )
        return llm

    def query_llm(self, query: str):
        # Retrieve top k relevant chunks
        relevant_docs = self.retriever.get_relevant_documents(query)
        if not relevant_docs:
            return {
                "answer": "No relevant information found.",
                "retrieved_chunks": []
            }

        # Combine context with numbered chunks
        context = "\n\n".join(
            [f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(relevant_docs)]
        )

        # Context-aware system prompt
        system_prompt = (
            "You are a smart assistant that answers user queries using the provided context.\n"
            "Rules:\n"
            "- Use only the context information.\n"
            "- If the answer is not present, say the information is unavailable.\n"
            "- Be clear and concise.\n"
            "- Use bullet points or summaries when helpful."
        )

        response = self.llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
        ])

        # Return structured output
        return {
            "answer": response.content,
            "retrieved_chunks": [doc.page_content for doc in relevant_docs]
        }