# app/frontend.py
import streamlit as st
import requests
import os
from dotenv import load_dotenv

# -----------------------------
# Load environment
# -----------------------------
load_dotenv()
API_URL = os.environ.get("API_URL", "http://localhost:8000")  # FastAPI URL

st.set_page_config(page_title="TrailblazeAI RAG Frontend", layout="wide")
st.title("TrailblazeAI RAG - Starbucks Document Search")

# -----------------------------
# User input
# -----------------------------
query = st.text_input(
    "Ask a question about Starbucks:",
    placeholder="Type your query here..."
)

show_chunks = st.checkbox("Show retrieved chunks (context)", value=True)

# -----------------------------
# Send request to FastAPI
# -----------------------------
if query:
    with st.spinner("Querying RAG Engine..."):
        try:
            response = requests.post(
                f"{API_URL}/query",
                json={"query": query},
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            st.error(f"Error querying backend: {e}")
            data = None

    # -----------------------------
    # Display results
    # -----------------------------
    if data:
        st.subheader("Answer")
        st.markdown(data.get("answer", "No answer returned"))

        if show_chunks:
            chunks = data.get("retrieved_chunks")
            if chunks:
                st.subheader("Retrieved Chunks (Context)")
                for i, chunk in enumerate(chunks, 1):
                    preview = chunk[:500].replace("\n", " ")
                    st.text(f"Chunk {i}: {preview} ...")
            else:
                st.info("No retrieved chunks returned.")