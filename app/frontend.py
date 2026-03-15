# app/frontend.py
import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="TrailblazeAI", layout="wide")
st.title("TrailblazeAI — Your ADHD Personal Companion")
st.caption("Ask me anything about your fitness, nutrition, or career.")

query = st.text_input(
    "What's on your mind?",
    placeholder="e.g. What should I eat today? / What's my next career step?"
)
show_chunks = st.checkbox("Show retrieved context", value=False)

if query:
    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                f"{API_URL}/query",
                json={"query": query},
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            st.error(f"Error: {e}")
            data = None

    if data:
        st.subheader("Here's what I found:")
        st.markdown(data.get("answer", "No answer returned"))

        if show_chunks:
            chunks = data.get("retrieved_chunks")
            if chunks:
                st.subheader("Context used")
                for i, chunk in enumerate(chunks, 1):
                    with st.expander(f"Chunk {i}"):
                        st.text(chunk[:500])
            else:
                st.info("No chunks retrieved.")