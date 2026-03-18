# Trailblaze AI

Trailblaze AI is a conversational RAG-powered assistant built for developers and individuals with ADHD. It uses your own documents as a knowledge base and lets you query them through a natural language interface — giving you instant, context-aware answers without digging through files manually.

---

## What it does

The system currently supports two knowledge domains out of the box:

**ADHD Support** — ingests articles and research on theory of mind, common ADHD challenges, coping strategies, and productivity techniques. Ask it things like "how do I stop forgetting tasks" or "why do I struggle socially" and it retrieves the most relevant guidance.

**Business Intelligence** — ingests structured documents like annual reports (Starbucks performance data is included). Ask financial and strategic questions like "what was revenue in 2023" or "which segment performed best" and it pulls the exact context.

**Nutrition** — ingests the Irish food pyramid and dietary guidelines. Ask questions like "how many servings of vegetables should I eat daily" or "what foods give me sustained energy for focus" and it returns evidence-based nutritional guidance tailored to your goals.

All domains live in the same Pinecone vector store, tagged with metadata so retrieval stays accurate across knowledge bases.

---

## How it works

```
Your documents → chunked + embedded → Pinecone
User query → embedded → similarity search → top-k chunks → LLM → answer
```

---

## Stack

- **Vector store** — Pinecone
- **Embeddings + LLM** — AWS Bedrock via boto3
- **Orchestration** — LangChain
- **API** — FastAPI
- **Frontend** — Streamlit
- **Environment** — AWS SageMaker Studio

---

## Getting started

**1. Clone the repo**
```bash
git clone https://github.com/your-org/trailblaze-ai
cd trailblaze-ai
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up environment variables**

Create a `.env` file at the root:
```
PINECONE_API_KEY=your_key_here
PINECONE_ENV=your_environment
AWS_REGION=us-east-1
```

**4. Ingest documents**
```bash
python vector_manager.py
```

**5. Start the API**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

**6. Start the frontend** (in a second terminal)
```bash
streamlit run frontend.py
```

**7. Access the UI in SageMaker Studio**
```
https://<APP_ID>.studio.us-east-1.sagemaker.aws/jupyterlab/default/proxy/8501/
```

---

## API usage

POST /query from any client — mobile app, web app, Slack bot, anything:

```http
POST /query
Content-Type: application/json

{
  "question": "What are the best strategies for managing ADHD at work?",
  "domain": "adhd"
}
```

Response:
```json
{
  "answer": "...",
  "sources": ["helpguide.org", "nimh.nih.gov"]
}
```
