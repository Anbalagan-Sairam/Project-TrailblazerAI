# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import boto3
import json

app = FastAPI(title="Bedrock RAG Demo", version="0.1.0")

# Request/Response models
class AskRequest(BaseModel):
    query: str

class AskResponse(BaseModel):
    answer: str
    sources: List[str] = []

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

# Create Bedrock client
bedrock_client = boto3.client("bedrock-runtime")

# /ask endpoint
@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        # Nova Lite payload
        payload = {
            "schemaVersion": "messages-v1",
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": request.query}]
                }
            ]
        }

        response = bedrock_client.invoke_model(
            modelId="amazon.nova-lite-v1:0",  # Replace if using another active Nova model
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json"
        )

        result = json.loads(response["body"].read())

        # Parse model response correctly
        answer = ""
        output_message = result.get("output", {}).get("message", {})
        content_list = output_message.get("content", [])
        if len(content_list) > 0:
            # Nova Lite returns text inside content[0]["text"]
            answer = content_list[0].get("text", "")
        
        if not answer:
            answer = "No answer returned from model"

    except Exception as e:
        answer = f"Error calling Bedrock: {e}"

    return AskResponse(answer=answer, sources=[])