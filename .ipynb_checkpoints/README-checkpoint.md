Access FrontEnd UI via: https://<Studio ID>.studio.us-east-1.sagemaker.aws/jupyterlab/default/proxy/8501/

Process:
1. Upload the PDF and text files inside data/
2. Run vector_manager.py to create text embeddings and store it in vectorstore/
3. Expose fastapi to listen to port 8000 via:
   uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
4. Use another tab to submit the query to obtain response:
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is Starbucks revenue for 2025?"}'

5. Open streamlit in:

https://<APP_ID>.studio.us-east-1.sagemaker.aws/jupyterlab/default/proxy/8501/