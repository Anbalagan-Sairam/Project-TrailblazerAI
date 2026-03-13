Trailblaze AI v1.0

Trailblaze AI is a conversational AI for developers purpose built for assisting ADHD individuals leverage the power of AI agents to manage their day to day activities.

How to use:
1. Clone this repo into your local machine.
2. Get a free tier for pinecone and grab the api and set it in .env
3. Run vector_manager.py to create text embeddings and store it in
4. Expose fastapi to listen to port 8000 via: uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
5. Access FrontEnd UI via: https://<Studio ID>.studio.us-east-1.sagemaker.aws/jupyterlab/default/proxy/8501/
6. Open streamlit in:
https://<APP_ID>.studio.us-east-1.sagemaker.aws/jupyterlab/default/proxy/8501/
7. Chat with your AI that now has a complete information about your exercise routines, nutrition, resume, career to build motivation bridge to progress your goals.