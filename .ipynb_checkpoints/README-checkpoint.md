We use uvicorn web server that listens to HTTP requests, passes them to FASTAPI app and sends the response back to client.

uvicorn app.main:app --host 0.0.0.0 --port 8000