import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import upload, chat

app = FastAPI(
    title="DMS AI",
    description="Document Management System AI (LangChain + Llama3)",
    version="0.1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "upload")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# root health check
@app.get("/")
async def root():
    return {"message": "Welcome to DMS AI API"}

app.include_router(upload.router, prefix="/upload", tags=["upload"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])
