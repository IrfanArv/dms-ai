from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional
import json
import asyncio
import aiohttp
import os
from datetime import datetime
import base64

from app.services.embedding import vectorstore
from app.services.llm_client import generate_response

router = APIRouter()

class ChatRequest(BaseModel):
    query: str
    context_window: Optional[int] = 5
    temperature: Optional[float] = 0.7

def get_doc_count():
    # Try to get count from Chroma collection, fallback to upload dir
    try:
        return vectorstore._collection.count()
    except Exception:
        upload_dir = "upload"
        return len([f for f in os.listdir(upload_dir) if not f.startswith('.')])

def get_file_metadata(upload_dir="upload"):
    files = [f for f in os.listdir(upload_dir) if not f.startswith('.')]
    total_files_uploaded = len(files)
    file_list = ", ".join(files)
    last_upload_date = None
    if files:
        last_upload_date = max(
            [datetime.fromtimestamp(os.path.getmtime(os.path.join(upload_dir, f))) for f in files]
        ).strftime("%Y-%m-%d %H:%M:%S")
    else:
        last_upload_date = "-"
    return total_files_uploaded, file_list, last_upload_date

def get_document_overview(files, upload_dir="upload"):
    overview = []
    for f in files:
        path = os.path.join(upload_dir, f)
        size = os.path.getsize(path) // 1024  # KB
        overview.append(f"{f} ({size}KB)")
    return " | ".join(overview)

async def generate_rag_response(query: str, context_window: int = 5, temperature: float = 0.7):
    """Generate RAG response with streaming."""
    try:
        # 1. Search vector DB
        results = vectorstore.similarity_search_with_score(
            query,
            k=context_window
        )
        
        # 2. Prepare context
        context = "\n\n".join([doc.page_content for doc, score in results])
        doc_count = get_doc_count()
        
        # Fetch file metadata and overview
        files_in_upload = [f for f in os.listdir("upload") if not f.startswith('.')]
        total_files_uploaded, file_list, last_upload_date = get_file_metadata()
        document_overview = get_document_overview(files_in_upload)

        # 3. Prepare the new unified prompt
        prompt = f"""Kamu adalah DMS AI, asisten AI cerdas yang bertugas membantu pengguna terkait dokumen perusahaan.
Saat ini kamu memiliki akses ke {doc_count} dokumen.
Berikut adalah metadata file yang tersedia:
- Total file diupload: {total_files_uploaded}
- Daftar file: {file_list}
- Tanggal upload terakhir: {last_upload_date}
- Ringkasan Dokumen: {document_overview}

Tugasmu adalah:
1.  PAHAMI DOKUMEN: Pahami isi dokumen, termasuk template dan format yang ada.
2.  BANTU BUAT DOKUMEN: Bantu pengguna membuat dokumen baru sesuai dengan template dan kebutuhan mereka, menggunakan informasi dari dokumen yang ada jika relevan.
3.  CARI & AMBIL DATA: Temukan dan sajikan informasi spesifik dari dalam dokumen.
4.  ANALISIS & LAPORKAN: Baca, pindai, buat ringkasan, atau laporan berdasarkan isi dan metadata dokumen.
5.  INTERAKSI ALAMI: Berinteraksilah seolah-olah kamu adalah admin dokumen yang kompeten, menggunakan Bahasa Indonesia yang profesional dan jelas.
6.  JANGAN BERI KOMENTAR: Jangan beri komentar apa-apa, hanya berikan jawaban saja.
7.  Ketika pembukaan percakapan, kamu harus memperkenalkan diri sebagai DMS AI dengan singkat, asisten AI cerdas yang bertugas membantu pengguna terkait dokumen perusahaan. dan tidak perlu beri data atau informasi perusahaan. atau baca 
Gunakan RETRIEVED_CHUNKS sebagai dasar faktual utama untuk jawabanmu.
Jika informasi tidak ada dalam RETRIEVED_CHUNKS, nyatakan bahwa data tidak ditemukan dalam dokumen yang tersedia.

Konteks (RETRIEVED_CHUNKS):
{context}

Pertanyaan Pengguna (USER_QUERY): {query}

Jawaban (dalam Bahasa Indonesia):
"""
        
        # 4. Generate streaming response
        async for chunk in generate_response(prompt, temperature=temperature):
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        
        # 5. Send source documents as the last message
        sources_data = [
            {
                "metadata": {
                    "source": doc.metadata.get("source", ""),
                    "file_type": doc.metadata.get("file_type", "")
                },
                "score": float(score)
            }
            for doc, score in results
        ]
        yield f"data: {json.dumps({'sources': sources_data})}\n\n"
            
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint with RAG."""
    return StreamingResponse(
        generate_rag_response(
            request.query,
            context_window=request.context_window,
            temperature=request.temperature
        ),
        media_type="text/event-stream"
    )

@router.post("/")
async def chat(request: ChatRequest):
    """Non-streaming chat endpoint with RAG."""
    try:
        results = vectorstore.similarity_search_with_score(
            request.query,
            k=request.context_window
        )
        context_chunks = [doc.page_content for doc, score in results]
        context = "\n\n".join(context_chunks)
        
        # Fetch file metadata and overview
        doc_count = get_doc_count()
        files_in_upload = [f for f in os.listdir("upload") if not f.startswith('.')]
        total_files_uploaded, file_list, last_upload_date = get_file_metadata()
        document_overview = get_document_overview(files_in_upload)

        # New unified prompt
        unified_prompt = f"""Kamu adalah DMS AI, asisten AI cerdas yang bertugas membantu pengguna terkait dokumen perusahaan.
Saat ini kamu memiliki akses ke {doc_count} dokumen.
Berikut adalah metadata file yang tersedia:
- Total file diupload: {total_files_uploaded}
- Daftar file: {file_list}
- Tanggal upload terakhir: {last_upload_date}
- Ringkasan Dokumen: {document_overview}

Tugasmu adalah:
1.  PAHAMI DOKUMEN: Pahami isi dokumen, termasuk template dan format yang ada.
2.  BANTU BUAT DOKUMEN: Bantu pengguna membuat dokumen baru sesuai dengan template dan kebutuhan mereka, menggunakan informasi dari dokumen yang ada jika relevan.
3.  CARI & AMBIL DATA: Temukan dan sajikan informasi spesifik dari dalam dokumen.
4.  ANALISIS & LAPORKAN: Baca, pindai, buat ringkasan, atau laporan berdasarkan isi dan metadata dokumen.
5.  INTERAKSI ALAMI: Berinteraksilah seolah-olah kamu adalah admin dokumen yang kompeten, menggunakan Bahasa Indonesia yang profesional dan jelas.
6.  JANGAN BERI KOMENTAR: Jangan beri komentar apa-apa, hanya berikan jawaban saja.
7.  Ketika pembukaan percakapan, kamu harus memperkenalkan diri sebagai DMS AI, asisten AI cerdas yang bertugas membantu pengguna terkait dokumen perusahaan. dan gausah beri data atau informasi lainnya.

Gunakan RETRIEVED_CHUNKS sebagai dasar faktual utama untuk jawabanmu.
Jika informasi tidak ada dalam RETRIEVED_CHUNKS, nyatakan bahwa data tidak ditemukan dalam dokumen yang tersedia.

Konteks (RETRIEVED_CHUNKS):
{context}

Pertanyaan Pengguna (USER_QUERY): {request.query}

Jawaban (dalam Bahasa Indonesia):
"""
        
     
        response_chunks = []
    
        async for chunk_json_str in generate_response(unified_prompt, temperature=request.temperature):
            try:
                # If generate_response sends JSON strings like {"text": "chunk_content"}
                if isinstance(chunk_json_str, str) and chunk_json_str.startswith('{') and chunk_json_str.endswith('}'):
                    data = json.loads(chunk_json_str)
                    if "text" in data:
                        response_chunks.append(data["text"])
                # If generate_response sends raw text chunks
                elif isinstance(chunk_json_str, str):
                     response_chunks.append(chunk_json_str)

            except json.JSONDecodeError:
                # If it's not JSON and not a simple string, or malformed JSON, append directly if it's a string
                if isinstance(chunk_json_str, str):
                    response_chunks.append(chunk_json_str) # Fallback for raw text not in JSON
                # else, ignore if it's some other data type we don't expect.
        
        response_text = "".join(response_chunks)
        
        return {
            "response": response_text,
            "sources": [
                {
                    "metadata": {
                        "source": doc.metadata.get("source", ""),
                        "file_type": doc.metadata.get("file_type", "")
                    },
                    "score": float(score)
                }
                for doc, score in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 

@router.get("/document/{filename}")
async def get_document(filename: str):
    """Get document file from upload directory."""
    file_path = os.path.join("upload", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/pdf"
    ) 