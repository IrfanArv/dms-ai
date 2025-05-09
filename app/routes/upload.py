from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List

from app.utils.file_handler import save_upload_file
from app.services.extractor import extract_text_and_metadata
from app.services.preprocessing import preprocess_text
from app.services.embedding import embed_and_store

router = APIRouter()

#soon add streaming response for better user experience

@router.post("/")
async def upload_doc(file: UploadFile = File(...)):
    # 1) Simpan file
    try:
        file_path = save_upload_file(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # 2) Ekstrak teks & metadata
    try:
        text, metadata = extract_text_and_metadata(file_path)
        # Add filename to metadata
        metadata["filename"] = file.filename
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Extraction error: {e}")

    # 3) Preprocessing + chunking
    chunks = preprocess_text(text)

    # 4) Embedding & simpan ke Vector DB
    embed_and_store(chunks, metadata)

    return JSONResponse(
        status_code=200,
        content={
            "filename": file.filename,
            "metadata": metadata,
            "chunks_count": len(chunks),
        },
    )


@router.post("/batch")
async def upload_multiple_docs(files: List[UploadFile] = File(...)):
    results = []
    
    for file in files:
        try:
            # 1) Simpan file
            file_path = save_upload_file(file)
            
            # 2) Ekstrak teks & metadata
            text, metadata = extract_text_and_metadata(file_path)
            
            # 3) Preprocessing + chunking
            chunks = preprocess_text(text)
            
            # 4) Embedding & simpan ke Vector DB
            embed_and_store(chunks, metadata)
            
            results.append({
                "filename": file.filename,
                "status": "success",
                "metadata": metadata,
                "chunks_count": len(chunks),
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    return JSONResponse(
        status_code=200,
        content={
            "total_files": len(files),
            "processed_files": len(results),
            "results": results
        }
    )
