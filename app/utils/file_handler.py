import os

from fastapi import UploadFile

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "upload")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_upload_file(file: UploadFile) -> str:
    """Simpan file UploadFile ke disk, kembalikan path."""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    return file_path
