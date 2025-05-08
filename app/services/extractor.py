import os
import logging
import tempfile

import pandas as pd
import pdfplumber
from docx import Document
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import spacy
from spacy.cli import download as spacy_download

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy_download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load spaCy model untuk metadata extraction
nlp = spacy.load("en_core_web_sm")


def extract_text_from_image(image):
    """Extract text from an image using OCR."""
    try:
        # Convert image to grayscale for better OCR
        if image.mode != 'L':
            image = image.convert('L')
        # Use pytesseract to extract text
        text = pytesseract.image_to_string(image, lang='eng')
        return text.strip()
    except Exception as e:
        logger.error(f"Error in OCR: {str(e)}")
        return ""


def extract_text_and_metadata(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""

    try:
        if ext == ".pdf":
            logger.info(f"Processing PDF file: {file_path}")
            
            # First try: Extract text directly from PDF
            try:
                with pdfplumber.open(file_path) as pdf:
                    pages_text = []
                    for i, page in enumerate(pdf.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                pages_text.append(page_text)
                            else:
                                logger.warning(f"No text extracted from page {i+1}")
                        except Exception as e:
                            logger.error(f"Error extracting text from page {i+1}: {str(e)}")
                    text = "\n".join(pages_text)
            except Exception as e:
                logger.error(f"Error in direct text extraction: {str(e)}")
                text = ""

            # Second try: If no text extracted, try OCR
            if not text.strip():
                logger.info("No text extracted directly, trying OCR...")
                try:
                    # Convert PDF to images
                    with tempfile.TemporaryDirectory() as temp_dir:
                        images = convert_from_path(file_path)
                        pages_text = []
                        
                        for i, image in enumerate(images):
                            try:
                                # Save image temporarily
                                temp_image_path = os.path.join(temp_dir, f"page_{i}.png")
                                image.save(temp_image_path, "PNG")
                                
                                # Extract text using OCR
                                page_text = extract_text_from_image(image)
                                if page_text:
                                    pages_text.append(page_text)
                                else:
                                    logger.warning(f"No text extracted from page {i+1} using OCR")
                            except Exception as e:
                                logger.error(f"Error processing page {i+1} with OCR: {str(e)}")
                        
                        text = "\n".join(pages_text)
                except Exception as e:
                    logger.error(f"Error in OCR process: {str(e)}")
                    text = ""

            if not text.strip():
                raise ValueError("No text could be extracted from the PDF using either direct extraction or OCR")

        elif ext in [".docx", ".doc"]:
            doc = Document(file_path)
            text = "\n".join(p.text for p in doc.paragraphs)

        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path, header=None)
            # Gabungkan setiap baris jadi satu paragraf
            text = "\n".join(df.astype(str).agg(" ".join, axis=1).tolist())

        elif ext in [".png", ".jpg", ".jpeg", ".tiff"]:
            image = Image.open(file_path)
            text = extract_text_from_image(image)

        else:
            raise ValueError(f"Unsupported file type: {ext}")

        if not text.strip():
            raise ValueError(f"No text could be extracted from the file: {file_path}")

        # Ekstrak entitas dasar sebagai metadata
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        metadata = {
            "entities": entities, 
            "file_type": ext[1:],  # Remove the dot from extension
            "extraction_method": "ocr" if ext == ".pdf" and "pdf2image" in str(text) else "direct"
        }

        logger.info(f"Successfully extracted text and metadata from {file_path}")
        return text, metadata

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        raise
