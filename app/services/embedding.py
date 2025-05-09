import os, json
import logging
from langchain_chroma.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from app.services.llm_client import generate_embedding
import chromadb
from chromadb.utils import embedding_functions

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_DIR = os.getenv("CHROMA_DIR", "vectorstore")
os.makedirs(CHROMA_DIR, exist_ok=True)

class OllamaEmbeddings(Embeddings):
    model: str = "llama3"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            logger.warning("No texts provided for embedding")
            return []
        embeddings = generate_embedding(texts, model=self.model)
        if not embeddings:
            logger.error("Failed to generate embeddings")
            raise ValueError("Failed to generate embeddings")
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        if not text:
            logger.warning("Empty text provided for embedding")
            return []
        embeddings = generate_embedding([text], model=self.model)
        if not embeddings:
            logger.error("Failed to generate embedding")
            raise ValueError("Failed to generate embedding")
        return embeddings[0]

class ChromaOllamaEmbeddingFunction(embedding_functions.EmbeddingFunction, Embeddings):
    def __init__(self, model: str = "llama3"):
        self.model = model

    def __call__(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            logger.warning("No texts provided for embedding")
            return []
        embeddings = generate_embedding(texts, model=self.model)
        if not embeddings:
            logger.error("Failed to generate embeddings")
            raise ValueError("Failed to generate embeddings")
        return embeddings
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self(texts)

    def embed_query(self, text: str) -> list[float]:
        return self([text])[0]

embeddings = ChromaOllamaEmbeddingFunction()
vectorstore = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

def embed_and_store(chunks: list[str], metadata: dict):
    if not chunks:
        logger.warning("No chunks provided for embedding")
        return
    
    # Filter out empty chunks
    chunks = [chunk for chunk in chunks if chunk.strip()]
    if not chunks:
        logger.warning("No non-empty chunks to embed")
        return

    ids = [f"doc_{i}" for i in range(len(chunks))]
    entities_str = json.dumps([
        {"text": ent[0], "label": ent[1]} for ent in metadata.get("entities", [])
    ])
    metadatas = [
        {
            "chunk_index": i,
            "entities": entities_str,
            "file_type": metadata.get("file_type", "unknown"),
            "source": metadata.get("filename", "")
        }
        for i in range(len(chunks))
    ]

    try:
        vectorstore.add_texts(texts=chunks, metadatas=metadatas, ids=ids)
        logger.info(f"Successfully stored {len(chunks)} chunks in vector store")
    except Exception as e:
        logger.error(f"Error storing chunks in vector store: {str(e)}")
        raise
