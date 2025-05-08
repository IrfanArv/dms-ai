from langchain.text_splitter import RecursiveCharacterTextSplitter


def preprocess_text(text: str):
    # 1) Bersihkan whitespace berlebih
    cleaned = " ".join(text.split())

    # 2) Split menjadi chunk (misal 1000 token, overlap 200)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_text(cleaned)
    return chunks
