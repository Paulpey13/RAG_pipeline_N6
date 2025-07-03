from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

def split_documents(docs):
    split_chunks = []
    for doc in docs:
        chunks = splitter.split_documents([doc])
        for chunk in chunks:
            chunk.metadata["source"] = doc.metadata.get("source", "unknown")
        split_chunks.extend(chunks)
    return split_chunks
