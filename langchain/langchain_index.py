import os
from pathlib import Path
from tqdm import tqdm
from langchain_config import SOURCE_FOLDER, CHROMA_PATH, CHUNK_SIZE, CHUNK_OVERLAP, PROGRESS_FILE

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Désactiver télémétrie LangChain
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = ""

# Modèle d'embedding local
EMBEDDING_MODEL = "BAAI/bge-m3"
embedding = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

# Base vectorielle persistante
vectordb = Chroma(
    collection_name="bge_docs",
    persist_directory=str(CHROMA_PATH),
    embedding_function=embedding
)

# Chargement dynamique selon extension
def load_documents(file_path: Path):
    try:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return PyPDFLoader(str(file_path)).load()
        elif suffix == ".txt":
            return TextLoader(str(file_path), encoding="utf-8").load()
        elif suffix == ".docx":
            return Docx2txtLoader(str(file_path)).load()
        elif suffix == ".csv":
            return CSVLoader(str(file_path)).load()
        elif suffix in [".xls", ".xlsx"]:
            return UnstructuredExcelLoader(str(file_path)).load()
    except Exception as e:
        print(f"[ERROR] {file_path}: {e}")
    return []

def already_indexed(file_path: Path):
    if not PROGRESS_FILE.exists():
        return False
    return str(file_path.resolve()) in PROGRESS_FILE.read_text()

def mark_indexed(file_path: Path):
    with PROGRESS_FILE.open("a", encoding="utf-8") as f:
        f.write(str(file_path.resolve()) + "\n")

def main():
    print("[START] Indexation avec LangChain")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    files = list(SOURCE_FOLDER.rglob("*"))

    for file in tqdm(files, desc="Indexing"):
        if not file.is_file():
            continue
        if already_indexed(file):
            continue
        if file.name == "indexfile.txt":
            continue
        docs = load_documents(file)
        
        if not docs:
            continue
        for doc in docs:
            doc.metadata["source"] = str(file)
        chunks = splitter.split_documents(docs)
        vectordb.add_documents(chunks)
        mark_indexed(file)

    print("[END] Indexation terminée")

if __name__ == "__main__":
    main()
