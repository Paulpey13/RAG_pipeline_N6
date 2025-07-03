from pathlib import Path

DATA_FOLDER = Path("C:/SynologyDrive/VITRO/ETUDES/2024/OTT24.01 SOD screening")
CHROMA_DIR = Path("chroma_db")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
#"all-MiniLM-L6-v2"
COLLECTION_NAME = "rag_collection"
PROGRESS_FILE = "indexed_files.json"
BATCH_SIZE=128
