from pathlib import Path

CHROMA_PATH = Path("./chroma_db")
# SOURCE_FOLDER = Path(r"C:/Users/Paul/Documents/code/Databases/Data_test/OTT24")
SOURCE_FOLDER = Path("C:/SynologyDrive/VITRO/ETUDES/2024/OTT24.01 SOD screening")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
BATCH_SIZE = 128
PROGRESS_FILE = Path("progress_tracker.jsonl")
EMBEDDING_MODEL="BAAI/bge-m3"
IGNORE_PATH = Path("files_to_ignore.txt")

# SOURCE_FOLDER = Path("C:/SynologyDrive/VITRO/ETUDES/2024/OTT24.01 SOD screening")
