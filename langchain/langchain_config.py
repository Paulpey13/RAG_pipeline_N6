from pathlib import Path

# Chemin vers la base de vecteurs persistante
CHROMA_PATH = Path("./chroma_db")

# Dossier source contenant les documents à indexer
SOURCE_FOLDER = Path("C:/SynologyDrive/VITRO/ETUDES/2024/OTT24.01 SOD screening")
# SOURCE_FOLDER = Path("C/Users/Paul/Documents/code/Databases/Data_test/OTT24")

# Paramètres de découpage
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
BATCH_SIZE = 128

# Fichier de suivi de progression (pour ne pas retraiter les mêmes fichiers)
PROGRESS_FILE = Path("progress_tracker.jsonl")
