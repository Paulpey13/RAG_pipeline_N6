import os
import time
import json
import uuid
import docx
import chromadb
from tqdm import tqdm
from pathlib import Path
from PyPDF2 import PdfReader
from main_config import *
from sentence_transformers import SentenceTransformer

# Init ChromaDB
client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_or_create_collection(name="bge_docs")

# Init modèle embedding local (GPU)
model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")

def load_progress():
    if not PROGRESS_FILE.exists():
        return set()
    return set(line.strip() for line in PROGRESS_FILE.read_text(encoding="utf-8").splitlines())


def save_progress(path):
    with PROGRESS_FILE.open("a", encoding="utf-8") as f:
        f.write(path + "\n")

import pandas as pd

def extract_text(file: Path):
    try:
        if file.suffix == ".txt":
            return file.read_text(encoding="utf-8", errors="ignore")
        elif file.suffix == ".pdf":
            return "\n".join([p.extract_text() or "" for p in PdfReader(file).pages])
        elif file.suffix == ".docx":
            return "\n".join([p.text for p in docx.Document(file).paragraphs])
        elif file.suffix == ".csv":
            df = pd.read_csv(file, encoding="utf-8", errors="ignore")
            return df.astype(str).apply(lambda x: ' '.join(x), axis=1).str.cat(sep='\n')
        elif file.suffix in [".xls", ".xlsx"]:
            df = pd.read_excel(file)
            return df.astype(str).apply(lambda x: ' '.join(x), axis=1).str.cat(sep='\n')
    except Exception as e:
        print(f"[ERREUR] {file}: {e}")
    return None




def split_text_fast(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    step = size - overlap
    return [text[i:i+size] for i in range(0, len(text), step)]

def process_file(file: Path):
    text = extract_text(file)
    if not text:
        return

    chunks = split_text_fast(text)
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]
        embeddings = model.encode(batch, batch_size=BATCH_SIZE, device="cuda")
        collection.add(
            ids=[str(uuid.uuid4()) for _ in batch],
            documents=batch,
            metadatas=[{"source": str(file)}] * len(batch),
            embeddings=embeddings
        )

def load_ignored_filenames():
    ignore_path = IGNORE_PATH
    if not ignore_path.exists():
        return set()
    return set(line.strip() for line in ignore_path.read_text().splitlines())

def main():
    print("[START] Indexation avec BGE local")
    done = load_progress()
    ignored_filenames = load_ignored_filenames()
    files = list(SOURCE_FOLDER.rglob("*"))

    # Étape de filtrage et sélection préférentielle des .docx
    base_name_map = {}
    for file in files:
        if not file.is_file():
            continue
        if file.suffix.lower() not in [".pdf", ".docx"]:
            continue
        if file.name in ignored_filenames:
            continue

        base_name = file.stem  # nom sans extension
        existing = base_name_map.get(base_name)

        if not existing:
            base_name_map[base_name] = file
        else:
            # Si le nouveau est un .docx, il écrase le .pdf existant
            if file.suffix.lower() == ".docx":
                base_name_map[base_name] = file

    for file in tqdm(base_name_map.values(), desc="Indexing"):
        str_path = str(file.resolve())
        if str_path in done:
            continue

        process_file(file)
        save_progress(str_path)

    print("[FIN] Indexation locale terminée")


if __name__ == "__main__":
    main()
