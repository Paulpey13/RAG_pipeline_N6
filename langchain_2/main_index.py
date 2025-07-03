import json
from pathlib import Path
from tqdm import tqdm

from config import DATA_FOLDER, PROGRESS_FILE
from loaders import load_file
from splitter import split_documents
from embedding import embed_texts
from vectorstore import reset_collection

def load_progress():
    progress_path = Path(PROGRESS_FILE)
    if progress_path.exists():
        with open(progress_path, "r") as f:
            return json.load(f)
    return {}

def save_progress(data):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def index_data():
    collection = reset_collection()
    docs = []

    indexed_files = load_progress()
    files = list(DATA_FOLDER.glob("**/*.*"))

    for f in tqdm(files, desc="Loading files"):
        mtime = f.stat().st_mtime
        str_path = str(f)
        if str_path in indexed_files and indexed_files[str_path] == mtime:
            # Fichier déjà indexé et inchangé
            continue
        try:
            loaded_docs = load_file(f)
            for doc in loaded_docs:
                doc.metadata["source"] = str(f)
            chunks = split_documents(loaded_docs)
            docs.extend(chunks)
            indexed_files[str_path] = mtime  # Marquer comme indexé
            save_progress(indexed_files)    # Sauvegarde après chaque fichier
        except Exception as e:
            print(f"Error loading {f}: {e}")

    if docs:
        texts = [d.page_content for d in docs]
        metadatas = [d.metadata for d in docs]
        embeddings = embed_texts(texts)
        collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=[f"doc_{i}" for i in range(len(texts))]
        )
    print(f"Indexed {len(docs)} new chunks")

if __name__ == "__main__":
    index_data()
