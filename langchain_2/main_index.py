import json
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from config import DATA_FOLDER, PROGRESS_FILE,BATCH_SIZE
from loaders import load_file
from splitter import split_documents
from embedding import embed_texts
from vectorstore import reset_collection
from filter import filter_files

def load_progress():
    progress_path = Path(PROGRESS_FILE)
    if progress_path.exists():
        try:
            with open(progress_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_progress(data):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def process_file(f):
    try:
        loaded_docs = load_file(f)
        for doc in loaded_docs:
            doc.metadata["source"] = str(f)
        return split_documents(loaded_docs)
    except Exception as e:
        print(f"Error processing {f}: {e}")
        return []

def batch(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


def index_data():
    collection = reset_collection()
    indexed_files = load_progress()
    files = list(DATA_FOLDER.glob("**/*.*"))
    filtered_files = filter_files(files, DATA_FOLDER)

    # Traitement parallèle du chargement + découpage
    docs = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for chunks in tqdm(executor.map(process_file, filtered_files), total=len(filtered_files), desc="Processing files"):
            docs.extend(chunks)

    # Filtrer docs déjà indexés par fichier modifié
    new_docs = []
    updated_indexed_files = indexed_files.copy()
    for doc in docs:
        src = doc.metadata.get("source", "")
        f = Path(src)
        mtime = f.stat().st_mtime if f.exists() else None
        if mtime is None:
            continue
        if src not in indexed_files or indexed_files[src] != mtime:
            new_docs.append(doc)
            updated_indexed_files[src] = mtime

    if new_docs:
        texts = [d.page_content for d in new_docs]
        metadatas = [d.metadata for d in new_docs]
        embeddings = embed_texts(texts, batch_size=BATCH_SIZE)
        for i, (batch_texts, batch_embeddings, batch_metas) in enumerate(zip(
            batch(texts, 5000),
            batch(embeddings.tolist(), 5000),
            batch(metadatas, 5000)
        )):
            collection.add(
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metas,
                ids=[f"doc_{i}_{j}" for j in range(len(batch_texts))]
            )

        save_progress(updated_indexed_files)
        print(f"Indexed {len(new_docs)} new chunks")
    else:
        print("No new documents to index.")

if __name__ == "__main__":
    index_data()
