import random
from collections import defaultdict
from pathlib import Path
from typing import List
import chromadb
from main_config import CHROMA_PATH

# Initialisation client Chroma
client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_collection("bge_docs")

def count_chunks():
    """Affiche le nombre total de chunks dans la base."""
    total = collection.count()
    print(f"Total de chunks dans la base : {total}")
    return total

def show_first_chunks(n=5):
    """Affiche les n premiers chunks et leurs sources."""
    print(f"Affichage des {n} premiers chunks :")
    results = collection.get(include=["documents", "metadatas"], limit=n)
    for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
        print(f"[{i+1}] Source : {meta['source']}")
        print(f"     Chunk : {doc[:200].strip()}...\n")


def stats_per_source():
    """Affiche combien de chunks proviennent de chaque fichier source."""
    data = collection.get(include=["metadatas"])
    counter = defaultdict(int)
    for meta in data["metadatas"]:
        counter[meta["source"]] += 1
    sorted_sources = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    print("Chunks par source :")
    for source, count in sorted_sources:
        print(f"{source} : {count}")

def average_chunk_length():
    """Affiche la longueur moyenne des chunks (en caractères)."""
    data = collection.get(include=["documents"])
    lengths = [len(doc) for doc in data["documents"]]
    if lengths:
        avg = sum(lengths) / len(lengths)
        print(f"Longueur moyenne des chunks : {avg:.2f} caractères")
    else:
        print("Base vide.")

def search_chunks_by_keyword(keyword: str, max_results=5):
    """Recherche de chunks contenant un mot-clé exact (simple scan brut)."""
    data = collection.get(include=["documents", "metadatas"])
    results = []
    for doc, meta in zip(data["documents"], data["metadatas"]):
        if keyword.lower() in doc.lower():
            results.append((doc, meta))
            if len(results) >= max_results:
                break
    print(f"Résultats contenant '{keyword}' :")
    for i, (doc, meta) in enumerate(results):
        print(f"[{i+1}] Source : {meta['source']}")
        print(f"     Chunk : {doc[:200].strip()}...\n")

def export_all_chunks_to_file(path: str = "exported_chunks.txt"):
    """Exporte tous les chunks avec leur source dans un fichier texte."""
    data = collection.get(include=["documents", "metadatas"])
    with open(path, "w", encoding="utf-8") as f:
        for doc, meta in zip(data["documents"], data["metadatas"]):
            f.write(f"[SOURCE] {meta['source']}\n")
            f.write(doc.strip() + "\n")
            f.write("\n" + "-"*40 + "\n\n")
    print(f"Export terminé dans : {path}")

def get_chunks_by_partial_path(substring: str, max_chunks: int = 10):
    """
    Affiche les chunks dont le chemin source contient un motif donné.
    
    :param substring: Partie du chemin à chercher (ex: nom de fichier ou dossier)
    :param max_chunks: Nombre maximum de chunks à afficher (défaut: 10)
    """
    print(f"Recherche de chunks contenant '{substring}' dans le chemin source :\n")
    data = collection.get(include=["documents", "metadatas"])
    count = 0
    for doc, meta in zip(data["documents"], data["metadatas"]):
        if substring.lower() in meta["source"].lower():
            print(f"[{count+1}] Fichier: {meta['source']}")
            print(f"Chunk : {doc[:200].strip()}...\n")
            count += 1
            if count >= max_chunks:
                break
    if count == 0:
        print("Aucun chunk trouvé pour ce motif.")


def get_largest_sources(n=5):
    """Affiche les fichiers ayant le plus de chunks."""
    data = collection.get(include=["metadatas"])
    counter = defaultdict(int)
    for meta in data["metadatas"]:
        counter[meta["source"]] += 1
    sorted_sources = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    print(f"{n} sources avec le plus de chunks :")
    for source, count in sorted_sources[:n]:
        print(f"{source} : {count} chunks")

def delete_chunks_by_partial_path(substring: str): # Peut être à modifier pour prendre un fichier txt avec tous les path à enlever
    """
    Supprime tous les chunks dont le chemin source contient le motif donné.
    """
    data = collection.get(include=["ids", "metadatas"])
    ids_to_delete = [
        id_ for id_, meta in zip(data["ids"], data["metadatas"])
        if substring.lower() in meta["source"].lower()
    ]
    if not ids_to_delete:
        print("Aucun chunk à supprimer pour ce motif.")
        return
    collection.delete(ids=ids_to_delete)
    print(f"{len(ids_to_delete)} chunks supprimés contenant '{substring}' dans leur chemin.")

get_largest_sources()