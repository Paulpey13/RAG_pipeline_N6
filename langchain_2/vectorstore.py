import chromadb
from config import CHROMA_DIR, COLLECTION_NAME

client = chromadb.PersistentClient(path=str(CHROMA_DIR))

def get_or_create_collection():
    try:
        return client.get_collection(COLLECTION_NAME)
    except chromadb.errors.CollectionNotFoundError:
        return client.create_collection(COLLECTION_NAME)

def reset_collection():
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass
    return client.create_collection(COLLECTION_NAME)
