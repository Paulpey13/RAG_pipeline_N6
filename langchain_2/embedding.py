from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
# model = SentenceTransformer("BAAI/bge-m3", device="cuda")

def embed_texts(texts):
    return embedding_model.encode(texts, show_progress_bar=True)
