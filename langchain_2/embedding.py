from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME,BATCH_SIZE

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cuda")  # forcer GPU

def embed_texts(texts, batch_size=BATCH_SIZE):
    return embedding_model.encode(texts, batch_size=batch_size, show_progress_bar=True)
