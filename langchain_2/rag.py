from embedding import embed_texts
from vectorstore import get_or_create_collection

collection = get_or_create_collection()

def rag_search(query, top_k=5):
    q_emb = embed_texts([query])[0]
    results = collection.query(query_embeddings=[q_emb.tolist()], n_results=top_k)
    return results
