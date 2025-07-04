import os
os.environ["TELEMETRY_DISABLED"] = "1"
import requests
from sentence_transformers import SentenceTransformer
import chromadb
from main_config import *

# Initialisations
client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_collection("bge_docs")
model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")

MISTRAL_API_URL = "https://api.mistral.ai/v1/generate"
MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]

def embed_query(query):
    return model.encode([query], device="cuda")[0]

def retrieve_docs(query_embedding, top_k=3):
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    return results['documents'][0]

def generate_answer(question, context):
    prompt = f"""Voici des informations extraites de documents :
{context}

Question : {question}
Réponse précise et complète en français :"""

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "mistral-large-2407",  # ou "mistral-small-latest" selon ton offre
        "messages": [
            {"role": "system", "content": "Tu es un expert qui répond en français de manière claire et précise."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 5120,
    }

    response = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        print(f"Erreur API Mistral: {response.status_code} {response.text}")
        return None


def rag_pipeline(question):
    emb = embed_query(question)
    docs = retrieve_docs(emb)
    context = "\n---\n".join(docs)
    answer = generate_answer(question, context)
    return answer

# Test
question = "Fais une analyse statistique de plate_results.txt"
print(rag_pipeline(question))
