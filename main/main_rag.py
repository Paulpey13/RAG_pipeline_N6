import os
import argparse
import requests
from sentence_transformers import SentenceTransformer
import chromadb
from main_config import *

# Désactive la télémétrie
os.environ["TELEMETRY_DISABLED"] = "1"

# Argument parsing
parser = argparse.ArgumentParser(description="RAG pipeline via Mistral API + ChromaDB")
parser.add_argument("--question", required=True, type=str, help="Question à poser")
parser.add_argument("--model", default="mistral-large-2407", type=str, help="Nom du modèle Mistral")
parser.add_argument("--temperature", default=0.7, type=float, help="Valeur de temperature")
parser.add_argument("--top_p", default=0.9, type=float, help="Valeur de top_p")
parser.add_argument("--top_k", default=5, type=float, help="Valeur de top_k")
parser.add_argument("--max_tokens", default=5120, type=int, help="Nombre max de tokens")

args = parser.parse_args()

# Initialisations
client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_collection("bge_docs")
model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]

def embed_query(query):
    return model.encode([query], device="cuda")[0]

def retrieve_docs(query_embedding, top_k=args.top_k):
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
        "model": args.model,
        "messages": [
            {"role": "system", "content": "Tu es un expert qui répond en français de manière claire et précise."},
            {"role": "user", "content": prompt}
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }

    response = requests.post(MISTRAL_API_URL, headers=headers, json=data)
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

# Appel pipeline
print(rag_pipeline(args.question))
