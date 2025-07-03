import os
import requests
from langchain_config import CHROMA_PATH  # optionnel si tu veux centraliser

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Configuration
MISTRAL_API_KEY = "6atvb0C3Zbpdr0TNvC6STCG6qDCsxJ02"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
COLLECTION_NAME = "bge_docs"

# Init embedding + vectordb
embedding = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
vectordb = Chroma(collection_name=COLLECTION_NAME, persist_directory=str(CHROMA_PATH), embedding_function=embedding)
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

def call_mistral(context, question):
    prompt = f"""Voici des informations extraites de documents :
{context}

Question : {question}
Réponds précisément en français."""

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistral-small-2506",
        "messages": [
            {"role": "system", "content": "Tu es un assistant expert qui répond avec précision et clarté."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 512,
    }

    response = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def rag_pipeline(question: str):
    docs = retriever.get_relevant_documents(question)
    context = "\n---\n".join([doc.page_content for doc in docs])
    return call_mistral(context, question)

# Test
if __name__ == "__main__":
    question = "Que contient aaaa.txt ?"
    answer = rag_pipeline(question)
    print("Réponse :", answer)
