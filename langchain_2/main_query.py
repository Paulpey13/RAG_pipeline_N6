from rag import rag_search

if __name__ == "__main__":
    question = input("Question : ")
    results = rag_search(question, top_k=5)
    docs = results['documents'][0]
    metadatas = results['metadatas'][0]

    for i, (doc, meta) in enumerate(zip(docs, metadatas)):
        print(f"Chunk {i} (source: {meta.get('source')}):\n{doc}\n{'-'*40}")
