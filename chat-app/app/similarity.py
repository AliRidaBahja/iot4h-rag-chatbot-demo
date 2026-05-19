import os
import logging
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from chromadb import HttpClient
import numpy as np 

load_dotenv()   
load_dotenv("../../.env")                                                                                                                                          # remove in docker!!!

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
CHROMA_HOST = os.getenv("CHROMA_HOST")    
CHROMA_PORT = int(os.getenv("CHROMA_PORT"))  
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
EMBEDDING_MODEL= os.getenv("EMBEDDING_MODEL") 

client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL) 


vector_store = client.get_collection(name=COLLECTION_NAME)

def get_embedding(text):
    """Get the embedding for a query using Langchain's OpenAIEmbeddings."""
    text = text.replace("\n", " ")  # recommended preprocessing
    return embedding.embed_query(text)

def cosine_similarity(a, b):
    """Compute the cosine similarity between two vectors."""
    if a is None or b is None:
        return 0  # Or you can raise an exception depending on your preference
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def custom_similarity_search(query, k=3, threshold=0.5):
    query_embedding = get_embedding(query)
    results = vector_store.get(include=["documents", "metadatas", "embeddings"])

    docs = results.get("documents", [])
    metas = results.get("metadatas", [])
    embeddings = results.get("embeddings", [])
    if len(docs) == 0 or len(metas) == 0 or len(embeddings) == 0:
        # logging.info("Keine Dokumente in der Datenbank.")
        return []
    similarities = []
    for i, emb in enumerate(embeddings):
        if emb is not None and isinstance(emb, (list, np.ndarray)):
            sim = cosine_similarity(query_embedding, emb)
            # logging.info(f"Ähnlichkeit zu '{metas[i].get('title', 'Kein Titel')}' = {sim:.4f}")   
            similarities.append((sim, i))

    sorted_similarities = sorted(similarities, key=lambda x: x[0], reverse=True)

    filtered = [(score, i) for score, i in sorted_similarities if score >= threshold]
    good_results = filtered[:k]  # get up to k good results
    
    if not good_results:
        logging.info(f"Keine relevanten Ergebnisse über dem Schwellenwert {threshold} gefunden.")
        return []

    return [
        ( Document(page_content=docs[i], metadata=metas[i]), score )
        for score, i in good_results
    ]