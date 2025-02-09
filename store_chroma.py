import json  
from sentence_transformers import SentenceTransformer
import chromadb


model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./znu_db")
collection = chroma_client.create_collection(name="znu_knowledge")


with open("clean_data.json", "r", encoding="utf-8") as file:
    data = json.load(file) 


for i, doc in enumerate(data):
    vector = model.encode(doc["text"]).tolist()
    collection.add(
        ids=[str(i)],
        documents=[doc["text"]],
        embeddings=[vector],
        metadatas=[{"url": doc["url"]}]
    )


