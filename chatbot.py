import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.llms import GPT4All 


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


chroma_client = chromadb.PersistentClient(path="./znu_db")
collection = chroma_client.get_collection(name="znu_knowledge")


llm = GPT4All(model="./models/gpt4all-falcon.Q3_K_S.gguf")

  

def chat_rag(query):

    query_vector = embedding_model.encode(query).tolist()

    
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=5 
    )

    # ترکیب پاسخ‌های بازیابی‌شده
    context = " ".join(results["documents"][0])

    # ارسال به مدل GPT4All برای تولید پاسخ
    prompt = f"""
اطلاعات زیر را با دقت مطالعه کن و فقط بر اساس آن‌ها به سوال پاسخ بده. اگر اطلاعات کافی نبود، بگو که اطلاعات کافی ندارم.
اطلاعات:
{context}

سوال: {query}
جواب:
"""

    response = llm.invoke(prompt)  # ✅ روش جدید


    return response

# # اجرای چت‌بات در یک حلقه (پرسش و پاسخ)
# print("🤖 چت‌بات RAG آماده است. برای خروج، 'exit' را تایپ کنید.")
# while True:
#     user_input = input(":سوال خود را بپرسید: ")
#     if user_input.lower() == "exit":
#         print("👋 خروج از چت‌بات...")
#         break
#     answer = chat_rag(user_input)
#     print(f"🤖 پاسخ: {answer}")
