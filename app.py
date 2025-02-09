from flask import Flask, request, render_template, jsonify
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.llms import GPT4All

app = Flask(__name__)


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./znu_db")
collection = chroma_client.get_collection(name="znu_knowledge")


llm = GPT4All(model="./models/gpt4all-falcon.Q3_K_S.gguf")


def search_chroma(query):
    query_vector = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_vector], n_results=3)
    docs = [hit for hit in results["documents"][0]]
    return " ".join(docs) if docs else "هیچ اطلاعات مرتبطی پیدا نشد."


def chat_rag(query):
    context = search_chroma(query)
    prompt = f"""
    اطلاعات زیر را با دقت بررسی کن و فقط بر اساس آن‌ها به سوال پاسخ بده. اگر اطلاعات کافی نبود، بگو که اطلاعات کافی ندارم.
    
    اطلاعات:
    {context}

    سوال: {query}
    جواب:
    """
    response = llm.invoke(prompt)
    return response


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["question"]
        answer = chat_rag(user_input)
        return jsonify({"answer": answer})
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
