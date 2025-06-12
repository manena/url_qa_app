from flask import Flask, request, render_template
import requests
from bs4 import BeautifulSoup
import openai
import chromadb
from chromadb.config import Settings
import tiktoken
import os
from dotenv import load_dotenv

# Configure API Key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# Create Flask app
app = Flask(__name__)

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./chroma_store")
collection_name = "web_chunks"

# Remove previous collection if needed (reset per session)
if collection_name in [c.name for c in client.list_collections()]:
    client.delete_collection(name=collection_name)
collection = client.create_collection(name=collection_name)

# --- Utility functions ---
def scrape_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)

def split_text(text, max_tokens=500, overlap=50):
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = enc.decode(tokens[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap
    return chunks

def embed_texts(texts):
    embeddings = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i + 100]
        response = openai.Embedding.create(input=batch, model="text-embedding-ada-002")
        embeddings.extend([e["embedding"] for e in response["data"]])
    return embeddings

def retrieve_context(query, collection, top_k=3):
    query_embedding = openai.Embedding.create(input=[query], model="text-embedding-ada-002")["data"][0]["embedding"]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results["documents"][0]

def ask_with_context(question, context_chunks):
    context = "\n".join(context_chunks)
    prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    error = None

    if request.method == "POST":
        url = request.form.get("url")
        question = request.form.get("question")

        try:
            # Scrape and process
            text = scrape_text_from_url(url)
            chunks = split_text(text)
            embeddings = embed_texts(chunks)

            # Store in vector DB
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                collection.add(documents=[chunk], embeddings=[embedding], ids=[f"doc_{i}"])

            # RAG flow
            context = retrieve_context(question, collection)
            answer = ask_with_context(question, context)

        except Exception as e:
            error = f"Error: {str(e)}"

    return render_template("index.html", answer=answer, error=error)

# --- Run app ---
if __name__ == "__main__":
    app.run(debug=True)
