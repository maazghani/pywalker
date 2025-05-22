import os
import json
import faiss
import openai
import numpy as np
import argparse

openai.api_key = os.getenv("OPENAI_API_KEY")

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"
TOP_K = 5


def load_index(codebase_name):
    index_path = os.path.join("vector", codebase_name, "faiss.index")
    meta_path = os.path.join("vector", codebase_name, "metadata.jsonl")

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing vector index or metadata for: {codebase_name}")

    index = faiss.read_index(index_path)
    metadata = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            metadata.append(json.loads(line.strip()))
    return index, metadata


def embed_query(query):
    response = openai.embeddings.create(
        model=EMBED_MODEL,
        input=[query]
    )
    return np.array(response.data[0].embedding, dtype=np.float32)


def search_index(index, metadata, query_vector, top_k=TOP_K):
    distances, indices = index.search(query_vector.reshape(1, -1), top_k)
    results = []
    for i in indices[0]:
        if i < len(metadata):
            results.append(metadata[i])
    return results


def load_snippet(entry):
    if entry["type"] == "function":
        return f"# From: {entry['source']}\n{entry['code']}\n'''{entry['doc']}'''"
    elif entry["type"] == "file":
        try:
            with open(entry["source"], "r", encoding="utf-8") as f:
                contents = f.read()
            return f"# From file: {entry['source']}\n{contents}"
        except Exception as e:
            return f"# Could not load file: {entry['source']} — {e}"
    else:
        return ""


def ask_gpt(context, question):
    response = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant who answers questions about a Python codebase."},
            {"role": "user", "content": f"{context}\n\nQuestion: {question}"}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query a vectorized codebase with GPT-4o.")
    parser.add_argument("question", help="Your natural language question about the codebase")
    parser.add_argument("--codebase", required=True, help="Name of the codebase folder under ./vector/")
    parser.add_argument("--answer", action="store_true", help="Use GPT-4o to answer the question")
    args = parser.parse_args()

    print(f"🔍 Searching in `{args.codebase}` for: {args.question}")
    index, metadata = load_index(args.codebase)
    query_vec = embed_query(args.question)
    top_chunks = search_index(index, metadata, query_vec)

    snippets = [load_snippet(entry) for entry in top_chunks]
    context = "\n\n".join(snippets)

    print("\n📚 Top Relevant Code Snippets:")
    for i, snippet in enumerate(snippets):
        print(f"\n--- [#{i+1}] ---\n{snippet}\n")

    if args.answer:
        print("🤖 Asking GPT-4o...")
        answer = ask_gpt(context, args.question)
        print("\n💬 GPT-4o Answer:\n" + answer)
