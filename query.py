import json
import faiss
import openai
import numpy as np
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"
TOP_K = 5


def load_index(codebase_dir):
    index_path = os.path.join("vector", codebase_dir, "faiss.index")
    meta_path = os.path.join("vector", codebase_dir, "metadata.jsonl")

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


def format_context(results):
    return "\n\n".join(
        f"# From: {r['source']}\n{r['code']}\n'''{r['doc']}'''" for r in results
    )


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
    import argparse
    parser = argparse.ArgumentParser(description="Query a vectorized codebase with GPT-4o.")
    parser.add_argument("question", help="Your natural language question about the codebase")
    parser.add_argument("--codebase", required=True, help="Name of the codebase folder under ./vector/")
    parser.add_argument("--answer", action="store_true", help="Use GPT-4o to answer the question")
    args = parser.parse_args()

    print(f"🔍 Searching in `{args.codebase}` for: {args.question}")
    index, metadata = load_index(args.codebase)
    query_vec = embed_query(args.question)
    top_chunks = search_index(index, metadata, query_vec)

    print("\n📚 Top Relevant Code Snippets:")
    for i, r in enumerate(top_chunks):
        print(f"\n--- [#{i+1}] {r['source']} ---\n{r['code']}\n'''{r['doc']}'''\n")

    if args.answer:
        print("🤖 Asking GPT-4o...")
        context = format_context(top_chunks)
        answer = ask_gpt(context, args.question)
        print("\n💬 GPT-4o Answer:\n" + answer)