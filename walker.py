import os
import json
import jedi
import faiss
import openai
import numpy as np

MAX_LENGTH = 7500
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 50

openai.api_key = os.getenv("OPENAI_API_KEY")


def extract_functions_from_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
    except Exception:
        return []

    try:
        script = jedi.Script(source, path=filepath)
    except Exception as e:
        print(f"⚠️ Skipping file (script error): {filepath} — {e}")
        return []

    functions = []
    seen = set()

    for defn in script.get_names(all_scopes=True, definitions=True):
        try:
            if defn.type != "function" or defn.line is None:
                continue
        except Exception as e:
            print(f"⚠️ Skipping defn (type error) in {filepath}: {e}")
            continue

        try:
            lines = source.splitlines()
            func_lines = lines[defn.line - 1:]
            func_source = "\n".join(func_lines)
            docstring = defn.docstring(raw=True) or ""

            prompt = func_source.strip()
            completion = docstring.strip()
            entry_key = (prompt, completion)

            if entry_key in seen or len(prompt) + len(completion) > MAX_LENGTH:
                continue
            seen.add(entry_key)

            functions.append({
                "id": f"{filepath}:{defn.name}",
                "text": f"{prompt}\n\n{completion}",
                "code": prompt,
                "doc": completion,
                "source": filepath
            })
        except Exception as e:
            print(f"⚠️ Skipping malformed function in {filepath}: {e}")
            continue

    return functions


def walk_directory_and_extract(directory):
    all_funcs = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".py"):
                path = os.path.join(root, filename)
                funcs = extract_functions_from_file(path)
                all_funcs.extend(funcs)
    return all_funcs


def get_embeddings(text_batch):
    response = openai.embeddings.create(
        model=EMBED_MODEL,
        input=[x["text"] for x in text_batch]
    )
    return [np.array(e.embedding, dtype=np.float32) for e in response.data]


def stream_to_faiss(functions, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    index_path = os.path.join(out_dir, "faiss.index")
    meta_path = os.path.join(out_dir, "metadata.jsonl")

    dim = 1536  # for text-embedding-3-small
    index = faiss.IndexFlatL2(dim)

    with open(meta_path, "w", encoding="utf-8") as meta_file:
        for i in range(0, len(functions), BATCH_SIZE):
            batch = functions[i:i + BATCH_SIZE]
            try:
                embeddings = get_embeddings(batch)
            except Exception as e:
                print(f"❌ Error during embedding batch {i}–{i + BATCH_SIZE}: {e}")
                continue

            index.add(np.vstack(embeddings))

            for j, func in enumerate(batch):
                func_record = {
                    "id": func["id"],
                    "source": func["source"],
                    "code": func["code"],
                    "doc": func["doc"]
                }
                meta_file.write(json.dumps(func_record) + "\n")

            print(f"✅ Embedded and saved batch {i}–{i + len(batch)}")

    faiss.write_index(index, index_path)
    print(f"💾 Saved FAISS index to {index_path}")
    print(f"💾 Metadata written to {meta_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parse Python codebase, embed functions, and stream to FAISS.")
    parser.add_argument("source_dir", help="Path to Python codebase")
    args = parser.parse_args()

    source_name = os.path.basename(os.path.abspath(args.source_dir.rstrip("/")))
    output_dir = os.path.join("vector", source_name)

    print(f"📂 Scanning {args.source_dir}...")
    funcs = walk_directory_and_extract(args.source_dir)
    print(f"📈 Found {len(funcs)} functions, embedding in batches of {BATCH_SIZE}...")

    stream_to_faiss(funcs, output_dir)
    print("✅ All done.")