import os
import json
import jedi
import faiss
import openai
import numpy as np
import argparse

MAX_LENGTH = 7500
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 50

openai.api_key = os.getenv("OPENAI_API_KEY")


def summarize_file_head(raw_code, head_lines=20):
    lines = raw_code.splitlines()
    head = [line for line in lines[:head_lines] if line.strip()]
    return "\n".join(head)


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
                "type": "function",
                "code": prompt,
                "doc": completion,
                "source": filepath
            })
        except Exception as e:
            print(f"⚠️ Skipping malformed function in {filepath}: {e}")
            continue

    return functions


def walk_directory_and_extract(directory):
    all_entries = []

    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".py"):
                path = os.path.join(root, filename)
                rel_path = os.path.relpath(path, directory)

                try:
                    with open(path, "r", encoding="utf-8") as f:
                        raw = f.read()
                        head_summary = summarize_file_head(raw)
                        all_entries.append({
                            "id": f"{rel_path}",
                            "text": f"File path: {rel_path}\n{head_summary}",
                            "type": "file",
                            "source": path
                        })
                except Exception as e:
                    print(f"⚠️ Could not read file {path}: {e}")

                all_entries.extend(extract_functions_from_file(path))

    return all_entries


def get_embeddings(text_batch):
    response = openai.embeddings.create(
        model=EMBED_MODEL,
        input=[x["text"] for x in text_batch]
    )
    return [np.array(e.embedding, dtype=np.float32) for e in response.data]


def stream_to_faiss(entries, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    index_path = os.path.join(out_dir, "faiss.index")
    meta_path = os.path.join(out_dir, "metadata.jsonl")

    dim = 1536  # for text-embedding-3-small
    index = faiss.IndexFlatL2(dim)

    with open(meta_path, "w", encoding="utf-8") as meta_file:
        for i in range(0, len(entries), BATCH_SIZE):
            batch = entries[i:i + BATCH_SIZE]
            try:
                embeddings = get_embeddings(batch)
            except Exception as e:
                print(f"❌ Error during embedding batch {i}–{i + BATCH_SIZE}: {e}")
                continue

            index.add(np.vstack(embeddings))

            for j, entry in enumerate(batch):
                record = {
                    "id": entry["id"],
                    "type": entry["type"],
                    "source": entry["source"]
                }
                if entry["type"] == "function":
                    record["code"] = entry["code"]
                    record["doc"] = entry["doc"]
                meta_file.write(json.dumps(record) + "\n")

            print(f"✅ Embedded and saved batch {i}–{i + len(batch)}")

    faiss.write_index(index, index_path)
    print(f"💾 Saved FAISS index to {index_path}")
    print(f"💾 Metadata written to {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index a Python codebase with code and file embeddings.")
    parser.add_argument("source_dir", help="Path to the Python codebase")
    args = parser.parse_args()

    source_name = os.path.basename(os.path.abspath(args.source_dir.rstrip("/")))
    output_dir = os.path.join("vector", source_name)

    print(f"📂 Scanning {args.source_dir}...")
    entries = walk_directory_and_extract(args.source_dir)
    print(f"📈 Found {len(entries)} items (functions + files), embedding in batches of {BATCH_SIZE}...")

    stream_to_faiss(entries, output_dir)
    print("✅ All done.")
