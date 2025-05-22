# 🐍 PyWalker

**PyWalker** is a Python toolset for indexing and querying codebases using OpenAI embeddings and FAISS vector search.

It scans a Python project, extracts all functions (with docstrings), generates embeddings using `text-embedding-3-small`, and stores them in a searchable vector database. You can then ask natural language questions about the codebase and retrieve relevant code + answers powered by GPT-4o.

---

## 🚀 Features

- ✅ Parse Python functions with [`jedi`](https://github.com/davidhalter/jedi)  
- ✅ Generate embeddings with OpenAI's `text-embedding-3-small`  
- ✅ Store vectors in `FAISS` for local, fast semantic search  
- ✅ Ask questions using GPT-4o with retrieval-augmented context  
- ✅ Supports multiple codebases via subdirectory indexing

---

## 📦 Requirements

```bash
pip install openai faiss-cpu jedi numpy
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY=sk-...
```

---

## 📂 Project Structure

```
pywalker/
├── walker.py       # Indexes a Python codebase
├── query.py        # Searches and queries an indexed codebase
└── vector/         # Contains FAISS indices and metadata per codebase
    └── fastapi/
        ├── faiss.index
        └── metadata.jsonl
```

---

## 🛠️ Usage

### 1. Clone a Python Project

```bash
git clone https://github.com/fastapi/fastapi.git
```

### 2. Index the Project

```bash
python walker.py ./fastapi
```

This creates a folder at `./vector/fastapi/` containing `faiss.index` and `metadata.jsonl`.

### 3. Ask a Question

```bash
python query.py "How does FastAPI define routes?" --codebase fastapi --answer
```

Use `--answer` to get a GPT-4o-generated response based on the most relevant code snippets.

---

## 🧠 Example Queries

```text
What operations are supported on quaternions in sympy?
How does FastAPI handle dependency injection?
Where is the __mul__ method defined in the Quaternion class?
```

---

## 📌 Notes

- Embedding model used: `text-embedding-3-small` (cheap + fast)
- GPT model used: `gpt-4o` (or switch to `gpt-3.5-turbo`)
- Vector search: `FAISS.IndexFlatL2`
- Output files per project: `faiss.index` + `metadata.jsonl`