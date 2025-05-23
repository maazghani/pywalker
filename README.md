# 🐍 PyWalker


This is my best guess at how GitHub Copilot "indexes" a repository. Create a vector database that indexes the functions of a codebase so it can use it as context for asking questions with contexts of entire codebases. 

`walker.py` uses [`jedi`](https://github.com/davidhalter/jedi) to walk a python codebase and generates embeddings using OpenAI's `text-embedding-3-small` out of the python functions and directory structure of the codebase. It then stores vectors in `FAISS` for local and fast semantic search, to be then used as retrieval-augmented context to ask GPT 4o questions pertaining to the codebase, and suggest either code improvements for said codebase, or generates code that would leverage said codebase(s). 

# How to Use

```bash
pip install -r requirements.txt
python -m venv . && source bin/activate

# set your OPEN AI API key:
export OPENAI_API_KEY="sk-...."
```

### 2. Index the Project
```bash
# clone a project
git clone https://github.com/fastapi/fastapi.git

# run pywalker
python walker.py ./fastapi
```

This creates a folder at `./vector/fastapi/` containing `faiss.index` and `metadata.jsonl`.

### 3. Ask a Question

```bash
python query.py "How does FastAPI define routes?" --codebase fastapi 
```
This will return just the raw embeddings.

Use `--answer` to get a GPT-4o-generated answer with the embeddings as context. 

---

## Example Queries

```text
How does FastAPI handle dependency injection?
```

---

