# Reference RAG Project

A retrieval-augmented generation (RAG) pipeline that answers questions from your own PDF documents. It loads and chunks PDFs, embeds the chunks, stores them in a persistent ChromaDB vector store, retrieves the most relevant chunks for a question, and passes them to an LLM hosted on Hugging Face to generate the answer.

## How it works

1. **Load**: `PyPDFLoader` reads every PDF under `data/pdf/` (one document per page, with source file and page metadata).
2. **Chunk**: `RecursiveCharacterTextSplitter` splits pages into 1000-character chunks with 200 characters of overlap.
3. **Embed**: `EmbeddingManager` encodes chunks with the `all-MiniLM-L6-v2` SentenceTransformer model (384 dimensions).
4. **Store**: `VectorStore` saves the chunks and embeddings in a persistent ChromaDB collection (`pdf_documents`) at `data/vector_store/`.
5. **Retrieve**: `VectorStore.retrieve()` embeds the query and returns the top-k closest chunks.
6. **Generate**: `rag_simple()` builds a prompt from the retrieved context and sends it to the LLM. If the answer isn't in the context, the model is instructed to say it doesn't know.

## Tech stack

- Python 3.12
- LangChain (document loading, text splitting, `ChatOpenAI` client)
- pypdf
- sentence-transformers (`all-MiniLM-L6-v2`)
- ChromaDB
- Hugging Face Inference Router (OpenAI-compatible endpoint)

## Project structure

```text
RAG/
├── data/
│   ├── pdf/             # put your PDFs here (not tracked by git)
│   └── vector_store/    # generated ChromaDB files (not tracked by git)
├── notebook/
│   └── pdf_loader.ipynb # the full pipeline
├── .env                 # your secrets (not tracked by git)
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

1. **Clone the repo and create a virtual environment**

```powershell
   git clone https://github.com/Kishor-J-K/Reference_RAG_projrct.git
   cd Reference_RAG_projrct
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
```

2. **Add your PDFs**

   Create `data/pdf/` if it doesn't exist and place your PDF files in it. The PDFs are not included in this repository.

3. **Configure the LLM**

   Create a `.env` file in the project root:

```text
   model_name=<a chat model available on the Hugging Face router>
   Token=<your Hugging Face access token>
```

   You can create a token at https://huggingface.co/settings/tokens. Never commit this file.

## Usage

1. Open `notebook/pdf_loader.ipynb` and run the cells from top to bottom. The notebook uses relative paths (`../data`), so it must be run from the `notebook/` folder.
2. The ingest cell embeds your PDFs and fills the vector store. It only adds documents when the collection is empty, so re-running it won't create duplicates. The first run takes a little while because every chunk is embedded.
3. Run the final cell, type a question when prompted, and read the answer.

After the first ingest, the vector store persists on disk. On later sessions you can skip loading, chunking and ingesting, and just run the imports, the `EmbeddingManager` and `VectorStore` definitions, and the retrieval and LLM cells.

## Troubleshooting

- **"No relevant context found"**: the vector store is empty or the query ran before ingest. Check `vectorstore.collection.count()`. It should be greater than 0.
- **Found 0 PDF files**: confirm your PDFs are in `data/pdf/` and that you're running the notebook from the `notebook/` folder.
- **LLM errors**: check that `model_name` and `Token` are set in `.env` and that the token has permission to call the Inference Router.

## Possible improvements

- Show source file and page citations alongside answers
- Return similarity scores and filter weak matches
- Add a simple UI (Streamlit or Gradio)
- Move the pipeline from the notebook into a Python package
