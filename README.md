# AI Agent for Your Files (RAG PDF Chat)

This project is a lightweight Retrieval-Augmented Generation (RAG) app that lets you chat with multiple PDFs through a Streamlit UI. It keeps the code in a single file (`app.py`) to make cloud deployment cheap and fast.

## Features
- Upload multiple PDFs and ask questions about their content.
- Page-aware retrieval (the answer can reference the source file and page).
- Visual preview of the source PDF page under each bot response.
- Simple UI with a guided flow (upload -> process -> ask).

## Tech Stack / Libraries
- `streamlit` - web UI
- `python-dotenv` - environment variables
- `PyPDF2` - PDF text extraction
- `langchain` - conversational chain and memory
- `langchain-openai` - OpenAI LLM + embeddings
- `langchain-community` - FAISS vector store integration
- `faiss-cpu` - vector similarity search backend (required by FAISS)
- `PyMuPDF` - render PDF pages as images

## Setup
1. Create a virtual environment (optional but recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your OpenAI key:

```env
OPENAI_API_KEY=your_key_here
```

## Run
```bash
streamlit run app.py
```

## Notes
- Everything is intentionally kept in `app.py` to simplify deployment on low-cost cloud hosting.
- The app extracts per-page text and stores page metadata so you can identify where answers come from.
- Source pages are rendered from the original PDFs to provide visual context.
