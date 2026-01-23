# AI_Agent_for_Your_Files

> **Status: Archived**  
> This repository is archived and kept for historical reference and learning purposes. It is not actively maintained.

---

## Overview

A lightweight **RAG (Retrieval-Augmented Generation)** Streamlit app that lets you **upload multiple PDFs and chat with all of them**, including:
- page-aware answers (reference file + page)
- visual preview of the source PDF page under each response

---

## Design goal (why “single-file”)

From the start, the intent of this project was to keep the core implementation in **one main file (`app.py`)** to:
- simplify the deployment surface for **cloud hosting**
- reduce operational complexity for quick experiments
- study the trade-offs of a “single-file RAG app” structure (clarity vs. maintainability, speed vs. modularity)

This repo is intentionally small and pragmatic: a compact reference implementation rather than a scalable framework.

---

## Features

- Upload multiple PDFs and ask questions about their content
- Page-aware retrieval (answers can point to the original file/page)
- Visual PDF page preview under the assistant response
- Guided UI flow: upload → process → ask

---

## Tech stack

- `streamlit` — UI
- `python-dotenv` — environment variables
- `PyPDF2` — PDF text extraction
- `langchain` / `langchain-openai` — LLM + embeddings + conversational chain
- `langchain-community` + `faiss-cpu` — vector store and similarity search
- `PyMuPDF` — render PDF pages as images

---

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
- Not designed for large document collections, multi-user concurrency, or production hardening.
