import warnings
import os
import streamlit as st
from dotenv import load_dotenv
import fitz
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from html_files import css, bot_template, user_template

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load environment variables at startup
load_dotenv()

# LLM settings
LLM_MODEL = "gpt-4o-mini"


def get_pdf_documents(pdf_docs):
    """Extract text per page and attach source + page metadata."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    documents = []
    empty_files = []

    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
        except Exception:
            empty_files.append(pdf.name)
            continue

        has_text = False
        for page_index, page in enumerate(pdf_reader.pages, start=1):
            page_text = page.extract_text() or ""
            if not page_text.strip():
                continue

            has_text = True
            page_docs = text_splitter.create_documents(
                [page_text],
                metadatas=[{"source": pdf.name, "page": page_index}]
            )
            documents.extend(page_docs)

        if not has_text:
            empty_files.append(pdf.name)

    return documents, empty_files


def get_vectorstore(documents):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model=LLM_MODEL)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True, output_key="answer")
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True
    )
    return conversation_chain


def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.error("Please upload and process PDFs first!")
        return

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    st.session_state.last_sources = response.get('source_documents', [])


def main():
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []
    if "pdf_bytes" not in st.session_state:
        st.session_state.pdf_bytes = {}
    if "page_images" not in st.session_state:
        st.session_state.page_images = {}

    def handle_question():
        if st.session_state.question_input:
            handle_userinput(st.session_state.question_input)
            st.session_state.last_question = st.session_state.question_input
            st.session_state.question_input = ""

    st.header("Chat with multiple PDFs :books:")

    if st.session_state.conversation is None:
        st.info("Upload and process your PDFs to enable the search bar.")
    else:
        st.text_input(
            "Ask a question about your documents:",
            key="question_input",
            on_change=handle_question
        )

    st.markdown("---")

    if st.session_state.conversation is not None and st.session_state.chat_history:
        for i, message in enumerate(reversed(st.session_state.chat_history)):
            if i % 2 == 0:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)

        if st.session_state.last_sources:
            st.markdown("---")
            st.subheader("Information Source")
            doc = st.session_state.last_sources[0]
            source_name = doc.metadata.get('source', 'Unknown')
            page_number = doc.metadata.get('page')
            if page_number:
                st.info(f"**Arquivo:** {source_name} (page {page_number})")
            else:
                st.info(f"**Arquivo:** {source_name}")

            if page_number and source_name in st.session_state.pdf_bytes:
                cache_key = (source_name, page_number)
                image_bytes = st.session_state.page_images.get(cache_key)
                if image_bytes is None:
                    try:
                        pdf_stream = st.session_state.pdf_bytes[source_name]
                        pdf_doc = fitz.open(stream=pdf_stream, filetype="pdf")
                        page = pdf_doc.load_page(page_number - 1)
                        pix = page.get_pixmap(dpi=150)
                        image_bytes = pix.tobytes("png")
                        st.session_state.page_images[cache_key] = image_bytes
                    except Exception:
                        image_bytes = None
                    finally:
                        try:
                            pdf_doc.close()
                        except Exception:
                            pass
                if image_bytes:
                    st.image(image_bytes, caption=f"{source_name} - page {page_number}")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file!")
                return

            with st.spinner("Processing"):
                st.session_state.pdf_bytes = {pdf.name: pdf.getvalue() for pdf in pdf_docs}
                st.session_state.page_images = {}

                documents, empty_files = get_pdf_documents(pdf_docs)
                if empty_files:
                    st.warning("No extractable text found in: " + ", ".join(empty_files))
                if not documents:
                    st.error("No text could be extracted from the uploaded PDFs.")
                    return

                vectorstore = get_vectorstore(documents)

                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
                st.session_state.vectorstore = vectorstore
                st.success("Processing complete! You can now ask questions about your documents.")
                st.rerun()

if __name__ == '__main__':
    main()
