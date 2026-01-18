import warnings
import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from html_files import css, bot_template, user_template

# Suprimir warnings desnecessários
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Carregar variáveis de ambiente no início
load_dotenv()


def get_pdf_text(pdf_docs):
    """Efficiently extract text from PDFs and track source files"""
    all_text = ""
    chunk_metadatas = []
    pdf_dict = {}  # Map PDF name to its content
    
    # Read all PDFs once and store their content
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        pdf_text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                pdf_text += page_text
        pdf_dict[pdf.name] = pdf_text
        all_text += pdf_text
    
    # Create chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(all_text)
    
    # Map chunks to source files more efficiently
    for chunk in chunks:
        # Find which PDF this chunk belongs to
        found = False
        for pdf_name, pdf_content in pdf_dict.items():
            if chunk in pdf_content:
                chunk_metadatas.append({"source": pdf_name})
                found = True
                break
        if not found:
            # Fallback to first PDF if not found
            first_pdf_name = next(iter(pdf_dict.keys())) if pdf_dict else "Unknown"
            chunk_metadatas.append({"source": first_pdf_name})
    
    return chunks, chunk_metadatas


def get_vectorstore(text_chunks, chunk_metadatas):
    embeddings = OpenAIEmbeddings()
    # Ensure every chunk has metadata
    for i, chunk in enumerate(text_chunks):
        if i >= len(chunk_metadatas):
            chunk_metadatas.append({"source": "Unknown"})
    
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings, metadatas=chunk_metadatas)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.error("Please upload and process PDFs first!")
        return
    
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    # Get source documents using the retriever
    source_documents = st.session_state.vectorstore.similarity_search(user_question, k=1)

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
    
    # Display source file
    if source_documents:
        st.markdown("---")
        st.subheader("📄 Fonte da Informação")
        doc = source_documents[0]
        source_name = doc.metadata.get('source', 'Unknown')
        st.info(f"**Arquivo:** {source_name}")
        with st.expander("Ver trecho completo"):
            st.write(doc.page_content)


def main():
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file!")
                return
            
            with st.spinner("Processing"):
                # get pdf text and chunks
                text_chunks, chunk_metadatas = get_pdf_text(pdf_docs)

                # create vector store
                vectorstore = get_vectorstore(text_chunks, chunk_metadatas)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
                st.session_state.vectorstore = vectorstore
                st.success("Processing complete! You can now ask questions about your documents.")

if __name__ == '__main__':
    main()