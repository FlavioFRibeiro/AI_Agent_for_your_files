import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings#, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from html_files import css, bot_template, user_template
#from langchain.llms import HuggingFaceHub

def get_pdf_text(pdf_docs):
    text = ""
    pdf_pages_info = []
    current_pos = 0
    
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages, 1):
            page_text = page.extract_text()
            text += page_text
            # Store where this page starts and ends in the concatenated text
            pdf_pages_info.append({
                "source": pdf.name,
                "page": page_num,
                "start_pos": current_pos,
                "end_pos": current_pos + len(page_text)
            })
            current_pos += len(page_text)
    
    return text, pdf_pages_info


def get_text_chunks(text, pdf_pages_info):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Map each chunk to its source page
    chunk_metadatas = []
    char_pos = 0
    
    for chunk in chunks:
        # Find which page this chunk's starting position falls into
        chunk_start = char_pos
        chunk_end = char_pos + len(chunk)
        
        # Find the page that contains this chunk's start
        page_info = None
        for info in pdf_pages_info:
            if info["start_pos"] <= chunk_start < info["end_pos"]:
                page_info = info
                break
        
        if page_info is None and pdf_pages_info:
            # Fallback: use the last page
            page_info = pdf_pages_info[-1]
        
        if page_info:
            chunk_metadatas.append({
                "source": page_info["source"],
                "page": page_info["page"]
            })
        
        char_pos = chunk_end
    
    return chunks, chunk_metadatas


def get_vectorstore(text_chunks, chunk_metadatas):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    # Ensure every chunk has metadata
    for i, chunk in enumerate(text_chunks):
        if i >= len(chunk_metadatas):
            chunk_metadatas.append({"source": "Unknown", "page": "N/A"})
    
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings, metadatas=chunk_metadatas)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

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
    
    # Get source documents using the retriever - retrieve only 1 to get the most relevant
    source_documents = st.session_state.vectorstore.similarity_search(user_question, k=1)

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
    
    # Display source documents
    if source_documents:
        st.markdown("---")
        st.subheader("📄 Fontes de Informação")
        for i, doc in enumerate(source_documents, 1):
            source_name = doc.metadata.get('source', 'Unknown')
            page_num = doc.metadata.get('page', 'N/A')
            with st.expander(f"Fonte {i}: {source_name} (Página {page_num})"):
                st.write(doc.page_content)


def main():
    load_dotenv()
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
            with st.spinner("Processing"):
                # get pdf text
                raw_text, pdf_pages_info = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks, chunk_metadatas = get_text_chunks(raw_text, pdf_pages_info)

                # create vector store
                vectorstore = get_vectorstore(text_chunks, chunk_metadatas)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
                st.session_state.vectorstore = vectorstore
                st.success("Processing complete! You can now ask questions about your documents.")

if __name__ == '__main__':
    main()