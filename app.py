import streamlit as st
import os

from ingest import ingest_pdf
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

st.title("📚 AI Academic Assistant")

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = Ollama(model="gemma3:1b")

DATA_FOLDER = "data"
VECTOR_FOLDER = "vector_store"

# Get existing subjects
subjects = os.listdir(DATA_FOLDER) if os.path.exists(DATA_FOLDER) else []

st.sidebar.header("Upload Study Material")

subject = st.sidebar.selectbox("Select Subject", subjects)

new_subject = st.sidebar.text_input("Or Create New Subject")

uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

if st.sidebar.button("Upload"):

    if new_subject:
        subject = new_subject
        os.makedirs(f"data/{subject}", exist_ok=True)
        os.makedirs(f"vector_store/{subject}", exist_ok=True)

    if uploaded_file:

        file_path = f"data/{subject}/{uploaded_file.name}"

        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        st.sidebar.success("PDF uploaded")

        chunks = ingest_pdf(subject)

        st.sidebar.success(f"{chunks} chunks indexed")

st.header("Ask Questions")

query = st.text_input("Enter your question")

if st.button("Ask"):

    vectordb = Chroma(
        persist_directory=f"vector_store/{subject}",
        embedding_function=embedding
    )

    retriever = vectordb.as_retriever(search_kwargs={"k":3})

    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer ONLY using the context below.

Context:
{context}

Question:
{query}
"""

    answer = llm.invoke(prompt)

    st.write("### Answer")
    st.write(answer)