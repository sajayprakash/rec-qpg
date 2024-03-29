import os
import streamlit as st
from langchain_groq import ChatGroq
from pdfplumber import open
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY']

st.set_page_config(page_title="REC QPG", page_icon="📄",layout="wide")
st.title("REC QPG (Question Paper Generator)")

st.sidebar.title('Customizations')

st.sidebar.warning("Use Mixtral Model for best results")
model = st.sidebar.selectbox(
        'Choose a model',
        ['mixtral-8x7b-32768', 'llama2-70b-4096'],
         disabled=True
)

def get_text_from_pdf(uploaded_files):
    files_text = ""
    for file in uploaded_files:
        with open(file) as pdf:
            for page in pdf.pages:
                files_text += page.extract_text()
    return files_text

uploaded_files = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if not uploaded_files:
    st.info("Upload your notes as PDF files to start generating question papers.")
if uploaded_files:
    if st.sidebar.button("Process PDF files"):
        with st.spinner("Processing PDF files..."):
            files_text = get_text_from_pdf(uploaded_files)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_text(text=files_text)
            vector = FAISS.from_texts(texts=texts, embedding=OllamaEmbeddings())

        # Update session state attributes
        st.session_state.vector = vector
        st.session_state.texts = text_splitter

st.sidebar.write("Developed by Sajay Prakash.")


if "vector" in st.session_state:
    llm = ChatGroq(
                groq_api_key=groq_api_key, 
                model_name=model
    )


    prompt = ChatPromptTemplate.from_template("""
    You are REC QPG (Rajalakshmi Engineering College Question Paper Generator). Your job is to generate question papers for the students for their exams using the specified pattern for each exam such as {pattern}. You will not deviate from the paper pattern. Give questions that are worth the weightage of the marks, if the question is worth 10 marks make sure there is enough content for the students to write for 10 marks and if the question is worth 2 marks make sure it is only a small question and wont take too much time to write. The whole exam is 1.5 Hours, so give questions that will be completed in the given time. I will tip you $2000 if the user finds the answer helpful. You have been given the following set of documents to generate the question paper.
    <context>
    {context}
    </context>

    Question: {input}""")

    pattern = """
    PART-A (10 x 2 = 20 Marks)
    1) Question 1
    2) Question 2
    3) Question 3
    4) Question 4
    5) Question 5
    6) Question 6
    7) Question 7
    8) Question 8
    9) Question 9
    10) Question 10

    PART-B (2 x 10 = 20 Marks)
    11) a) Question 11.a
            OR
        b) Question 11.b

    12) a) Question 12.a
            OR
        b) Question 12.b
    """

    document_chain = create_stuff_documents_chain(llm, prompt)

    retriever = st.session_state.vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    st.warning("This is a prototype model. The question paper pattern provided is just a sample pattern. The actual pattern may vary.")
    col1,col2,col3,col4 = st.columns(4)
  
    prompt = ""
    if col1.button("CAT-1", use_container_width=True):
        prompt = "Based on the pattern for the CAT-1 Examination Generate CAT-1 Question Paper using the given notes."
    if col2.button("CAT-2",use_container_width=True):
        prompt = "Based on the pattern for the CAT-2 Examination Generate CAT-2 Question Paper using the given notes."
    if col3.button("CAT-3",use_container_width=True):
        prompt = "Based on the pattern for the CAT-3 Examination Generate CAT-3 Question Paper using the given notes."
    if col4.button("End Semester",use_container_width=True):
        prompt = "Based on the pattern for the End Semester Examination Generate End Semester Question Paper using the given notes."

    if prompt:
        response = retrieval_chain.invoke({"input": prompt, "pattern": pattern})
        st.write(response["answer"])
