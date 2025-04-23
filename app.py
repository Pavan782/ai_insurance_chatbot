import google.generativeai as genai
import fitz  # PyMuPDF
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from google.genai import types

# Configure Gemini API Key
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

def extract_pdf_text(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def create_knowledge_base(text):
    # Split the text into manageable chunks
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents([Document(page_content=text)])

    embeddings = []
    for doc in docs:
        response = genai.generate_embeddings(
            model="gemini-embedding-exp-03-07",
            contents=doc.page_content
        )
        embeddings.append(response.embeddings)

    # Create FAISS vector store for searching
    vectorstore = FAISS.from_embeddings(embeddings, docs)
    return vectorstore

def ask_gemini_with_context(query, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion: {query}"

    # Use the Gemini model for text generation with the context
    model = genai.GenerativeModel(model_name='gemini-1.5-flash')
    response = model.generate_content(prompt)

    return response.text if response.text else "I'm not sure how to answer that. Let me connect you to our agent."

# Streamlit UI
st.set_page_config(page_title="Insurance Chatbot", page_icon="🤖")
st.title("🤖 AI Insurance Policy Chatbot")

uploaded_file = st.file_uploader("Upload Insurance PDF", type="pdf")
if uploaded_file:
    with st.spinner("Processing PDF..."):
        pdf_text = extract_pdf_text(uploaded_file)
        kb = create_knowledge_base(pdf_text)
    st.success("Knowledge base created!")

if 'chat' not in st.session_state:
    st.session_state.chat = []

user_query = st.text_input("Ask a question about your insurance policy:")

if user_query and uploaded_file:
    with st.spinner("Thinking..."):
        answer = ask_gemini_with_context(user_query, kb)
        st.session_state.chat.append(("You", user_query))
        st.session_state.chat.append(("Bot", answer))

for sender, msg in st.session_state.chat:
    st.markdown(f"**{sender}:** {msg}")
