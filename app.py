import streamlit as st
import google.generativeai as genai
import fitz  # PyMuPDF
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.base import Embeddings

# Configure Gemini API
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Custom Embeddings Wrapper for Gemini
class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )["embedding"] for text in texts]

    def embed_query(self, text):
        return genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_query"
        )["embedding"]

# Extract text from PDF
def extract_pdf_text(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "".join(page.get_text() for page in doc)

# Create vectorstore using Gemini embeddings
def create_knowledge_base(text):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents([Document(page_content=text)])
    texts = [doc.page_content for doc in docs]
    embeddings = GeminiEmbeddings()

    try:
        vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Vectorstore creation error: {e}")
        return None

# Query the chatbot with context
def ask_gemini_with_context(query, vectorstore):
    if vectorstore is None:
        return "Error: Knowledge base not initialized."

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    prompt = f"Answer based on the context:\n{context}\n\nQuestion: {query}"

    model = genai.GenerativeModel(model_name='gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text or "Sorry, I couldn't find the answer. Let me connect you to a human agent."

# Streamlit UI
st.set_page_config(page_title="Insurance Chatbot", page_icon="🤖")
st.title("🤖 AI Insurance Policy Chatbot")

uploaded_file = st.file_uploader("Upload Insurance PDF", type="pdf")
if uploaded_file:
    with st.spinner("Processing PDF..."):
        pdf_text = extract_pdf_text(uploaded_file)
        kb = create_knowledge_base(pdf_text)
        if kb:
            st.success("Knowledge base created!")
            st.session_state.knowledge_base = kb
        else:
            st.error("Failed to create knowledge base.")

if "chat" not in st.session_state:
    st.session_state.chat = []
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None

user_query = st.text_input("Ask a question about your insurance policy:")
if user_query and uploaded_file and st.session_state.knowledge_base:
    with st.spinner("Thinking..."):
        answer = ask_gemini_with_context(user_query, st.session_state.knowledge_base)
        st.session_state.chat.append(("You", user_query))
        st.session_state.chat.append(("Bot", answer))

for sender, msg in st.session_state.chat:
    st.markdown(f"**{sender}:** {msg}")
