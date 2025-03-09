import os
import faiss
import json
import numpy as np
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# File Paths
FAISS_INDEX_PATH = "faiss_index.bin"
TEXT_MAP_PATH = "faiss_text_map.json"
PDF_FOLDER = r"E:\Ai Muesuem Guide\Books"

# ✅ Supported Indian Languages for Translation
indian_languages = {
    "English": "English",
    "Hindi": "Hindi",
    "Bengali": "Bengali",
    "Telugu": "Telugu",
    "Marathi": "Marathi",
    "Tamil": "Tamil",
    "Urdu": "Urdu",
    "Gujarati": "Gujarati",
    "Malayalam": "Malayalam",
    "Kannada": "Kannada",
    "Odia": "Odia",
    "Punjabi": "Punjabi",
    "Assamese": "Assamese"
}

# ✅ Strictly Historical Keywords (No AI/ML topics)
history_keywords = {
    "museum", "monument", "artifact", "historical", "history", "heritage", "exhibit",
    "shivaji", "sambhaji", "maharana pratap", "chetak", "battle of haldighati",
    "maratha", "mughal", "rajput", "akbar", "peshwa", "gupta", "ashoka", "buddha"
}

# ✅ Load PDFs & Extract Text
def load_pdfs_from_folder(folder_path=PDF_FOLDER):
    pdf_texts = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            pdf_texts.extend(documents)
    return pdf_texts

# ✅ Store Text in FAISS
def store_embeddings():
    pdf_texts = load_pdfs_from_folder()
    if not pdf_texts:
        return False

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(pdf_texts)
    if not chunks:
        return False

    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = np.array([embeddings_model.embed_query(chunk.page_content) for chunk in chunks], dtype=np.float32)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)

    text_map = {i: chunk.page_content for i, chunk in enumerate(chunks)}
    with open(TEXT_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(text_map, f, ensure_ascii=False, indent=4)

    return True

# ✅ Retrieve Text from FAISS
def retrieve_relevant_text(query):
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(TEXT_MAP_PATH):
        return None

    index = faiss.read_index(FAISS_INDEX_PATH)
    if index.ntotal == 0:
        return None

    with open(TEXT_MAP_PATH, "r", encoding="utf-8") as f:
        text_map = json.load(f)

    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = np.array([embeddings_model.embed_query(query)], dtype=np.float32)

    distances, retrieved_index = index.search(query_embedding, k=1)

    if retrieved_index[0][0] == -1 or str(retrieved_index[0][0]) not in text_map:
        return None

    return text_map[str(retrieved_index[0][0])]

# ✅ Check Keywords Before Answering
def check_keywords(question):
    return any(keyword.lower() in question.lower() for keyword in history_keywords)

# ✅ Generate Historical Prompt
def generate_history_prompt(question, relevant_context):
    base_prompt = f"""
    You are a historian and museum expert specializing in Indian history.
    - Respond **only** if the topic is historical.
    - Use **verified sources** (FAISS PDF data or approved history keywords).
    - If no historical context is found, reject the question.

    **HISTORICAL CONTEXT (if available):**
    {relevant_context if relevant_context else "⚠️ No direct reference found."}

    **USER QUESTION:**
    {question}

    **YOUR RESPONSE FORMAT:**
    - If relevant history is found, provide an accurate answer.
    - Otherwise, say: **"❌ This is not my expertise. I only provide historical knowledge."**
    """
    return base_prompt

# ✅ Answer Question using FAISS, Keywords, or Reject
def answer_question(question):
    relevant_context = retrieve_relevant_text(question)
    if relevant_context:
        strict_prompt = generate_history_prompt(question, relevant_context)
        llm = Ollama(model="mistral")
        return llm(strict_prompt)

    if check_keywords(question):
        strict_prompt = generate_history_prompt(question, relevant_context=None)
        llm = Ollama(model="mistral")
        return llm(strict_prompt)

    return "❌ This is not my expertise."

# ✅ Translate Response
def translate_response(model, response_text, selected_language):
    if selected_language == "English":
        return response_text  # No translation needed

    prompt = f"""Translate the following historical response into {selected_language}:
    
    Text: {response_text}
    
    Provide an accurate translation while keeping historical accuracy."""
    
    return model(prompt)

# ✅ Summarize Response
def summarize_response(model, response_text):
    prompt = f"""Summarize the following historical text in simple terms:
    
    Text: {response_text}
    
    Provide a concise summary."""
    
    return model(prompt)

# ✅ Streamlit UI
def main():
    st.set_page_config(page_title="🏛️ Indian History Guide", page_icon="📜", layout="wide")
    st.title("🏛️ Indian History AI Guide")
    st.subheader("📜 Ask about historical figures, battles, monuments & more!")

    if not os.path.exists(FAISS_INDEX_PATH):
        with st.spinner("📖 Indexing historical texts..."):
            if store_embeddings():
                st.success("✅ FAISS embeddings created successfully!")
            else:
                st.error("⚠️ Failed to load PDFs. Check the file path!")

    user_query = st.text_input("❓ Ask a history-related question:")

    if st.button("Get Answer"):
        if user_query.strip():
            response = answer_question(user_query)
            if response.startswith("❌"):
                st.error(response)
            else:
                st.session_state["latest_response"] = response  # Store response
                st.subheader("📜 Answer:")
                st.write(response)
        else:
            st.warning("⚠️ Please enter a valid question.")

    # Show buttons for Summarization and Translation after response is generated
    if "latest_response" in st.session_state:
        response_text = st.session_state["latest_response"]

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Summarize Response"):
                summarized_text = summarize_response(Ollama(model="mistral"), response_text)
                st.subheader("📝 Summarized Text:")
                st.write(summarized_text)
                st.session_state["latest_response"] = summarized_text

        with col2:
            selected_language = st.selectbox("🌍 Translate Response To:", list(indian_languages.keys()), key="lang_select")
            if st.button("Translate Response"):
                translated_text = translate_response(Ollama(model="mistral"), response_text, selected_language)
                st.subheader(f"🌍 Translated Text ({selected_language}):")
                st.write(translated_text)

    st.markdown("---")
    st.markdown("📌 **Note:** This AI answers only history-related queries. Off-topic questions will be rejected.")

if __name__ == "__main__":
    main()
