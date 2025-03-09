import os
import faiss
import json
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# ✅ File Paths
FAISS_INDEX_PATH = "faiss_index.bin"
TEXT_MAP_PATH = "faiss_text_map.json"
PDF_FOLDER = r"E:\Ai Muesuem Guide\Books"

# ✅ Step 1: Load All PDFs and Extract Text
def load_pdfs_from_folder():
    pdf_texts = []
    
    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, file)
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            pdf_texts.extend(documents)

    return pdf_texts

# ✅ Step 2: Rebuild FAISS Index (Force Refresh)
def rebuild_faiss_index():
    print("🔄 Rebuilding FAISS Index...")
    pdf_texts = load_pdfs_from_folder()
    
    if not pdf_texts:
        print("⚠️ No PDFs found!")
        return False

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(pdf_texts)

    if not chunks:
        print("⚠️ No valid text chunks found.")
        return False

    print(f"✅ Split into {len(chunks)} text chunks.")

    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = np.array([embeddings_model.embed_query(chunk.page_content) for chunk in chunks], dtype=np.float32)

    # ✅ Create a NEW FAISS index (overwrite old one)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)

    print(f"✅ {len(chunks)} embeddings stored in FAISS!")

    # ✅ Overwrite `faiss_text_map.json` with new data
    text_map = {i: chunk.page_content for i, chunk in enumerate(chunks)}
    with open(TEXT_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(text_map, f, ensure_ascii=False, indent=4)

    print(f"✅ FAISS Text Map Updated with {len(text_map)} entries!")
    return True

# ✅ Step 3: Retrieve Relevant Context from FAISS
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

# ✅ Step 4: Generate Historical Prompt
def generate_history_prompt(question, relevant_context):
      return f"""
    You are a historian specializing in Indian history.
    - Answer **only** if the topic is historical.
    - Use **verified sources** (FAISS data or history-related keywords).
    - If no historical context is found, reject the question.

    **HISTORICAL CONTEXT (if available):**
    {relevant_context if relevant_context else "⚠️ No direct reference found."}

    **USER QUESTION:**
    {question}

    **YOUR RESPONSE:**
    - If relevant history is found, provide an accurate answer.
    - Otherwise, say: **"❌ This is not my expertise. I only provide historical knowledge."**
    """

# ✅ Step 5: Answer Question using FAISS, or Reject
def answer_question(question):
    relevant_context = retrieve_relevant_text(question)
    if relevant_context:
        prompt = generate_history_prompt(question, relevant_context)
        llm = Ollama(model="mistral")
        return llm(prompt)

    return "❌ This is not my expertise."

# ✅ Step 6: CLI for Local Testing
def main():
    print("\n📖 AI Museum Guide (Local Mode)")
    print("🔄 Refreshing FAISS Index...")
    rebuild_faiss_index()

    print("\n💬 Ask me anything about Indian History (type 'exit' to stop):")
    
    while True:
        question = input("\n❓ Question: ")
        if question.lower() == "exit":
            break

        answer = answer_question(question)
        print("\n💡 Answer:", answer)

if __name__ == "__main__":
    main()
