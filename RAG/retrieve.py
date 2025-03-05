import os
import faiss
import numpy as np
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# 🔹 Configurations
DATA_PATH = "Books/"  # Folder containing PDF files
EMBEDDING_MODEL = "all-mpnet-base-v2"

# 🔥 Load Sentence Transformer for Embeddings
embed_model = SentenceTransformer(EMBEDDING_MODEL)

# 🔹 Initialize FAISS Index
dimension = 768  
index = faiss.IndexFlatL2(dimension)
historical_texts = []  # Store text chunks for retrieval


# 📥 Load PDFs, Process & Store Embeddings
def load_and_index_pdfs():
    global historical_texts

    if not os.path.exists(DATA_PATH):
        print("📂 Books directory not found!")
        return

    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]

    if not pdf_files:
        print("❌ No PDF files found!")
        return

    all_documents = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(os.path.join(DATA_PATH, pdf_file))
        documents = loader.load()
        all_documents.extend(documents)

    # 🔹 Text Splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(all_documents)

    # 🔹 Convert text into embeddings
    historical_texts = [chunk.page_content for chunk in text_chunks]
    embeddings = np.array([embed_model.encode(text) for text in historical_texts])

    # 🔹 Store in FAISS Index
    index.add(embeddings)
    print(f"✅ Indexed {len(historical_texts)} text chunks!")

# 🔍 Retrieve Most Relevant Historical Context
def retrieve_relevant_text(query):
    query_embedding = embed_model.encode(query)
    _, index_results = index.search(np.array([query_embedding]), k=1)
    return historical_texts[index_results[0][0]]
