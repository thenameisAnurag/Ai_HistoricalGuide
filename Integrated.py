import os
import faiss
import json
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# ✅ Paths
FAISS_INDEX_PATH = "faiss_index.bin"
TEXT_MAP_PATH = "faiss_text_map.json"
LOCAL_TEXT_FILE = "historical_data.txt"  # New: Local file for faster response
PDF_FOLDER = r"E:\Ai Muesuem Guide\Books"

# ✅ Museum & Historical Keywords (Filtering Allowed Questions)
museum_keywords = [
    "museum", "monument", "artifact", "historical", "history", "ancient", "heritage", "exhibit",
    "dynasty", "king", "queen", "ruler", "warrior", "battle", "fort", "palace", "kingdom", "inscription", "scripture",
    "mauryan", "ashoka", "bindusara", "chandragupta", "gupta", "samudragupta", "vikramaditya",
    "kushan", "kanishka", "satavahana", "pallava", "chola", "rashtrakuta", "chalukya",
    "maratha", "shivaji", "sambhaji", "shahu", "peshwa", "bajirao", "balaji vishwanath", "balaji baji rao",
    "nana saheb", "raghunath rao", "madhav rao", "holkar", "scindia", "bhonsle", "gaikwad",
    "mughal", "babur", "humayun", "akbar", "jahangir", "shah jahan", "aurangzeb", "bahadur shah",
    "prithviraj chauhan", "rana pratap", "rana sanga", "man singh", "guru gobind singh", "guru nanak",
    "ranjit singh", "mysore", "tipu sultan", "hyder ali", "nizam", "qutb shahi", "adil shahi",
    "battle of panipat", "battle of haldighati", "battle of plassey", "battle of buxar",
    "hampi", "ellora", "ajanta", "konark", "sanchi", "qutub minar", "taj mahal", "red fort",
    "charminar", "mysore palace", "amer fort", "gwalior fort", "chittorgarh",
    "coin", "script", "manuscript", "weapon", "sword", "shield", "armor", "horse", "elephant",
    "indus valley", "harappa", "mohenjo daro", "ashokan edicts", "sarasvati civilization"
]

# ✅ Step 1: Load Local Text File
def check_local_text_file(query):
    if not os.path.exists(LOCAL_TEXT_FILE):
        return None  # File doesn't exist yet

    with open(LOCAL_TEXT_FILE, "r", encoding="utf-8") as f:
        data = f.readlines()
    
    # Search for an exact match (or partial match)
    for line in data:
        if query.lower() in line.lower():
            return line.strip()  # Return the found data
    
    return None  # No match found in local file

# ✅ Step 2: Load PDFs & Extract Text
def load_pdfs_from_folder(folder_path=PDF_FOLDER):
    print(f"📖 Loading PDFs from '{folder_path}'...")
    pdf_texts = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            pdf_texts.extend(documents)
    print(f"✅ Loaded {len(pdf_texts)} PDF documents.")
    return pdf_texts

# ✅ Step 3: Store Embeddings in FAISS
def store_embeddings():
    pdf_texts = load_pdfs_from_folder()
    if not pdf_texts:
        print("⚠️ No PDFs loaded! Please check the folder path.")
        return

    print("🔠 Splitting Text into Chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(pdf_texts)

    if not chunks:
        print("⚠️ No valid text chunks found. Ensure PDFs contain readable text.")
        return

    print(f"✅ Split into {len(chunks)} text chunks.")

    # Initialize Embedding Model
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Generate embeddings
    embeddings = np.array([embeddings_model.embed_query(chunk.page_content) for chunk in chunks], dtype=np.float32)

    # Save FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"✅ {len(chunks)} embeddings stored in FAISS!")

    # Save text mapping
    text_map = {i: chunk.page_content for i, chunk in enumerate(chunks)}
    with open(TEXT_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(text_map, f, ensure_ascii=False, indent=4)
    
    print(f"✅ Text mapping saved to '{TEXT_MAP_PATH}'")

# ✅ Step 4: Retrieve Relevant Text from FAISS
def retrieve_relevant_text(query):
    print("🔍 Retrieving Relevant Context from FAISS...")

    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(TEXT_MAP_PATH):
        print("⚠️ FAISS index or text mapping not found! Run store_embeddings() first.")
        return None

    # Load FAISS index
    index = faiss.read_index(FAISS_INDEX_PATH)

    if index.ntotal == 0:
        print("⚠️ FAISS index is empty! No embeddings found.")
        return None

    # Load text map
    with open(TEXT_MAP_PATH, "r", encoding="utf-8") as f:
        text_map = json.load(f)

    # Generate query embedding
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = np.array([embeddings_model.embed_query(query)], dtype=np.float32)

    # Retrieve the closest match
    distances, retrieved_index = index.search(query_embedding, k=1)

    if retrieved_index is None or retrieved_index.shape[1] == 0 or retrieved_index[0][0] == -1:
        print("⚠️ No relevant match found in FAISS!")
        return None

    retrieved_idx = retrieved_index[0][0]

    # ✅ Ensure the retrieved index exists in our text map
    if str(retrieved_idx) not in text_map:
        print(f"⚠️ Invalid index {retrieved_idx}. No corresponding text found.")
        return None

    print(f"✅ FAISS retrieved match at index {retrieved_idx} with distance {distances[0][0]}")
    
    return text_map[str(retrieved_idx)]

# ✅ Step 5: Answer Question with Filters
def answer_question(question):
    # **Check Local File First**
    local_answer = check_local_text_file(question)
    if local_answer:
        return local_answer  # Return instantly if found

    # **Check Keywords**
    if not any(keyword in question.lower() for keyword in museum_keywords):
        return "❌ Sorry, I specialize in museums and Indian history. I can't answer that."

    # **Retrieve Context from FAISS**
    relevant_context = retrieve_relevant_text(question)
    if relevant_context is None:
        return "⚠️ No relevant historical data found. Try rephrasing your question."

    prompt = f"""You are a professional museum guide specializing in Indian history. 
    Use the following historical context to answer the question:

    Context: {relevant_context}

    Question: {question}
    
    Provide a concise and informative answer."""

    llm = Ollama(model="mistral")
    return llm(prompt)

# ✅ Step 6: Run the Model
def main():
    print("📖 Loading PDFs and Storing in FAISS...")
    if not os.path.exists(FAISS_INDEX_PATH):
        store_embeddings()

    while True:
        question = input("\n❓ Question: ")
        if question.lower() == "exit":
            break
        print("\n💡 Answer:", answer_question(question))

# Run the system
if __name__ == "__main__":
    main()
