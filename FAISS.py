import os
import faiss
import json
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# File Paths
FAISS_INDEX_PATH = "faiss_index.bin"
TEXT_MAP_PATH = "faiss_text_map.json"
PDF_FOLDER = r"E:\Ai Muesuem Guide\Books"

# ✅ Strictly Historical Keywords (NO AI-RELATED TERMS)
history_keywords = {
   # General terms
    "museum", "monument", "artifact", "historical", "history", "ancient", "heritage", "exhibit",
    "dynasty", "king", "queen", "ruler", "warrior", "battle", "fort", "palace", "kingdom", "inscription", "scripture",
    
    # Major Indian dynasties
    "mauryan", "ashoka", "bindusara", "chandragupta", "gupta", "samudragupta", "vikramaditya",
    "kushan", "kanishka", "satavahana", "pallava", "chola", "rashtrakuta", "chalukya",
    "maratha", "shivaji", "sambhaji", "shahu", "peshwa", "bajirao", "balaji vishwanath", "balaji baji rao",
    "nana saheb", "raghunath rao", "madhav rao", "holkar", "scindia", "bhonsle", "gaikwad",

    # Mughal emperors
    "mughal", "babur", "humayun", "akbar", "jahangir", "shah jahan", "aurangzeb", "bahadur shah", "farrukhsiyar", 
    "muhammad shah", "ahmad shah", "shah alam", "akbar ii", "bahadur shah ii",

    # Rajput rulers
    "rajput", "prithviraj chauhan", "rana pratap", "rana sanga", "man singh", "amar singh", "sawai jai singh", "udai singh", 
    "bundi", "jodhpur", "bikaner", "jaipur", "chittorgarh", "udaipur", "meewar", "marwar", "hadi rani",
    
    # Sikh empire
    "sikh", "guru gobind singh", "guru nanak", "ranjit singh", "hari singh nalwa", "jassa singh ahluwalia",

    # Mysore and Deccan rulers
    "mysore", "tipu sultan", "hyder ali", "nizam", "qutb shahi", "adil shahi", "bidar sultanate",

    # Famous battles
    "battle of panipat", "battle of haldighati", "third battle of panipat", "battle of plassey", "battle of buxar",

    # Historical sites
    "hampi", "ellora", "ajanta", "konark", "sanchi", "qutub minar", "gateway of india", "taj mahal", "red fort",
    "charminar", "mysore palace", "amer fort", "gwalior fort", "chittorgarh", "halebidu", "khajuraho", "elephanta caves",
    "jantar mantar", "brihadeeswarar temple", "ramappa temple", "rani ki vav",

    # Artifacts & weapons
    "coin", "script", "manuscript", "weapon", "armory", "sword", "shield", "turban", "armor", "horse", "elephant",

    # Indus Valley Civilization
    "indus valley", "harappa", "mohenjo daro", "ashokan edicts", "sarasvati civilization"
}

# ✅ Step 1: Load PDFs and Extract Text
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

# ✅ Step 2: Process and Store Text in FAISS
def store_embeddings():
    pdf_texts = load_pdfs_from_folder()
    if not pdf_texts:
        print("⚠️ No PDFs loaded! Please check the folder path.")
        return

    print("🔠 Splitting Text into Chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(pdf_texts)

    if not chunks:
        print("⚠️ No valid text chunks found.")
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

    # Save text mapping to JSON
    text_map = {i: chunk.page_content for i, chunk in enumerate(chunks)}
    with open(TEXT_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(text_map, f, ensure_ascii=False, indent=4)

    print(f"✅ Text mapping saved to '{TEXT_MAP_PATH}'.")

# ✅ Step 3: Retrieve Relevant Text from FAISS
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

    # Load stored text mapping
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

    # Validate retrieved index
    if str(retrieved_idx) not in text_map:
        print(f"⚠️ Invalid index {retrieved_idx}. No corresponding text found.")
        return None

    print(f"✅ FAISS retrieved match at index {retrieved_idx} with distance {distances[0][0]}")
    return text_map[str(retrieved_idx)]

# ✅ Step 4: Check Keywords Before Answering
def check_keywords(question):
    return any(keyword.lower() in question.lower() for keyword in history_keywords)

# ✅ Step 5: Answer Question using FAISS, Keywords, or Fallback
def answer_question(question):
    # Step 1: **Try FAISS First**
    relevant_context = retrieve_relevant_text(question)
    if relevant_context:
        prompt = f"""You are a professional museum guide. Answer based on the following historical context:
        
        Context: {relevant_context}

        Question: {question}
        
        Provide a concise, accurate response.""" 
        
        llm = Ollama(model="mistral")
        return llm(prompt)

    # Step 2: **If FAISS Fails, Check Museum-Related Keywords**
    if check_keywords(question):
        prompt = f"""You are a professional museum guide. Answer this history-related question:

        Question: {question}
        
        Provide an informative answer.""" 
        
        llm = Ollama(model="mistral")
        return llm(prompt)

    # Step 3: **If Both Fail, Reject the Query**
    print("❌ This is not my expertise.")  # Log the rejection
    return "❌ This is not my expertise."

# ✅ Step 6: Main Execution Loop
def main():
    print("📖 Loading PDFs and Storing in FAISS...")

    if not os.path.exists(FAISS_INDEX_PATH):
        store_embeddings()

    print("\n💬 Ask me anything about Indian History (type 'exit' to stop):")

    while True:
        question = input("\n❓ Question: ")
        if question.lower() == "exit":
            break

        answer = answer_question(question)
        print("\n💡 Answer:", answer)

# Run the system
if __name__ == "__main__":
    main()
