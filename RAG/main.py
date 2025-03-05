import streamlit as st
from langchain_community.llms import Ollama
from retrieve import load_and_index_pdfs, retrieve_relevant_text

# 🔥 Load Model
def load_model():
    return Ollama(model="mistral")

# ✅ Expanded Historical Keywords List
museum_keywords = [
    "museum", "monument", "artifact", "historical", "history", "ancient", "heritage", "exhibit",
    "dynasty", "king", "queen", "ruler", "warrior", "battle", "fort", "palace", "kingdom", "inscription", "scripture",
    "maharaja", "maharani", "chatrapati", "maharana", "pratap", "shivaji", "ashoka", "gupta", "kushan", "maratha", 
    "rajput", "mughal", "sultanate", "delhi sultanate", "chola", "pallava", "rashtrakuta", "satavahana", "vikramaditya",
    "tipu sultan", "ranjit singh", "aurangzeb", "bajirao", "peshwa", "hyder ali", "bahadur shah", "akbar", "samrat", 
    "hampi", "ellora", "ajanta", "konark", "sanchi", "qutub minar", "gateway of india", "taj mahal", "red fort",
    "charminar", "mysore palace", "amer fort", "gwalior fort", "chittorgarh", "halebidu", "khajuraho", "elephanta caves",
    "sundarban", "bhimbetka", "jantar mantar", "brihadeeswarar temple", "ramappa temple", "rani ki vav",
    "coin", "script", "manuscript", "weapon", "armory", "sword", "shield", "turban", "armor", "horse", "elephant",
    "battle of panipat", "battle of haldighati", "third battle of panipat", "plassey", "buxar", "angkor wat",
    "indus valley", "harappa", "mohenjo daro", "ashokan edicts", "sarasvati civilization", 
    "buddha", "mahabharata", "ramayana", "mauryan empire", "chandragupta maurya", "bindusara",
    "gupta empire", "samudragupta", "harshavardhana", "nalanda university", "vishnu", "shiva",
    "parvati", "krishna", "radha", "rama", "sita", "lakshmana", "hanuman", "ravana", "vedic period",
    "vedic civilization", "aryan migration", "ancient temples", "hoysala dynasty", "kakatiya dynasty",
    "sikh empire", "guru nanak", "guru gobind singh", "ranjit singh", "anglo-maratha wars",
    "anglo-mysore wars", "first war of independence", "1857 revolt", "jhansi ki rani",
    "bhagat singh", "subhash chandra bose", "chandrasekhar azad", "ashfaqulla khan",
    "netaji", "quit india movement", "dandi march", "indigo rebellion", "salt satyagraha",
    "battle of plassey", "battle of buxar", "third anglo-maratha war", "fort william",
    "pattadakal", "ellora caves", "badami caves", "elephanta caves", "delhi durbar",
    "sultan ghiyasuddin tughlaq", "allauddin khilji", "tughlaq dynasty", "lodhi dynasty"
]

# 🚀 Generate Response using Ollama
def generate_response(model, user_query):
    if not any(keyword in user_query.lower() for keyword in museum_keywords):
        return "❌ Sorry, museums and Indian history are my expertise. I can't answer that."

    retrieved_context = retrieve_relevant_text(user_query)

    prompt = f"""You are a professional museum guide specializing in Indian history. 
    Answer only museum-related and historical questions based on the given context. 

    Context: {retrieved_context}

    Question: {user_query}
    Provide an accurate, detailed response.
    """

    return model(prompt)

# 🏛️ Streamlit Web UI
def main():
    st.title("🏛️ Indian Museum AI Guide")
    st.subheader("🧐 Get expert information about Indian historical monuments, artifacts, and rulers!")

    # 🔄 Load FAISS index & documents
    load_and_index_pdfs()

    model = load_model()
    user_query = st.text_area("Ask me anything about Indian history and museums:", height=100)

    if st.button("Get Information"):
        if user_query.strip():
            response = generate_response(model, user_query)
            st.subheader("📜 Museum Knowledge:")
            st.write(response)
        else:
            st.warning("⚠️ Please enter a museum-related question.")

if __name__ == "__main__":
    main()
