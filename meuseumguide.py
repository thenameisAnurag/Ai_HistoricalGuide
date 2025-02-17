import streamlit as st
from langchain_community.llms import Ollama

def load_model():
    return Ollama(model="mistral")

# Expanded keyword list with Indian history focus
def generate_response(model, user_query):
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
        "indus valley", "harappa", "mohenjo daro", "ashokan edicts", "sarasvati civilization"
    ]

    if not any(keyword in user_query.lower() for keyword in museum_keywords):
        return "❌ Sorry, museums and Indian history are my expertise. I can't answer that."

    prompt = f"""You are a professional museum guide specializing in Indian history. Answer only museum-related and Indian history questions with expert knowledge.
    
    Question: {user_query}

    Provide a specific response based on historical accuracy, cultural heritage, and famous exhibits in Indian museums.
    """

    print("Prompt Sent to Model:", prompt)  # Debugging Log
    return model(prompt)

def main():
    st.title("🏛️ Indian Museum AI Guide")
    st.subheader("🧐 Get expert information about Indian historical monuments, artifacts, and rulers!")

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
