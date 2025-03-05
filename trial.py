import streamlit as st
from langchain_community.llms import Ollama

def load_model():
    return Ollama(model="mistral")

# Expanded keywords with all major Indian rulers, dynasties, and historical topics
museum_keywords = [
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
]

# Supported Indian Languages for Translation
indian_languages = {
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

# Function to generate response based on user query
def generate_response(model, user_query):
    if not any(keyword in user_query.lower() for keyword in museum_keywords):
        return "❌ Sorry, I specialize in museums and Indian history. I can't answer that."

    prompt = f"""You are a professional museum guide. Answer only museum-related questions with expert knowledge.
    
    Question: {user_query}

    Provide a specific response based on historical accuracy, cultural heritage, and famous exhibits in museums.
    """
    
    response = model(prompt)
    return response

# Function to summarize generated response
def summarize_response(model, response_text):
    prompt = f"""Summarize the following museum-related text in simple terms:
    
    Text: {response_text}
    
    Provide a concise summary."""
    
    return model(prompt)

# Function to translate response to selected Indian language
def translate_response(model, response_text, selected_language):
    prompt = f"""Translate the following museum-related text into {selected_language}:
    
    Text: {response_text}
    
    Provide an accurate translation."""
    
    return model(prompt)

def main():
    st.title("🏛️ Indian Museum AI Guide")
    st.subheader("🧐 Get expert information about Indian historical monuments, artifacts, and rulers!")

    model = load_model()
    
    user_query = st.text_area("Enter your museum-related query:", height=100)

    if st.button("Get Information"):
        if user_query.strip():
            response = generate_response(model, user_query)
            st.session_state["latest_response"] = response  # Store latest response
            st.subheader("📜 Museum Knowledge:")
            st.write(response)
        else:
            st.warning("⚠️ Please enter a museum-related question.")

    # Show buttons for Summarization and Translation after response is generated
    if "latest_response" in st.session_state:
        response_text = st.session_state["latest_response"]

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Summarize Response"):
                summarized_text = summarize_response(model, response_text)
                st.subheader("📝 Summarized Text:")
                st.write(summarized_text)
                st.session_state["latest_response"] = summarized_text  # Update latest response with summary

        with col2:
            selected_language = st.selectbox("Select Language for Translation:", list(indian_languages.keys()), key="lang_select")
            if st.button("Translate Response"):
                translated_text = translate_response(model, response_text, selected_language)
                st.subheader(f"🌍 Translated Text ({selected_language}):")
                st.write(translated_text)

if __name__ == "__main__":
    main()
