from flask import Flask, request, jsonify
from langchain_community.llms import Ollama

app = Flask(__name__)  # Initialize Flask app
model = Ollama (model="mistral")  # Load AI model

# Supported Indian Languages
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

@app.route('/')  # Home route
def home():
    return "Welcome to the Indian Museum AI API!"

# ✅ **1. Ask Question API**
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_query = data.get('question', '')

    if not user_query:
        return jsonify({"error": "No question provided!"}), 400

    response = model(f"Answer this museum-related question: {user_query}")
    return jsonify({"answer": response})

# ✅ **2. Summarization API**
@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text_to_summarize = data.get('text', '')

    if not text_to_summarize:
        return jsonify({"error": "No text provided for summarization!"}), 400

    summary_prompt = f"Summarize the following text in simple words:\n\n{text_to_summarize}"
    summary = model(summary_prompt)

    return jsonify({"summary": summary})

# ✅ **3. Translation API**
@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    text_to_translate = data.get('text', '')
    target_language = data.get('language', '')

    if not text_to_translate or not target_language:
        return jsonify({"error": "Text and target language are required!"}), 400

    if target_language not in indian_languages:
        return jsonify({"error": f"Language '{target_language}' not supported!"}), 400

    translate_prompt = f"Translate the following text into {target_language}:\n\n{text_to_translate}"
    translated_text = model(translate_prompt)

    return jsonify({"translated_text": translated_text})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
