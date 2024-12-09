from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from transformers import pipeline  # Import pipeline for LLM usage

app = Flask(__name__)
CORS(app)

# Load the JSON file containing questions and answers
try:
    with open(r"questions_and_answers.json", "r") as file:
        data = json.load(file)
except FileNotFoundError:
    print("Error: 'questions_and_answers.json' file not found.")
    data = []

# Convert JSON data to DataFrame for easier processing
df = pd.DataFrame(data)

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Check if DataFrame is empty
if not df.empty:
    # Fit and transform the questions to create TF-IDF vectors
    question_vectors = vectorizer.fit_transform(df["prompt"])
else:
    print("Error: No data found in JSON file.")

# Function to find the closest matching question and return its answer
def get_answer_from_files(user_question, df, question_vectors):
    user_question_vector = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_question_vector, question_vectors)
    closest_idx = similarities.argmax()
    
    # Check if the match is strong enough (e.g., similarity > 0.7)
    if similarities[0][closest_idx] > 0.7:
        return df.iloc[closest_idx]["response"]
    return None

# Initialize LLM (GPT-Neo or similar)
llm = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')  # You can change this to any model you prefer

# Fallback to LLM if no match in files
def get_answer_from_llm(user_question):
    response = llm(user_question, max_length=100, num_return_sequences=1)
    return response[0]['generated_text']

# Root route to confirm API is running
@app.route('/')
def home():
    return "Chatbot API is running!"

# Define a route for chatbot responses
@app.route('/get_response', methods=['POST'])
def chatbot_response():
    user_question = request.json.get("question")
    if user_question:
        # First, try to find an answer in your files
        answer = get_answer_from_files(user_question, df, question_vectors)
        
        # If no answer found, fall back to LLM
        if not answer:
            answer = get_answer_from_llm(user_question)
        
        return jsonify({"response": answer})
    else:
        return jsonify({"response": "Please provide a question."}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))  # Get the PORT from environment or use 5002 as default
    app.run(
        host='0.0.0.0',
        port=port
    )  # Bind to all addresses and specified port
