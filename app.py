import os
from flask import Flask, request, render_template, session
from flask_session import Session
from openai import OpenAI
from dotenv import load_dotenv
from rag_retriever import retrieve_codex_context, check_and_update_vectorstore

# Load environment variables
load_dotenv()

# Load OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file. Make sure it's set correctly.")

print(f"✅ OpenAI key loaded successfully (length={len(api_key)} chars)")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default_secret")
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Knowledge base folder location
KB_FOLDER = "knowledge_base"

# Perform vectorstore check at startup
print("🔎 Checking vectorstore at app startup...")
check_and_update_vectorstore(KB_FOLDER)

@app.route("/", methods=["GET", "POST"])
def index():
    if "conversation" not in session:
        session["conversation"] = []

    venue = ""
    user_prompt = ""
    email = ""

    if request.method == "POST":
        venue = request.form.get("venue_concept", "")
        user_prompt = request.form.get("user_prompt", "")
        email = request.form.get("email", "")

        # Re-check vectorstore at each user interaction
        check_and_update_vectorstore(KB_FOLDER)

        # Retrieve RAG context
        rag_context = retrieve_codex_context(user_prompt)

        # Build conversation
        conversation = session["conversation"]
        conversation.append({"role": "user", "content": f"Venue concept: {venue}\n{user_prompt}"})
        conversation.insert(1, {"role": "system", "content": f"Helpful reference:\n{rag_context}"})

        # Load system prompt
        with open("system_prompt.txt", "r") as file:
            system_prompt = file.read()

        conversation.insert(0, {"role": "system", "content": system_prompt})

        # Call OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=conversation,
            temperature=0.7
        )

        ai_response = response.choices[0].message.content
        conversation.append({"role": "assistant", "content": ai_response})
        session["conversation"] = conversation

    else:
        session["conversation"] = []

    return render_template("index.html", conversation=session["conversation"])

if __name__ == "__main__":
    app.run(debug=True)