from flask import Flask, render_template, request, jsonify
from rag import rag_pipeline

app = Flask(__name__)

# Home page
@app.route("/", methods=["GET", "POST"])
def home():
    answer = ""
    if request.method == "POST":
        query = request.form["query"]
        if query:
            answer = rag_pipeline(query)
    return render_template("index.html", answer=answer)

# API chat endpoint
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"answer": "❌ No question provided."})

    answer = rag_pipeline(query)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)