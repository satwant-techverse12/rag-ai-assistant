from flask import Flask, request, jsonify, render_template
from rag_pipeline import ask_question
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
print("App starting...")
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()

        if not data or "question" not in data:
            return jsonify({"error": "Question is required"}), 400

        question = data["question"]

        if not question.strip():
            return jsonify({"error": "Empty question"}), 400

        answer = ask_question(question)

        return jsonify({"answer": answer})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "Something went wrong"}), 500


UPLOAD_FOLDER = "documents"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"message": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"message": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    file.save(filepath)

    # 🔥 process immediately
    from ingest import process_pdf
    process_pdf(filepath)

    return jsonify({"message": "✅ PDF uploaded & indexed!"})




if __name__ == "__main__":
    app.run(debug=True)