from flask import Blueprint, request, jsonify, render_template
from services.rag_service import ask_question
from utils.ingest import process_pdf
from werkzeug.utils import secure_filename
import os

api = Blueprint("api", __name__)

UPLOAD_FOLDER = "documents"
ALLOWED_EXTENSIONS = {"pdf"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# -----------------------------
# Home (UI)
# -----------------------------
@api.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# -----------------------------
# File validation
# -----------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# -----------------------------
# Ask Question
# -----------------------------
@api.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify({"success": False, "error": "Invalid JSON"}), 400

        question = data.get("question", "").strip()

        if not question:
            return jsonify({"success": False, "error": "Question is required"}), 400

        answer = ask_question(question)

        return jsonify({
            "success": True,
            "question": question,
            "answer": answer
        })

    except Exception as e:
        print("ASK ERROR:", str(e))
        return jsonify({"success": False, "error": "Failed to process question"}), 500


# -----------------------------
# Upload PDF
# -----------------------------
@api.route("/upload", methods=["POST"])
def upload_pdf():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"success": False, "error": "No selected file"}), 400

        if not allowed_file(file.filename):
            return jsonify({"success": False, "error": "Only PDF files are allowed"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        file.save(filepath)

        # Process PDF → build vector DB
        process_pdf(filepath)

        return jsonify({
            "success": True,
            "message": "PDF uploaded and indexed successfully"
        })

    except Exception as e:
        print("UPLOAD ERROR:", str(e))
        return jsonify({"success": False, "error": "Upload failed"}), 500