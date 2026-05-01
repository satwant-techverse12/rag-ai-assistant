from dotenv import load_dotenv
load_dotenv()

import os
from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException
from routes.api import api


def create_app():
    app = Flask(__name__, template_folder="templates")

    app.config.update(
        MAX_CONTENT_LENGTH=10 * 1024 * 1024,
        JSON_SORT_KEYS=False
    )

    # ✅ NO prefix here (Option B)
    app.register_blueprint(api)

    @app.route("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.errorhandler(HTTPException)
    def handle_http_error(e):
        return jsonify({
            "error": e.name,
            "message": e.description
        }), e.code

    @app.errorhandler(Exception)
    def handle_exception(e):
        print("ERROR:", str(e))
        return jsonify({"error": "Internal server error"}), 500

    return app


app = create_app()

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )