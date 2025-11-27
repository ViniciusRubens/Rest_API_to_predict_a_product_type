from flask import Flask, jsonify
from src.routes.predict_routes import predict_bp

def create_app():
    """
    Application Factory Pattern.
    Creates and configures the Flask application.
    """
    app = Flask(__name__)
    app.register_blueprint(predict_bp)

    @app.route('/health')
    def health():
        """
        A simple health check endpoint.
        """
        return jsonify({"status": "up", "service": "ML Prediction API"})

    return app