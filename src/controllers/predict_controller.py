from flask import request, jsonify
from pydantic import ValidationError
from typing import Optional
from src.services.prediction_service import PredictionService
from src.models.schemas import PredictionRequest

class PredictionController:
    """
    Handles HTTP requests for the /predict endpoint.
    It validates input using Pydantic schemas and uses the
    PredictionService to get a result.
    """
    def __init__(self, service: Optional[PredictionService]):
        """
        Initializes the controller with an injected prediction service.
        """
        if service is None:
            print("[CONTROLLER_FATAL] PredictionService is None. The controller cannot operate.")
        self.service = service

    def predict(self):
        """
        Handles the POST /predict request.
        Validates JSON, calls the service, and formats the response.
        """
        if not self.service:
            return jsonify({"error": "Service is not available. Check server logs."}), 503

        try:
            raw_data = request.get_json()
            if not raw_data:
                return jsonify({"error": "Invalid request. No JSON data received."}), 400
            
            input_data = PredictionRequest(**raw_data)
        
        except ValidationError as e:
            return jsonify({"error": "Invalid input.", "details": e.json()}), 422

        # Prediction
        try:
            prediction_label = self.service.predict(
                package_weight=float(input_data.package_weight_gr), 
                package_size=input_data.package_size
            )
            
            response_data = {
                "input_received": input_data.model_dump(),
                "predicted_product_type": prediction_label
            }
            return jsonify(response_data), 200

        except ValueError as e:
            return jsonify({"error": f"Bad Request: {e}"}), 400
        except Exception as e:
            print(f"[CONTROLLER_ERROR] Unexpected error: {e}")
            return jsonify({"error": "An internal server error occurred."}), 500

# --- Singleton ---
from src.services.prediction_service import prediction_service
prediction_controller = PredictionController(prediction_service)