import joblib
import pandas as pd
from src.config import settings
from typing import Optional, Any

class PredictionService:
    """
    Encapsulates the ML model and all pre/post-processing logic.
    
    This class loads artifacts on initialization and provides
    a clean method for making predictions.
    """
    
    # Expected feature order for the model
    MODEL_EXPECTED_COLS = ['package_weight_gr', 'package_size']

    def __init__(self, model_path: str, size_encoder_path: str, type_encoder_path: str):
        """
        Initializes the service by loading all required artifacts.
        
        Args:
            model_path (str): Path to the model.pkl file.
            size_encoder_path (str): Path to the package_size_encoder.pkl file.
            type_encoder_path (str): Path to the product_type_encoder.pkl file.
            
        Raises:
            RuntimeError: If any artifact fails to load.
        """
        print("Initializing PredictionService...")
        self.model: Optional[Any] = None
        self.size_encoder: Optional[Any] = None
        self.type_encoder: Optional[Any] = None
        
        try:
            print(f"Loading model from: {model_path}")
            self.model = joblib.load(model_path)
            
            print(f"Loading size encoder from: {size_encoder_path}")
            self.size_encoder = joblib.load(size_encoder_path)

            print(f"Loading target encoder from: {type_encoder_path}")
            self.type_encoder = joblib.load(type_encoder_path)
            
        except FileNotFoundError as e:
            print(f"[SERVICE_ERROR] Critical artifact not found: {e}")
            raise RuntimeError(f"Failed to load artifact. Service cannot start. {e}")
        except Exception as e:
            print(f"[SERVICE_ERROR] An unexpected error occurred during initialization: {e}")
            raise RuntimeError(f"Failed to initialize service. {e}")
            
        print("PredictionService initialized successfully.")

    def predict(self, package_weight: float, package_size: str) -> str:
        """
        Performs pre-processing, prediction, and post-processing.
        
        Args:
            package_weight (float): The package weight in grams.
            package_size (str): The package size (e.g., "Small Package").
            
        Returns:
            str: The predicted product label (e.g., "Smartphone").
            
        Raises:
            ValueError: If the input 'package_size' is unknown.
            ValueError: If the model returns an unexpected class.
        """
        
        if not self.model or not self.size_encoder or not self.type_encoder:
            raise RuntimeError("PredictionService is not fully initialized.")

        try:
            size_encoded = self.size_encoder.transform([package_size])[0]
        except ValueError as e:
            print(f"[SERVICE_WARNING] Unknown 'package_size' value: {package_size}")
            raise ValueError(f"Invalid or unknown 'package_size' value: '{package_size}'")

        # Create a DataFrame in the exact order the model expects
        input_data = pd.DataFrame(
            [[package_weight, size_encoded]], 
            columns=self.MODEL_EXPECTED_COLS
        )

        # Prediction
        prediction_encoded = self.model.predict(input_data)[0]

        # Decoding
        try:
            prediction_label = self.type_encoder.inverse_transform([prediction_encoded])[0]
        except ValueError as e:
            print(f"[SERVICE_ERROR] Model returned a class index '{prediction_encoded}' that the target encoder does not know.")
            raise ValueError("Model prediction is incompatible with the target encoder.")

        return prediction_label

# --- Singleton Instance ---
# Create a single instance of the service when the module is imported.
# This ensures artifacts are loaded only ONCE at application startup.
try:
    prediction_service = PredictionService(
        model_path = settings.MODEL_PATH,
        size_encoder_path = settings.SIZE_ENCODER_PATH,
        type_encoder_path = settings.TYPE_ENCODER_PATH
    )
except RuntimeError as e:
    print(f"[FATAL] Could not initialize PredictionService: {e}")
    prediction_service = None