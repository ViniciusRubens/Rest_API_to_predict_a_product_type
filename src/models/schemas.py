from pydantic import BaseModel, Field, condecimal
from decimal import Decimal
from typing import Dict, Any

class PredictionRequest(BaseModel):
    """
    Schema for the input data for prediction.
    Validates the incoming JSON payload.
    """
    
    # Use condecimal (constrained decimal) to ensure weight is positive
    package_weight_gr: condecimal(gt=Decimal(0.0)) = Field(
        ...,
        description="The weight of the package in grams.",
        examples=[150.5, 300.0]
    )
    
    package_size: str = Field(
        ...,
        description="The size category of the package.",
        examples=["Small Package", "Large Package"]
    )
    
    class Config:
        """
        Pydantic config.
        'forbid' prevents users from sending extra fields.
        """
        extra = "forbid"

class PredictionResponse(BaseModel):
    """
    Schema for a successful prediction response.
    """
    input_received: PredictionRequest
    predicted_product_type: str

class ErrorResponse(BaseModel):
    """
    Schema for a generic error response.
    """
    error: str

class ValidationErrorResponse(BaseModel):
    """
    Schema for a Pydantic validation error response.
    """
    error: str
    details: Dict[str, Any]