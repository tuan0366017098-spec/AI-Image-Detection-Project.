from pydantic import BaseModel, Field
from typing import Dict, Optional, List

class HealthResponse(BaseModel):
    """Health check response schema"""
    status: str = Field(..., example="healthy")
    message: str = Field(..., example="API is running")
    version: str = Field(..., example="1.0.0")
    model_loaded: bool = Field(..., example=True)

class PredictionResponse(BaseModel):
    """Prediction response schema"""
    label: str = Field(..., example="AI", description="Predicted label")
    confidence: float = Field(..., example=0.95, description="Confidence score")
    is_ai: bool = Field(..., example=True, description="True if AI-generated")
    processing_time: float = Field(..., example=0.125, description="Processing time in seconds")
    probabilities: Dict[str, float] = Field(
        ...,
        example={"ai": 0.95, "real": 0.05},
        description="Probability distribution"
    )
    message: str = Field(..., example="High confidence prediction")

class ErrorResponse(BaseModel):
    """Error response schema"""
    error: bool = Field(..., example=True)
    message: str = Field(..., example="Error message")
    status_code: int = Field(..., example=400)

class BatchPredictionResult(BaseModel):
    """Single result in batch prediction"""
    filename: str
    label: Optional[str] = None
    confidence: Optional[float] = None
    is_ai: Optional[bool] = None
    processing_time: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    message: Optional[str] = None
    error: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    """Batch prediction response schema"""
    results: List[BatchPredictionResult]
    total_files: int
    total_processing_time: float

class ModelInfoResponse(BaseModel):
    """Model information response schema"""
    model_name: str
    model_type: str
    input_size: List[int]
    classes: Dict[int, str]
    device: str
    loaded_at: Optional[str] = None