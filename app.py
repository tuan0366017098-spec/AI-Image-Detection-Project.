# import os
# import logging
# from typing import Dict, Any, List
# from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# import uvicorn
# from PIL import Image
# import io
# import time
# import asyncio

# # Import local modules
# from model_loader import ModelLoader
# from schemas import PredictionResponse, HealthResponse, ErrorResponse
# from config import settings
# from middleware import log_requests_middleware
# from utils import save_uploaded_image, cleanup_old_files, validate_image

# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('backend.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # Initialize FastAPI app
# app = FastAPI(
#     title="AI vs Real Image Detector API",
#     description="Deep Learning API for detecting AI-generated images",
#     version="1.0.0",
#     docs_url="/docs",
#     redoc_url="/redoc"
# )

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=settings.ALLOWED_ORIGINS,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Add custom middleware
# app.middleware("http")(log_requests_middleware)

# # Global model instance
# model_loader = None


# @app.on_event("startup")
# async def startup_event():
#     """Initialize model on startup"""
#     global model_loader
#     try:
#         logger.info("🚀 Starting up AI Image Detector API...")
#         model_loader = ModelLoader()
#         await model_loader.load_model()
#         logger.info("✅ Model loaded successfully")
#     except Exception as e:
#         logger.error(f"❌ Failed to load model: {str(e)}")
#         raise


# @app.on_event("shutdown")
# async def shutdown_event():
#     """Cleanup on shutdown"""
#     logger.info("🛑 Shutting down AI Image Detector API...")
#     if model_loader:
#         model_loader.cleanup()


# @app.get("/", response_model=HealthResponse)
# async def root():
#     """Root endpoint - API information"""
#     return HealthResponse(
#         status="healthy",
#         message="AI vs Real Image Detector API is running",
#         version="1.0.0",
#         model_loaded=model_loader is not None
#     )


# @app.get("/health", response_model=HealthResponse)
# async def health_check():
#     """Health check endpoint"""
#     model_status = "loaded" if model_loader and model_loader.model is not None else "not loaded"
#     return HealthResponse(
#         status="healthy",
#         message=f"API is operational. Model: {model_status}",
#         version="1.0.0",
#         model_loaded=model_loader is not None
#     )


# @app.post("/predict", response_model=PredictionResponse)
# async def predict_image(
#         background_tasks: BackgroundTasks,
#         file: UploadFile = File(..., description="Image file to analyze (JPEG, PNG, WebP)")
# ):
#     """
#     Predict if an image is AI-generated or Real

#     - **file**: Image file (JPEG, PNG, WebP supported, max 10MB)
#     """
#     start_time = time.time()

#     try:
#         # Validate file type
#         if not file.content_type.startswith('image/'):
#             raise HTTPException(
#                 status_code=400,
#                 detail="File must be an image (JPEG, PNG, WebP)"
#             )

#         # Read image
#         contents = await file.read()

#         # Check file size
#         if len(contents) > settings.MAX_FILE_SIZE:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"File too large. Maximum size is {settings.MAX_FILE_SIZE // 1024 // 1024}MB"
#             )

#         # Open and validate image
#         try:
#             image = Image.open(io.BytesIO(contents))
#         except Exception:
#             raise HTTPException(status_code=400, detail="Invalid image file")

#         # Validate image
#         validation_result = validate_image(image)
#         if not validation_result["valid"]:
#             raise HTTPException(status_code=400, detail=validation_result["message"])

#         # Save uploaded image for debugging (optional)
#         temp_path = None
#         if settings.SAVE_UPLOADED_IMAGES:
#             temp_path = save_uploaded_image(contents, file.filename)
#             background_tasks.add_task(cleanup_old_files)

#         # Make prediction
#         prediction = await model_loader.predict(image)

#         processing_time = time.time() - start_time

#         logger.info(f"📊 Prediction: {prediction['label']} "
#                     f"(Confidence: {prediction['confidence']:.4f}) "
#                     f"Time: {processing_time:.3f}s")

#         return PredictionResponse(
#             label=prediction["label"],
#             confidence=prediction["confidence"],
#             is_ai=prediction["is_ai"],
#             processing_time=processing_time,
#             probabilities={
#                 "ai": prediction["probabilities"]["ai"],
#                 "real": prediction["probabilities"]["real"]
#             },
#             message=prediction["message"]
#         )

#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"❌ Prediction error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# @app.post("/batch-predict")
# async def batch_predict(
#         background_tasks: BackgroundTasks,
#         files: List[UploadFile] = File(..., description="Multiple image files")
# ):
#     """
#     Batch prediction for multiple images (max 10 files)
#     """
#     if len(files) > settings.MAX_BATCH_SIZE:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Maximum {settings.MAX_BATCH_SIZE} files allowed for batch processing"
#         )

#     results = []
#     total_start_time = time.time()

#     for file in files:
#         file_start_time = time.time()
#         try:
#             # Validate file type
#             if not file.content_type.startswith('image/'):
#                 results.append({
#                     "filename": file.filename,
#                     "error": "File must be an image"
#                 })
#                 continue

#             # Read image
#             contents = await file.read()

#             # Check file size
#             if len(contents) > settings.MAX_FILE_SIZE:
#                 results.append({
#                     "filename": file.filename,
#                     "error": f"File too large (max {settings.MAX_FILE_SIZE // 1024 // 1024}MB)"
#                 })
#                 continue

#             # Open image
#             try:
#                 image = Image.open(io.BytesIO(contents))
#             except Exception:
#                 results.append({
#                     "filename": file.filename,
#                     "error": "Invalid image file"
#                 })
#                 continue

#             # Validate image
#             validation_result = validate_image(image)
#             if not validation_result["valid"]:
#                 results.append({
#                     "filename": file.filename,
#                     "error": validation_result["message"]
#                 })
#                 continue

#             # Make prediction
#             prediction = await model_loader.predict(image)
#             processing_time = time.time() - file_start_time

#             results.append({
#                 "filename": file.filename,
#                 **prediction,
#                 "processing_time": processing_time
#             })

#         except Exception as e:
#             results.append({
#                 "filename": file.filename,
#                 "error": f"Processing failed: {str(e)}"
#             })

#     total_time = time.time() - total_start_time
#     logger.info(f"📦 Batch prediction completed: {len(results)} files, Time: {total_time:.3f}s")

#     return {
#         "results": results,
#         "total_files": len(results),
#         "total_processing_time": total_time
#     }


# @app.get("/model-info")
# async def model_info():
#     """Get information about the loaded model"""
#     if not model_loader or not model_loader.model:
#         raise HTTPException(status_code=503, detail="Model not loaded")

#     return {
#         "model_name": model_loader.model_name,
#         "model_type": model_loader.model_type,
#         "input_size": model_loader.input_size,
#         "classes": model_loader.classes,
#         "device": str(model_loader.device),
#         "loaded_at": model_loader.loaded_at.isoformat() if model_loader.loaded_at else None
#     }


# # Error handlers
# @app.exception_handler(HTTPException)
# async def http_exception_handler(request, exc):
#     return JSONResponse(
#         status_code=exc.status_code,
#         content=ErrorResponse(
#             error=True,
#             message=exc.detail,
#             status_code=exc.status_code
#         ).dict()
#     )


# @app.exception_handler(Exception)
# async def general_exception_handler(request, exc):
#     logger.error(f"Unexpected error: {str(exc)}")
#     return JSONResponse(
#         status_code=500,
#         content=ErrorResponse(
#             error=True,
#             message="Internal server error",
#             status_code=500
#         ).dict()
#     )


# if __name__ == "__main__":
#     uvicorn.run(
#         "app:app",
#         host=settings.HOST,
#         port=settings.PORT,
#         reload=settings.DEBUG,
#         log_level="info"
#     )