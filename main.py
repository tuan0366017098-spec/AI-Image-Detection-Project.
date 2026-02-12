import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import logging
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Image Detector - REAL MODEL")

# CORS middleware (thêm origin của frontend bạn nếu có)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
model_loaded = False

class RealAIModel:
    def __init__(self):
        self.model = None
        self.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

        self.load_model()

    def load_model(self):
        global model_loaded
        model_path = "models/fast_model.pth"
        if not os.path.exists(model_path):
            logger.error(f"❌ Model file không tồn tại: {model_path}")
            model_loaded = False
            return False

        logger.info(f"🔄 Đang load model từ: {model_path}")

        # Khởi tạo model ResNet34 đúng cấu trúc
        self.model = models.resnet34(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)  # 2 class: Real và AI

        checkpoint = torch.load(model_path, map_location='cpu')

        # Kiểm tra checkpoint dạng dict hoặc không
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Load state dict với strict=False để tránh lỗi mismatch key
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        if missing:
            logger.warning(f"⚠ Missing keys khi load model: {missing}")
        if unexpected:
            logger.warning(f"⚠ Unexpected keys khi load model: {unexpected}")

        self.model.eval()
        model_loaded = True
        logger.info("✅ Model đã load thành công")
        return True

    def predict(self, image_bytes):
        if not model_loaded or self.model is None:
            return {"error": "Model chưa được load"}

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)

                confidence, predicted = torch.max(probabilities, 1)

                is_ai = (predicted.item() == 1)
                confidence_score = confidence.item()

                # Lấy xác suất riêng cho AI và Real
                ai_confidence = probabilities[0][1].item()
                real_confidence = probabilities[0][0].item()

            return {
                "is_ai_generated": is_ai,
                "confidence_score": round(confidence_score, 4),
                "label": "AI Generated" if is_ai else "Real Image",
                "class_index": predicted.item(),
                "ai_confidence": round(ai_confidence, 4),
                "real_confidence": round(real_confidence, 4),
            }
        except Exception as e:
            logger.error(f"❌ Lỗi dự đoán: {e}")
            return {"error": f"Lỗi xử lý ảnh: {str(e)}"}


ai_model = RealAIModel()


@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Khởi động backend với MODEL THẬT...")
    if model_loaded:
        logger.info("🎯 SẴN SÀNG NHẬN ẢNH VÀ DỰ ĐOÁN")
    else:
        logger.error("❌ KHÔNG THỂ LOAD MODEL THẬT!")


@app.get("/")
def read_root():
    return {
        "message": "AI Image Detector - REAL MODEL",
        "model_loaded": model_loaded,
        "status": "ready" if model_loaded else "error"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model_loaded else "error",
        "model_loaded": model_loaded
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Model chưa được load")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File phải là ảnh")

    contents = await file.read()
    result = ai_model.predict(contents)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    logger.info(f"📊 Dự đoán: {result['label']} ({result['confidence_score']:.2%})")

    return {
        "is_ai_generated": result["is_ai_generated"],
        "confidence_score": result["confidence_score"],
        "label": result["label"],
        "ai_confidence": result["ai_confidence"],
        "real_confidence": result["real_confidence"],
        "filename": file.filename,
        "model_type": "REAL_MODEL"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
