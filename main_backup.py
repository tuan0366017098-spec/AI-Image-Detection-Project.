from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import torch
from PIL import Image
import io
import os
import sys

# Thêm thư mục hiện tại vào path để import
sys.path.append(os.path.dirname(__file__))

app = FastAPI(title="AI vs Real Image Detector API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
model_loaded = False
model_type = "none"


def load_real_model():
    """Load model thật từ file"""
    global model, model_loaded, model_type

    try:
        print("🔍 Đang tìm kiếm model thật...")

        # Các vị trí có thể có model
        possible_paths = [
            "models/model_best.pth",
            "models/best_model.pth",
            "models/checkpoint.pth",
            "models/ai_real_classifier.h5",
            "../Training-models/src/checkpoints/model_best.pth",
            "../Training-models/models/model_best.pth"
        ]

        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                print(f"✅ Tìm thấy model: {path}")
                break

        if not model_path:
            print("❌ Không tìm thấy file model nào!")
            return False

        # Import model_loader
        try:
            from model_loader import load_model, predict_ai_image
            print("✅ Import model_loader thành công")
        except ImportError as e:
            print(f"❌ Lỗi import model_loader: {e}")
            return False

        # Load model
        print(f"🔄 Đang load model từ: {model_path}")
        model = load_model(model_path)

        # Kiểm tra xem model có phải dummy không
        if hasattr(model, '__class__') and 'DummyModel' in str(model.__class__):
            print("❌ Model được load là DUMMY MODEL")
            return False
        else:
            print("✅ Load MODEL THẬT thành công!")
            model_loaded = True
            model_type = "real"
            return True

    except Exception as e:
        print(f"❌ Lỗi khi load model: {e}")
        import traceback
        traceback.print_exc()
        return False


def dummy_predict(image_bytes, filename):
    """Dummy model dự phòng"""
    return {
        "is_ai_generated": len(filename) % 2 == 0,
        "confidence_score": 0.75,
        "label": "AI Generated" if len(filename) % 2 == 0 else "Real Image"
    }


@app.on_event("startup")
async def startup_event():
    print("🚀 Khởi động AI Image Detector...")

    # Thử load model thật
    if load_real_model():
        print("🎯 HỆ THỐNG ĐANG SỬ DỤNG MODEL THẬT")
    else:
        print("⚠️  SỬ DỤNG DUMMY MODEL (không tìm thấy model thật)")
        model_type = "dummy"


@app.get("/")
def read_root():
    model_status = "REAL MODEL" if model_loaded else "DUMMY MODEL"
    return {
        "message": f"AI Image Detector API - Đang sử dụng: {model_status}",
        "model_loaded": model_loaded,
        "model_type": model_type
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "model_type": model_type,
        "message": "Đang sử dụng model thật" if model_loaded else "Đang sử dụng dummy model"
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()

        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large")

        # Sử dụng model thật nếu có
        if model_loaded:
            try:
                from model_loader import predict_ai_image
                result = predict_ai_image(contents)
                model_used = "REAL"
            except Exception as e:
                print(f"❌ Lỗi khi dự đoán với model thật: {e}")
                result = dummy_predict(contents, file.filename)
                model_used = "DUMMY (fallback)"
        else:
            result = dummy_predict(contents, file.filename)
            model_used = "DUMMY"

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        # Chuyển đổi kết quả
        is_ai = result["is_ai_generated"]
        confidence = result["confidence_score"]
        class_name = "ẢNH DO AI TẠO" if is_ai else "ẢNH THẬT"

        print(f"📊 Dự đoán ({model_used}): {class_name} - Độ tin cậy: {confidence:.2%}")

        return JSONResponse({
            "prediction": 1 if is_ai else 0,
            "class_name": class_name,
            "confidence": confidence,
            "filename": file.filename,
            "model_used": model_used,
            "message": f"Kết quả từ {model_used}: {class_name} ({(confidence * 100):.1f}%)"
        })

    except Exception as e:
        print(f"❌ Lỗi dự đoán: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý ảnh: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)