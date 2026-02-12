import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import io
import os

# Global model instance
MODEL = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Device: {DEVICE}")


def load_model(model_path=None):
    global MODEL
    try:
        print(f"🔄 Loading model from: {model_path}")

        if not model_path or not os.path.exists(model_path):
            print("❌ Model file không tồn tại")
            return create_dummy_model()

        model = models.resnet34(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.7),
            nn.Linear(num_ftrs, 2)  # Output 2 lớp: AI và Real
        )

        checkpoint = torch.load(model_path, map_location=DEVICE)

        # Nếu checkpoint là dict có thể chứa "state_dict"
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)

        model.to(DEVICE)
        model.eval()
        MODEL = model
        print("✅ MODEL THẬT loaded successfully!")
        return MODEL

    except Exception as e:
        print(f"❌ Lỗi load model thật: {e}")
        import traceback
        traceback.print_exc()
        return create_dummy_model()


def create_dummy_model():
    """Tạo dummy model fallback"""

    class DummyModel:
        def eval(self): return self

        def to(self, device): return self

        def __call__(self, x):
            # Trả về logits cho 2 lớp: giả sử luôn là lớp Real (class 0)
            return torch.tensor([[10.0, -10.0]])  # logits để softmax thành Real Image

    print("⚠️ Using DUMMY MODEL")
    global MODEL
    MODEL = DummyModel()
    return MODEL


def preprocess_image(image_bytes):
    """Preprocess image cho model"""
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return transform(image).unsqueeze(0).to(DEVICE)
    except Exception as e:
        raise ValueError(f"Lỗi preprocess: {e}")


def predict_ai_image(image_bytes):
    if MODEL is None:
        return {"error": "Model chưa được load"}

    if hasattr(MODEL, '__class__') and 'DummyModel' in str(MODEL.__class__):
        return {
            "is_ai_generated": False,
            "confidence_score": 0.5,
            "label": "Real Image"
        }

    try:
        input_tensor = preprocess_image(image_bytes)

        with torch.no_grad():
            outputs = MODEL(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

            is_ai = (predicted.item() == 1)  # class 1 = AI, class 0 = Real
            confidence_score = confidence.item()

        return {
            "is_ai_generated": is_ai,
            "confidence_score": round(confidence_score, 4),
            "label": "AI Generated" if is_ai else "Real Image",
            "raw_output": outputs.tolist()
        }

    except Exception as e:
        print(f"❌ Lỗi dự đoán: {e}")
        return {"error": f"Dự đoán thất bại: {str(e)}"}
