import os
import torch


def check_model_files():
    print("🔍 Checking for model files...")

    possible_locations = [
        "models/model_best.pth",
        "models/ai_real_classifier.h5",
        "../Training-models/src/checkpoints/model_best.pth",
        "checkpoints/model_best.pth"
    ]

    for location in possible_locations:
        if os.path.exists(location):
            print(f"✅ Found: {location}")
            try:
                # Thử load model để kiểm tra
                model = torch.load(location, map_location='cpu')
                print(f"   Model type: {type(model)}")
                if isinstance(model, dict):
                    print(f"   Checkpoint keys: {list(model.keys())}")
                return True
            except Exception as e:
                print(f"   ❌ Error loading: {e}")
        else:
            print(f"   ❌ Not found: {location}")

    print("⚠️  No valid model files found. System will use dummy model.")
    return False


if __name__ == "__main__":
    check_model_files()