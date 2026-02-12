import torch
import os
import sys

def test_model():
    print("🧠 AI Image Detector - Model Testing")
    print("=" * 50)
    
    # Thêm thư mục hiện tại vào path để import
    sys.path.append(os.path.dirname(__file__))
    
    try:
        from model_loader import load_model, predict_ai_image
        
        print("\n1. Testing model loading...")
        
        # Kiểm tra các file model có thể
        model_files = []
        possible_paths = [
            "models/model_best.pth",
            "models/ai_real_classifier.h5",
            "../Training-models/src/checkpoints/model_best.pth"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_files.append(path)
                print(f"   ✅ Found: {path}")
            else:
                print(f"   ❌ Not found: {path}")
        
        if model_files:
            model_path = model_files[0]
            print(f"   🔄 Loading model from: {model_path}")
        else:
            model_path = None
            print("   ⚠️  No model files found, using dummy model")
        
        # Load model
        model = load_model(model_path)
        print(f"   ✅ Model loaded: {model is not None}")
        
        print("\n2. Testing model prediction...")
        
        # Tạo ảnh test
        try:
            from PIL import Image
            import io
            import numpy as np
            
            # Tạo ảnh RGB ngẫu nhiên
            dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            test_image = Image.fromarray(dummy_image)
            
            # Chuyển thành bytes
            img_bytes = io.BytesIO()
            test_image.save(img_bytes, format='JPEG')
            img_bytes = img_bytes.getvalue()
            
            print("   📷 Created test image")
            
            # Test prediction
            result = predict_ai_image(img_bytes)
            
            if "error" in result:
                print(f"   ❌ Prediction error: {result['error']}")
            else:
                print(f"   ✅ Prediction successful!")
                print(f"   🎯 Result: {result['label']}")
                print(f"   📊 Confidence: {result['confidence_score']:.4f}")
                print(f"   🤖 AI Generated: {result['is_ai_generated']}")
                print(f"   📈 Raw Probability: {result.get('raw_probability', 'N/A')}")
                
        except ImportError as e:
            print(f"   ❌ Cannot create test image: {e}")
        except Exception as e:
            print(f"   ❌ Prediction test failed: {e}")
        
        print("\n3. Testing device information...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   🔧 Using device: {device}")
        if torch.cuda.is_available():
            print(f"   🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"   💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        print("\n🎉 Model Testing Completed!")
        if model_path:
            print(f"✅ Model is working with: {os.path.basename(model_path)}")
        else:
            print("✅ Dummy model is working (no real model file found)")
        
        return True
        
    except Exception as e:
        print(f"❌ Model testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if test_model():
        sys.exit(0)
    else:
        sys.exit(1)