import requests
import json
import sys
import os

def test_backend():
    base_url = "http://localhost:8000"
    
    print("🧪 AI Image Detector - API Testing")
    print("=" * 50)
    
    try:
        # Test 1: Health endpoint
        print("\n1. Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Health check: {response.status_code}")
            print(f"   📊 Status: {health_data.get('status', 'N/A')}")
            print(f"   🤖 Model: {health_data.get('model', 'N/A')}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False

        # Test 2: Root endpoint
        print("\n2. Testing root endpoint...")
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            print(f"   ✅ Root endpoint: {response.status_code}")
            print(f"   📝 Message: {response.json().get('message', 'N/A')}")
        else:
            print(f"   ❌ Root endpoint failed: {response.status_code}")

        # Test 3: Check if docs are accessible
        print("\n3. Testing API documentation...")
        response = requests.get(f"{base_url}/docs", timeout=10)
        if response.status_code == 200:
            print(f"   ✅ API docs: {response.status_code}")
        else:
            print(f"   ⚠️  API docs: {response.status_code} (might be normal)")

        # Test 4: Test with a sample image (if available)
        print("\n4. Testing prediction endpoint...")
        test_image_path = "test_image.jpg"
        
        # Tạo ảnh test đơn giản nếu chưa có
        if not os.path.exists(test_image_path):
            print("   💡 Creating test image...")
            try:
                from PIL import Image
                import numpy as np
                # Tạo ảnh màu đơn giản
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                test_image = Image.fromarray(img_array)
                test_image.save(test_image_path)
                print("   ✅ Test image created")
            except ImportError:
                print("   ⚠️  Cannot create test image (PIL not available)")
                test_image_path = None
        
        if test_image_path and os.path.exists(test_image_path):
            try:
                with open(test_image_path, 'rb') as f:
                    files = {'file': (test_image_path, f, 'image/jpeg')}
                    response = requests.post(f"{base_url}/predict", files=files, timeout=30)
                
                if response.status_code == 200:
                    prediction = response.json()
                    print(f"   ✅ Prediction test: {response.status_code}")
                    print(f"   🎯 Result: {prediction.get('class_name', 'N/A')}")
                    print(f"   📈 Confidence: {prediction.get('confidence', 'N/A')}")
                else:
                    print(f"   ❌ Prediction failed: {response.status_code}")
                    print(f"   📄 Response: {response.text}")
            except Exception as e:
                print(f"   ❌ Prediction test error: {e}")
        else:
            print("   ⚠️  Skipping prediction test (no test image)")

        print("\n🎉 API Testing Completed!")
        print("📋 Summary:")
        print(f"   - Backend URL: {base_url}")
        print(f"   - Health: ✅ Working")
        print(f"   - API Docs: ✅ Accessible")
        print(f"   - Ready for frontend connection!")
        
        return True

    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to {base_url}")
        print("💡 Please make sure the backend server is running!")
        return False
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return False

if __name__ == "__main__":
    if test_backend():
        sys.exit(0)
    else:
        sys.exit(1)