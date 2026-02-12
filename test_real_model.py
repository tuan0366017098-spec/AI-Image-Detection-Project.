import requests
import os


def test_real_model():
    print("🧪 TEST MODEL THẬT")

    # Test health endpoint
    response = requests.get("http://localhost:8000/health")
    data = response.json()

    print(f"Health: {data}")

    if data["model_loaded"]:
        print("✅ MODEL THẬT ĐANG CHẠY!")

        # Test với ảnh sample
        test_image = "test_image.jpg"
        if os.path.exists(test_image):
            with open(test_image, "rb") as f:
                files = {"file": f}
                response = requests.post("http://localhost:8000/predict", files=files)
                result = response.json()
                print(f"📊 Kết quả thật: {result}")
        else:
            print("💡 Tạo ảnh test...")
            from PIL import Image
            import numpy as np
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            Image.fromarray(img_array).save(test_image)
            print("✅ Đã tạo ảnh test")
    else:
        print("❌ MODEL THẬT CHƯA LOAD ĐƯỢC")


if __name__ == "__main__":
    test_real_model()