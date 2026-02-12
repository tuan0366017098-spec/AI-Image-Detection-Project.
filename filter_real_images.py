import os
import shutil
from model_loader import load_model, predict_ai_image

def filter_ai_images(source_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    count_ai = 0
    count_all = 0

    for filename in os.listdir(source_folder):
        if not filename.lower().endswith(valid_ext):
            continue

        filepath = os.path.join(source_folder, filename)
        with open(filepath, "rb") as f:
            image_bytes = f.read()

        result = predict_ai_image(image_bytes)
        label = result.get("label", "")

        count_all += 1
        if label == "AI Generated":
            count_ai += 1
            dest_path = os.path.join(output_folder, filename)
            shutil.copy2(filepath, dest_path)
            print(f"[AI] {filename} => Đã copy")
        else:
            print(f"[REAL] {filename} => Bỏ qua")

    print("\n===== KẾT QUẢ =====")
    print(f"Tổng ảnh kiểm tra: {count_all}")
    print(f"Số ảnh được lọc là AI Generated: {count_ai}")



if __name__ == "__main__":
    model_path = "models/fast_model.pth"

    print("🔄 Đang load model...")
    load_model(model_path)

    test_folder = "Test/FAKE"  # folder chứa ảnh gốc (hỗn hợp AI & Real)
    output_folder = "FAKE_filtered"  # folder lưu ảnh AI đã lọc

    filter_ai_images(test_folder, output_folder)
