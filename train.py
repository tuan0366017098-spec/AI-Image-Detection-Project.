import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm
import random
import time
import warnings

warnings.filterwarnings('ignore')

# 🆕 CẤU HÌNH TỐI ƯU TỐC ĐỘ
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

# Set random seeds
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


class FastAIVsRealDataset(Dataset):
    def __init__(self, ai_dir, real_dir, transform=None):
        self.transform = transform
        self.data = []

        # 🆕 Sử dụng list comprehension nhanh hơn
        ai_images = [os.path.join(ai_dir, f) for f in os.listdir(ai_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # 🆕 Balance dataset nhanh hơn
        min_len = min(len(ai_images), len(real_images))
        self.data = ([(img, 1) for img in random.sample(ai_images, min_len)] +
                     [(img, 0) for img in random.sample(real_images, min_len)])
        random.shuffle(self.data)

        print(f"📊 Dataset: {len(ai_images)} AI, {len(real_images)} Real -> {min_len} each")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            # Fallback image
            image = Image.new('RGB', (224, 224), color='gray')

        if self.transform:
            image = self.transform(image)

        return image, label


def train_one_epoch_fast(model, dataloader, criterion, optimizer, device, scaler):
    """🆕 SỬA LỖI: Xử lý cả CPU và GPU"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    accumulation_steps = 2
    use_amp = scaler is not None  # Kiểm tra có dùng AMP không

    for i, (images, labels) in enumerate(tqdm(dataloader, desc="🚀 Training")):
        images, labels = images.to(device), labels.to(device)

        # 🆕 SỬA LỖI: Xử lý cả CPU và GPU
        if use_amp:
            # Mixed precision training (GPU)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels) / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            # Normal training (CPU)
            outputs = model(images)
            loss = criterion(outputs, labels) / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        _, predictions = torch.max(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def train_fast(path, num_epochs=25):
    # 🆕 Tối ưu hóa DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # 🆕 SỬA LỖI: Khởi tạo scaler đúng cách
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Transform
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Dataset và DataLoader
    ai_dir = os.path.join(path, "data/AiArtData")
    real_dir = os.path.join(path, "data/RealArt")

    dataset = FastAIVsRealDataset(ai_dir, real_dir, transform=train_transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # DataLoader tối ưu
    num_workers = min(4, os.cpu_count())
    pin_memory = device.type == 'cuda'

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,  # 🆕 Giảm batch size để ổn định hơn
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"🚀 Training với: {len(train_dataset)} ảnh, Batch size: 32, Workers: {num_workers}")

    # Model
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, 2)
    )
    model = model.to(device)

    # Optimizer và scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # OneCycleLR
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )

    # Training loop
    best_accuracy = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        print(f"\n🎯 Epoch {epoch + 1}/{num_epochs}")

        # Training
        train_loss, train_acc = train_one_epoch_fast(model, train_loader, criterion, optimizer, device, scaler)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predictions = torch.max(outputs, dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        scheduler.step()

        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time

        print(f"✅ Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"✅ Val Acc: {val_acc:.2f}%")
        print(f"⏱️  Epoch time: {epoch_time:.1f}s, Total: {total_time / 60:.1f}m")

        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), 'models/fast_model_v2.pth')
            print(f"💾 Best model saved! Accuracy: {val_acc:.2f}%")

    print(f"\n🎉 Training completed! Best accuracy: {best_accuracy:.2f}%")
    return model


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("🚀 BẮT ĐẦU TRAINING TỐI ƯU TỐC ĐỘ...")
    start_time = time.time()

    # 🆕 Tạo thư mục models nếu chưa có
    os.makedirs('models', exist_ok=True)

    model = train_fast(project_root, num_epochs=25)

    total_time = time.time() - start_time
    print(f"⏱️  Total training time: {total_time / 60:.1f} minutes")