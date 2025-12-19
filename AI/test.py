import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score

# ==========================================
# 1. CẤU HÌNH (SETTINGS)
# ==========================================
MODEL_PATH = 'model/parkinson_cbam7x7_best.pth'
TEST_SET_PATH = 'dataset/test_set.npz'

# Ảnh cần test riêng lẻ
IMAGE_1_PATH = 'Parkinson_Prediction/dataset/test_image_healthy.png'
IMAGE_2_PATH = 'dataset/test_image_parkinson.png'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 2. ĐỊNH NGHĨA LẠI MODEL (PHẢI GIỐNG FILE TRAIN)
# ==========================================

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv1(x_cat)
        x_out = self.bn(x_out)
        return self.sigmoid(x_out)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class ResNet18_CBAM(nn.Module):
    def __init__(self, num_classes=2, dropout_prob=0.5):
        super(ResNet18_CBAM, self).__init__()

        base_model = models.resnet18(pretrained=False)

        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.cbam1 = CBAM(64, kernel_size=7)
        self.cbam2 = CBAM(128, kernel_size=7)
        self.cbam3 = CBAM(256, kernel_size=7)
        self.cbam4 = CBAM(512, kernel_size=7)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob / 2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.cbam1(x)

        x = self.layer2(x)
        x = self.cbam2(x)

        x = self.layer3(x)
        x = self.cbam3(x)

        x = self.layer4(x)
        x = self.cbam4(x)

        x = self.avgpool(x)
        x = self.classifier(x)
        return x


# ==========================================
# 3. CHUẨN BỊ DỮ LIỆU
# ==========================================
# Transform cho inference (chỉ resize và normalize, không augment)
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class TestDataset(Dataset):
    def __init__(self, x_data, y_data, class_map):
        self.samples = []
        for idx in range(len(x_data)):
            img_arr = x_data[idx]
            if isinstance(img_arr, np.ndarray):
                img_arr = img_arr.astype(np.uint8)
                original_pil = Image.fromarray(img_arr).convert('RGB')

            label_raw = y_data[idx]
            # Xử lý label
            if isinstance(label_raw, (str, np.str_)):
                label_idx = class_map.get(label_raw, 0)
            elif isinstance(label_raw, np.ndarray):
                item = label_raw.item()
                label_idx = class_map.get(item, 0) if isinstance(item, (str, np.str_)) else item
            else:
                label_idx = label_raw

            self.samples.append((inference_transform(original_pil), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        return img, torch.tensor(label, dtype=torch.long)


# ==========================================
# 4. HÀM DỰ ĐOÁN
# ==========================================

def predict_single_image(model, image_path, class_names, device):
    """
    Dự đoán một ảnh lẻ từ đường dẫn
    """
    if not os.path.exists(image_path):
        print(f"[CẢNH BÁO] Không tìm thấy ảnh: {image_path}")
        return

    # Mở ảnh
    image = Image.open(image_path).convert('RGB')

    # Tiền xử lý (Transform + thêm Batch dimension)
    input_tensor = inference_transform(image).unsqueeze(0).to(device)

    # Dự đoán
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class_idx = torch.max(probabilities, 1)

    predicted_label = class_names[predicted_class_idx.item()]
    conf_score = confidence.item() * 100

    print("-" * 40)
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Predict: {predicted_label}")
    print(f"Confidence: {conf_score:.2f}%")

    # In ra xác suất của từng lớp để tham khảo
    for idx, name in enumerate(class_names):
        print(f"   - {name}: {probabilities[0][idx].item() * 100:.2f}%")
    print("-" * 40)


def evaluate_entire_test_set(model, test_path, device):
    """
    Đánh giá lại toàn bộ tập test
    """
    if not os.path.exists(test_path):
        print("Not dataset.")
        return

    print(f"\nImage from dataset{test_path}...")
    data_test = np.load(test_path, allow_pickle=True)
    x_test, y_test = data_test['arr_0'], data_test['arr_1']

    # Tạo class map giả định (hoặc load từ file cấu hình nếu có)
    unique_labels = np.unique(y_test)
    class_map = {label: idx for idx, label in enumerate(unique_labels)}
    class_names = [str(label) for label in unique_labels]

    test_dataset = TestDataset(x_test, y_test, class_map)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Số lượng ảnh test: {len(test_dataset)}")

    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Tính toán metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print("\n=== RESULT ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    return class_names  # Trả về tên lớp để dùng cho dự đoán ảnh lẻ


# ==========================================
# 5. MAIN
# ==========================================
def main():
    print(f"Sử dụng thiết bị: {DEVICE}")

    # 1. Khởi tạo model
    # Lưu ý: num_classes=2. Nếu tập dữ liệu của bạn khác 2 lớp, hãy sửa số này.
    model = ResNet18_CBAM(num_classes=2).to(DEVICE)

    # 2. Load Weights
    if os.path.exists(MODEL_PATH):
        print(f"Đang load model từ: {MODEL_PATH}")
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print("Load model thành công!")
        except Exception as e:
            print(f"Lỗi khi load model: {e}")
            return
    else:
        print(f"Lỗi: Không tìm thấy file model {MODEL_PATH}")
        return

    # 3. Đánh giá tập Test Set (và lấy danh sách tên lớp)
    # Nếu không có file dataset test, ta mặc định tên lớp là ['Healthy', 'Parkinson']
    # Cần đảm bảo thứ tự này khớp với lúc train (thường là a-z: Healthy trước Parkinson)
    class_names = evaluate_entire_test_set(model, TEST_SET_PATH, DEVICE)

    if class_names is None:
        # Fallback nếu không load được tập test
        class_names = ['Healthy', 'Parkinson']
        print(f"Sử dụng class names mặc định: {class_names}")

    print("\n=== Test===")
    # 4. Dự đoán ảnh Healthy
    predict_single_image(model, IMAGE_1_PATH, class_names, DEVICE)

    # 5. Dự đoán ảnh Parkinson
    predict_single_image(model, IMAGE_2_PATH, class_names, DEVICE)


if __name__ == "__main__":
    main()