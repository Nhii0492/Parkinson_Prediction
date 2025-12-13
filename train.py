import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, \
    precision_score

# ==========================================
# 1. CẤU HÌNH (SETTINGS)
# ==========================================
TRAIN_PATH = 'dataset/train_set.npz'
TEST_PATH = 'dataset/test_set.npz'

BATCH_SIZE = 64
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100
MULTIPLIER = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ==========================================
# 2. CBAM MODULES (UPDATE 7x7)
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
    def __init__(self, kernel_size=7):  # <--- MẶC ĐỊNH LÀ 7
        super(SpatialAttention, self).__init__()

        # Logic padding tự động: Nếu 7x7 thì padding=3 để giữ nguyên kích thước ảnh
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
    def __init__(self, in_planes, ratio=16, kernel_size=7):  # <--- UPDATE kernel_size=7
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# ==========================================
# 3. CUSTOM RESNET18 + CBAM (7x7)
# ==========================================
class ResNet18_CBAM(nn.Module):
    def __init__(self, num_classes=2, dropout_prob=0.5):
        super(ResNet18_CBAM, self).__init__()

        try:
            from torchvision.models import ResNet18_Weights
            base_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except:
            base_model = models.resnet18(pretrained=True)

        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        # --- UPDATE: GỌI CBAM VỚI KERNEL 7x7 ---
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
# 4. DATA PROCESSING (Giữ nguyên)
# ==========================================
augment_transform = transforms.Compose([
    transforms.RandomRotation(degrees=360),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

to_tensor_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class GeneratedDataset(Dataset):
    def __init__(self, x_data, y_data, class_map, multiplier=1):
        self.samples = []
        print(f"--> Đang khởi tạo dữ liệu (Multiplier={multiplier})...")

        for idx in range(len(x_data)):
            img_arr = x_data[idx]
            if isinstance(img_arr, np.ndarray):
                img_arr = img_arr.astype(np.uint8)
                original_pil = Image.fromarray(img_arr).convert('RGB')

            label_raw = y_data[idx]
            if isinstance(label_raw, (str, np.str_)):
                label_idx = class_map[label_raw]
            elif isinstance(label_raw, np.ndarray):
                item = label_raw.item()
                label_idx = class_map[item] if isinstance(item, (str, np.str_)) else item
            else:
                label_idx = label_raw

            self.samples.append((to_tensor_transform(original_pil), label_idx))

            for _ in range(multiplier - 1):
                aug_img = augment_transform(original_pil)
                self.samples.append((to_tensor_transform(aug_img), label_idx))

        print(f"    Hoàn tất! Tổng số ảnh: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        return img, torch.tensor(label, dtype=torch.long)


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (CBAM 7x7)')
    plt.savefig('confusion_matrix_cbam7x7.png')
    plt.show()


# ==========================================
# 5. MAIN
# ==========================================
def main():
    try:
        data_train = np.load(TRAIN_PATH, allow_pickle=True)
        x_train, y_train = data_train['arr_0'], data_train['arr_1']
        data_test = np.load(TEST_PATH, allow_pickle=True)
        x_test, y_test = data_test['arr_0'], data_test['arr_1']

        unique_labels = np.unique(y_train)
        class_map = {label: idx for idx, label in enumerate(unique_labels)}
        class_names = [str(label) for label in unique_labels]
        print(f"Class Map: {class_map}")
    except Exception as e:
        print(f"Lỗi load data: {e}")
        return

    train_dataset = GeneratedDataset(x_train, y_train, class_map, multiplier=MULTIPLIER)
    test_dataset = GeneratedDataset(x_test, y_test, class_map, multiplier=1)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("\nKhởi tạo ResNet18 + CBAM (Kernel 7x7)...")
    model = ResNet18_CBAM(num_classes=len(unique_labels), dropout_prob=0.5)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nBẮT ĐẦU HUẤN LUYỆN...")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_correct = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_correct += (outputs.argmax(1) == labels).sum().item()

        epoch_acc = train_correct / len(train_dataset)

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Acc: {epoch_acc:.4f} | Test Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"--> Kỷ lục mới: {best_acc:.4f}")

    print("\n" + "=" * 40)
    print(f"KẾT QUẢ CHI TIẾT (CBAM 7x7)")
    print("=" * 40)

    model.load_state_dict(best_model_wts)
    model.eval()

    y_true_final = []
    y_pred_final = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true_final.extend(labels.cpu().numpy())
            y_pred_final.extend(preds.cpu().numpy())

    final_acc = accuracy_score(y_true_final, y_pred_final)
    final_prec = precision_score(y_true_final, y_pred_final, average='macro')
    final_rec = recall_score(y_true_final, y_pred_final, average='macro')
    final_f1 = f1_score(y_true_final, y_pred_final, average='macro')

    print(f"Overall Accuracy  : {final_acc:.15f}")
    print(f"Average Precision : {final_prec:.15f}")
    print(f"Average Recall    : {final_rec:.15f}")
    print(f"Average F1 Score  : {final_f1:.15f}")

    save_path = 'parkinson_cbam7x7_best.pth'
    torch.save(model.state_dict(), save_path)
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)

    print("-" * 40)
    print(f"Đã lưu model tại: {save_path}")
    print(f"Model Size      : {file_size_mb:.6f} MB")
    print("-" * 40)

    plot_confusion_matrix(y_true_final, y_pred_final, class_names)


if __name__ == '__main__':
    main()