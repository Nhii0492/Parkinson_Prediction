import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import random  

# Import GradCAM++
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ==========================================
# 1. Config
# ==========================================
MODEL_PATH = 'parkinson_cbam7x7_best.pth'
TEST_DATA_PATH = 'dataset/test_set.npz'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['Healthy', 'Parkinson']


# ==========================================
# 2. MODEL (ResNet18 + CBAM 7x7)
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
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3
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
# 3. Supporting
# ==========================================
def load_trained_model():
    print(f"Loading custom model from {MODEL_PATH}...")
    try:
        model = ResNet18_CBAM(num_classes=2)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        print(f"Lỗi load model: {e}")
        return None


def preprocess_image(img_arr):
    img_arr = img_arr.astype(np.uint8)
    img_pil = Image.fromarray(img_arr).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(img_pil).unsqueeze(0), img_pil


def visualize_and_save(model, img_raw, true_label_str, pred_str, conf, save_name):
    input_tensor, img_pil = preprocess_image(img_raw)
    input_tensor = input_tensor.to(DEVICE)
    pred_idx = CLASS_NAMES.index(pred_str)

    # Grad-CAM++
    target_layers = [model.layer4[-1]]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    # Visualize
    img_resized = img_pil.resize((224, 224))
    rgb_img = np.float32(img_resized) / 255.0
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img_resized)
    plt.title(f"Original\nTrue: {true_label_str}")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(grayscale_cam, cmap='jet')
    plt.title("Grad-CAM++ Heatmap\n(ResNet18 + CBAM)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(visualization)
    plt.title(f"Pred: {pred_str}\nConf: {conf * 100:.2f}%")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_name)
    print(f"Đã lưu ảnh giải thích tại: {save_name}")
    plt.close()


# ==========================================
# 4. Main GradCam++
# ==========================================
def main():
    # 1. Load Model & Data
    model = load_trained_model()
    if model is None: return

    data = np.load(TEST_DATA_PATH, allow_pickle=True)
    x_test, y_test = data['arr_0'], data['arr_1']


    correct_healthy_indices = []
    correct_parkinson_indices = []

    print("\n Processing...")

    for i in range(len(x_test)):
        img_raw = x_test[i]
        label_raw = y_test[i]
        true_label_str = str(label_raw)  # 'healthy' hoặc 'parkinson'

        input_tensor, _ = preprocess_image(img_raw)
        input_tensor = input_tensor.to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            pred_idx = torch.argmax(probs).item()
            pred_str = CLASS_NAMES[pred_idx].lower()  

        # If Ok
        if pred_str == true_label_str:
            if true_label_str == 'healthy':
                correct_healthy_indices.append((i, probs[pred_idx].item()))
            else:
                correct_parkinson_indices.append((i, probs[pred_idx].item()))

    print(f"Tìm thấy {len(correct_healthy_indices)} ảnh Healthy dự đoán đúng.")
    print(f"Tìm thấy {len(correct_parkinson_indices)} ảnh Parkinson dự đoán đúng.")

    # ==========================================
    # RANDOM CHỌN VÀ VẼ ẢNH HEALTHY
    # ==========================================
    if len(correct_healthy_indices) > 0:
        # Chọn ngẫu nhiên 1 ảnh
        idx, conf = random.choice(correct_healthy_indices)
        print(f"\n--> Đã random chọn ảnh Healthy index {idx} (Conf: {conf * 100:.2f}%)")

        img = x_test[idx]
        visualize_and_save(model, img, 'healthy', 'Healthy', conf, 'xai_cbam_random_healthy.png')
    else:
        print("Không tìm thấy ảnh Healthy nào dự đoán đúng!")

    # ==========================================
    # RANDOM CHỌN VÀ VẼ ẢNH PARKINSON
    # ==========================================
    if len(correct_parkinson_indices) > 0:
        # Chọn ngẫu nhiên 1 ảnh
        idx, conf = random.choice(correct_parkinson_indices)
        print(f"\n--> Đã random chọn ảnh Parkinson index {idx} (Conf: {conf * 100:.2f}%)")

        img = x_test[idx]
        visualize_and_save(model, img, 'parkinson', 'Parkinson', conf, 'xai_cbam_random_parkinson.png')
    else:
        print("Không tìm thấy ảnh Parkinson nào dự đoán đúng!")

    print("\nHOÀN TẤT!")


if __name__ == '__main__':
    main()
