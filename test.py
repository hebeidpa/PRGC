
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from transformers import AutoConfig, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from tqdm import tqdm

# ============ 配置 ============
output_dir = "/mnt/home/clam/CLAM-master/output"
model_path = os.path.join(output_dir, "unipath_model_best.pth")
data_path = "/mnt/home/clam/CLAM-master/all_image"
num_classes = 8
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ 数据预处理 ============
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============ 数据加载 ============
full_dataset = datasets.ImageFolder(data_path, transform=transform)
indices = list(range(len(full_dataset)))
labels = [full_dataset[i][1] for i in indices]
_, test_idx = train_test_split(indices, test_size=0.3, stratify=labels, random_state=42)
test_dataset = Subset(full_dataset, test_idx)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# ============ 模型 ============
config = AutoConfig.from_pretrained("/mnt/home/clam/CLAM-master/checkpoints/UNI/config.json")
base_model = AutoModel.from_pretrained("/mnt/home/clam/CLAM-master/checkpoints/UNI/pytorch_model.bin", config=config)

class UniClassifier(nn.Module):
    def __init__(self, base_model, num_classes):
        super(UniClassifier, self).__init__()
        self.backbone = base_model
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            outputs = base_model(pixel_values=dummy_input, return_dict=True)
            hidden_size = outputs.last_hidden_state.shape[-1]
        self.classifier = nn.Linear(hidden_size, num_classes)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        torch.nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        outputs = self.backbone(pixel_values=x, return_dict=True)
        feat = outputs.last_hidden_state[:, 0]
        return self.classifier(feat)

# ============ 测试评估 ============
print("=== Evaluating on Test Set ===")
model = UniClassifier(base_model, num_classes).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Testing"):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(probs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

try:
    y_true_bin = np.eye(num_classes)[all_labels]
    auc = roc_auc_score(y_true_bin, all_probs, average='macro', multi_class='ovr')
except:
    auc = np.nan

cm = confusion_matrix(all_labels, all_preds)
TN = np.sum(cm) - np.sum(np.diag(cm))
FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
sensitivity = np.mean(TP / (TP + FN + 1e-6))
specificity = np.mean(TN / (TN + FP + 1e-6))
accuracy = np.mean(all_preds == all_labels)

print(f"[Test Set] Accuracy: {accuracy:.4f}")
print(f"[Test Set] ROC AUC: {auc:.4f}")
print(f"[Test Set] Sensitivity: {sensitivity:.4f}")
print(f"[Test Set] Specificity: {specificity:.4f}")

# 保存测试结果
pd.DataFrame({
    "Accuracy": [accuracy],
    "AUC": [auc],
    "Sensitivity": [sensitivity],
    "Specificity": [specificity]
}).to_csv(os.path.join(output_dir, "unipath_test_results.csv"), index=False)
