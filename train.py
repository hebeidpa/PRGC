
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from transformers import AutoConfig, AutoModel
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np
import pandas as pd
from tqdm import tqdm

# ============ 可配置输出路径 ============
output_dir = "/mnt/home/clam/CLAM-master/output"
os.makedirs(output_dir, exist_ok=True)

# ============ 参数配置 ============
num_classes = 8
batch_size = 32
num_epochs = 50
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ 数据预处理 ============
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============ 数据加载 ============
full_dataset = datasets.ImageFolder("/mnt/home/clam/CLAM-master/all_image", transform=transform)
indices = list(range(len(full_dataset)))
labels = [full_dataset[i][1] for i in indices]
train_idx, val_idx = train_test_split(indices, test_size=0.3, stratify=labels, random_state=42)
val_dataset = Subset(full_dataset, val_idx)
val_loader_final = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
train_labels = [full_dataset[i][1] for i in train_idx]

# ============ 模型配置 ============
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

# ============ 十折交叉验证训练 ============
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
results = []
best_acc = 0.0
best_model_path = os.path.join(output_dir, "unipath_model_best.pth")

for fold, (train_idx_fold, val_idx_fold) in enumerate(skf.split(train_idx, train_labels)):
    print(f"Fold {fold+1}")
    train_subset = Subset(full_dataset, [train_idx[i] for i in train_idx_fold])
    val_subset = Subset(full_dataset, [train_idx[i] for i in val_idx_fold])
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = UniClassifier(base_model, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total, correct, total_loss = 0, 0, 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{num_epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
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

        print(f"[Fold {fold+1} Epoch {epoch+1}] Loss: {total_loss:.4f}, Accuracy: {train_acc:.4f}, AUC: {auc:.4f}, Sens: {sensitivity:.4f}, Spec: {specificity:.4f}")

        results.append({
            "Fold": fold+1,
            "Epoch": epoch+1,
            "Loss": total_loss,
            "Accuracy": train_acc,
            "AUC": auc,
            "Sensitivity": sensitivity,
            "Specificity": specificity
        })

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model updated at Fold {fold+1} Epoch {epoch+1}, accuracy: {best_acc:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_dir, "unipath_training_results_cv.csv"), index=False)
print("Training complete. Results saved.")

# ============ 最终验证集评估 ============
print("\nEvaluating best model on 30% hold-out validation set...")
best_model = UniClassifier(base_model, num_classes).to(device)
best_model.load_state_dict(torch.load(best_model_path))
best_model.eval()

val_preds, val_labels, val_probs = [], [], []
with torch.no_grad():
    for imgs, labels in val_loader_final:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = best_model(imgs)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(probs, 1)
        val_preds.extend(preds.cpu().numpy())
        val_labels.extend(labels.cpu().numpy())
        val_probs.extend(probs.cpu().numpy())

val_preds = np.array(val_preds)
val_labels = np.array(val_labels)
val_probs = np.array(val_probs)

try:
    y_true_bin = np.eye(num_classes)[val_labels]
    val_auc = roc_auc_score(y_true_bin, val_probs, average='macro', multi_class='ovr')
except:
    val_auc = np.nan

cm = confusion_matrix(val_labels, val_preds)
TN = np.sum(cm) - np.sum(np.diag(cm))
FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
val_sens = np.mean(TP / (TP + FN + 1e-6))
val_spec = np.mean(TN / (TN + FP + 1e-6))
val_acc = np.mean(val_preds == val_labels)

print(f"\n=== Final Validation Performance (30% hold-out set) ===")
print(f"ACC: {val_acc:.4f}")
print(f"AUC: {val_auc:.4f}")
print(f"Sensitivity: {val_sens:.4f}")
print(f"Specificity: {val_spec:.4f}")
