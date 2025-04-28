#!/usr/bin/env python
# coding: utf-8

import openslide
import numpy as np
import torch
import cv2
from torchvision import models, transforms
from scipy.ndimage import gaussian_filter
import pandas as pd
import os
import glob
import torch.nn as nn

# 配置参数
WSI_FOLDER = r"D:\BaiduNetdiskDownload\CLAM-master1\heatmaps\demo\slides"
weights_path = r"D:\陈圣的res\best_model.pth"
PATCH_SIZE = 224
LEVEL = 0
STRIDE = 224
MODEL_TYPE = "resnet18"
feature_num = 'all'

# 判断组织区域
def is_tissue(patch, threshold=0.5):
    gray_patch = np.array(patch.convert('L'))
    tissue_ratio = np.count_nonzero(gray_patch < 200) / gray_patch.size
    return tissue_ratio > threshold

# 模型定义
class FeatureExtractor(nn.Module):
    def __init__(self, num_classes=8, num_features=2048):
        super(FeatureExtractor, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.resnet(x)
        logits = self.classifier(features)
        return features, logits

# 初始化模型
def _init_feature_extractor(model_type):
    if model_type == "resnet18":
        feature_extractor = FeatureExtractor(num_classes=8, num_features=2048)
        feature_extractor.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        feature_extractor.eval()
        resnet18 = feature_extractor.resnet
        resnet18.fc = nn.Identity()
        model = torch.nn.Sequential(*list(resnet18.children())[:-1])
        model.eval()
        return model

# 特征提取
def _extract_feature(patch, model, device):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(patch).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model(input_tensor).cpu().numpy().flatten()
    return feature

# 初始化模型和设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = _init_feature_extractor(MODEL_TYPE).to(device)

# 遍历所有WSI
svs_files = glob.glob(os.path.join(WSI_FOLDER, '*.svs'))
summary_records = []

for svs_path in svs_files:
    slide_name = os.path.splitext(os.path.basename(svs_path))[0]
    slide = openslide.OpenSlide(svs_path)
    wsi_dims = slide.level_dimensions[LEVEL]
    print(f"\n处理: {slide_name} 大小: {wsi_dims}")

    heatmap_matrix = np.zeros((wsi_dims[1] // STRIDE + 1, wsi_dims[0] // STRIDE + 1))
    patch_features = []

    for y in range(0, wsi_dims[1], STRIDE):
        for x in range(0, wsi_dims[0], STRIDE):
            patch = slide.read_region((x, y), LEVEL, (PATCH_SIZE, PATCH_SIZE))
            patch_rgb = np.array(patch.convert("RGB"))

            if not is_tissue(patch):
                heatmap_matrix[y // STRIDE, x // STRIDE] = 0
                continue

            feature = _extract_feature(patch_rgb, model, device)
            if feature_num == 'all':
                feature_norm = np.linalg.norm(feature)
            else:
                feature_norm = np.linalg.norm(feature[feature_num])

            heatmap_matrix[y // STRIDE, x // STRIDE] = feature_norm
            patch_features.append(feature)

    # 保存 patch 特征
    patch_features = np.array(patch_features)
    feature_csv_path = os.path.join(WSI_FOLDER, f"{slide_name}_features.csv")
    #pd.DataFrame(patch_features).to_csv(feature_csv_path, index=False)

    # 生成热力图
    thumbnail_size = slide.level_dimensions[-1]
    heatmap = gaussian_filter(heatmap_matrix, sigma=0)
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    heatmap_resized = cv2.resize(heatmap, (thumbnail_size[0], thumbnail_size[1]), interpolation=cv2.INTER_LINEAR)
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_colored[heatmap_resized == 0] = [255, 255, 255]
    heatmap_path = os.path.join(WSI_FOLDER, f"{slide_name}_heatmap.jpg")
    cv2.imwrite(heatmap_path, heatmap_colored)

    # 计算每列平均值并添加一行记录
    df = pd.DataFrame(patch_features)
    column_averages = df.mean()
    column_averages.name = slide_name  # 将 WSI 名设为 index
    summary_records.append(column_averages)

# 将所有WSI的平均特征保存为 DataFrame，行为WSI，列为特征
summary_df = pd.DataFrame(summary_records)
summary_df.index.name = "WSI"
summary_csv_path = os.path.join(WSI_FOLDER, "wsi_feature_summary.csv")
summary_df.to_csv(summary_csv_path)
print(f"\n✅ 每个WSI每个特征的平均值已汇总保存至: {summary_csv_path}")
