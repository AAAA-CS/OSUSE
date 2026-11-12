import os
import random
import numpy as np
import torch
import torch.nn as nn
from model.net import Model
import time
import torch.nn.functional as F
import utils
from torch.utils.data import TensorDataset, DataLoader
from config.config_Houston import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
utils.set_seed(seeds[0])

data_s, label_s = utils.load_data(data_path_s, label_path_s)
trainX, trainY = utils.get_sample_data(data_s, label_s, HalfWidth, 500)
unlabel_trainX = utils.get_unlabel_data(data_s, label_s, HalfWidth)

train_dataset = TensorDataset(torch.tensor(trainX), torch.tensor(trainY))
train_loader_s = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
len_source_loader = len(train_loader_s)

unlabel_dataset = TensorDataset(torch.tensor(unlabel_trainX))
unlabel_loader = DataLoader(unlabel_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

print("Training...")

# model
net = Model(CLASS_NUM).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
net.train()

for epoch in range(1, unknow_epochs + 1):
    total_hit = 0.0
    size = 0.0

    for data, labels in train_loader_s:
        features, logit = net(data.to(device))
        pred = torch.max(logit, dim=1)[1]

        cls_loss = F.cross_entropy(logit, labels.to(device))
        optimizer.zero_grad()
        cls_loss.backward()
        optimizer.step()

        total_hit += pred.eq(labels.data.to(device)).sum()
        size += labels.data.size()[0]

    test_accuracy = 100. * float(total_hit) / size
    print('epoch {:>3d}: cset_loss: {:6.4f}, acc: {:6.4f}'
          .format(epoch, cls_loss.item(), test_accuracy, ))

print('network training is completed')

net.eval()
train_features = []
train_labels = []

with torch.no_grad():
    for data, labels in train_loader_s:
        features, _ = net(data.to(device))
        train_features.append(features.cpu())
        train_labels.append(labels.cpu())

train_features = torch.cat(train_features, dim=0)
train_label = torch.cat(train_labels, dim=0)

unique, counts = torch.unique(train_label, return_counts=True)
for u, c in zip(unique.tolist(), counts.tolist()):
    print(f"类别 {u}: {c} 个样本")

class_means = torch.zeros((CLASS_NUM, feat_dim), dtype=torch.float32)
class_thresholds = []

for c in range(CLASS_NUM):
    mask = (train_label == c)
    class_feats = train_features[mask]
    mean_feat = class_feats.mean(dim=0)
    class_means[c] = mean_feat

    dists = torch.norm(class_feats - mean_feat, dim=1)
    l = dists.max()
    class_thresholds.append(l)

class_means = class_means.to(device)
class_thresholds_tensor = torch.tensor(class_thresholds, device=device, dtype=torch.float32)


unknown_samples = []
conf_thresh = 0.5
with torch.no_grad():
    for (data,) in unlabel_loader:
        feats, logit = net(data.to(device))
        dists = torch.cdist(feats, class_means)

        probs = torch.softmax(logit, dim=1)  # 转为概率
        max_probs, preds = torch.max(probs, dim=1)  # 每个样本的最大置信度和预测类

        far_mask = (dists > class_thresholds_tensor.unsqueeze(0)).all(dim=1)  # [B]
        low_conf_mask = (max_probs < conf_thresh)  # [B]
        unknown_mask = far_mask & low_conf_mask

        if unknown_mask.any():
            unknown_samples.extend(data[unknown_mask].cpu().numpy())

print("未知样本数量：", len(unknown_samples))

# 把获得的未知样本保存起来
unknown_samples = np.array(unknown_samples)
np.save("data/unknown_samples_hs.npy", unknown_samples)
print("未知样本已保存到data/unknown_samples_hs.npy")