import os
import numpy as np
import torch
import torch.nn as nn
from model.net import Network
import time
import utils
from itertools import cycle
from torch.utils.data import TensorDataset, DataLoader
from config.config_Houston import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
utils.set_seed(1336)

data_s, label_s = utils.load_data(data_path_s, label_path_s)
trainX, trainY = utils.get_sample_data(data_s, label_s, HalfWidth, label_num)
classes, counts = np.unique(trainY, return_counts=True)
class_counts = dict(zip(classes, counts))
print("每个类的数量:", class_counts)

train_dataset = TensorDataset(torch.tensor(trainX), torch.tensor(trainY))
train_loader_s = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

unknown_samples = np.load("data/unknown_samples_hs.npy")
unknown_dataset = TensorDataset(torch.tensor(unknown_samples, dtype=torch.float32))
unknown_loader = DataLoader(unknown_dataset, batch_size=128, shuffle=True, drop_last=True)

print("Training...")

# model
Net = Network(CLASS_NUM).to(device)
optimizer_Net = torch.optim.Adam(Net.parameters(), lr=LEARNING_RATE)
Net.train()

unk_cycle = cycle(unknown_loader)
for epoch in range(1, pre_epochs + 1):
    total_hit = 0.0
    size = 0.0
    iter_unk = iter(unknown_loader)

    for data, labels in train_loader_s:
        (unk_data,) = next(unk_cycle)
        pred, loss_Net, metric = Net(
            None,
            data.to(device),
            labels.to(device),
            None,
            unk_data.to(device),
            None,
            None)

        optimizer_Net.zero_grad()
        loss_Net.backward()
        optimizer_Net.step()

        total_hit += pred.eq(labels.data.to(device)).sum()
        size += labels.data.size()[0]

    test_accuracy = 100. * float(total_hit) / size
    print('epoch {:>3d}:source_cset_loss: {:6.4f}, source_oset_known_loss: {:6.4f}, cc_loss: {:6.4f}, source_oset_unknown_loss: {:6.4f}, acc: {:6.4f}, loss_net: {:6.4f}'
          .format(epoch, metric['source_cset_loss'], metric['source_oset_known_loss'], metric['cc_loss'], metric['source_oset_unknown_loss'], test_accuracy, loss_Net.item()))

save_dir = 'saved_models'
os.makedirs(save_dir, exist_ok=True)
Net_save_path = os.path.join(save_dir, f'model-HS.pth')
torch.save(Net.state_dict(), Net_save_path)
print(f"模型已成功保存到：{Net_save_path}")
