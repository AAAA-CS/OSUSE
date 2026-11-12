import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch
import numpy as np
from itertools import cycle
import torch.nn.functional as F
from model.net import Network
import utils
from torch.utils.data import TensorDataset, DataLoader
from config.config_Houston import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_t, label_t = utils.load_data(data_path_t, label_path_t)

unknown_acc = np.zeros(nDataSet)
known_acc = np.zeros(nDataSet)
hscore = np.zeros(nDataSet)
per_class_acc_all = np.zeros((nDataSet, SHARED_CLASS_NUM))

for iDataSet in range(nDataSet):
    print('#######################idataset######################## ', iDataSet)
    utils.set_seed(seeds[iDataSet])

    testID, testX, testY, G, RandPerm, Row, Column = utils.get_all_data(data_t, label_t, HalfWidth)
    coords = utils.get_coords_from_RandPerm(RandPerm, Row, Column)
    print("坐标 shape:", coords.shape)

    print("标签集合（testY）:", np.unique(testY))
    classes, counts = np.unique(testY, return_counts=True)
    class_counts = dict(zip(classes, counts))
    print("每个类的样本数量:", class_counts)

    test_dataset = TensorDataset(torch.tensor(testX), torch.tensor(testY), torch.tensor(coords))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    print("Training...")

    # model
    model = Network(CLASS_NUM).to(device)
    # 加载预训练参数
    checkpoint_path = r"saved_models\model-HS.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    with torch.no_grad():
        source_prototypes = F.normalize(model.cset_fc.weight.clone(), dim=1)

    pseudo_loader = None
    for epoch in range(1, epochs + 1):
        running_losses = []
        running_target_cluster_loss = []
        running_cset_loss = []
        running_cc_loss = []
        running_neighbor_loss = []

        if epoch == 1 or epoch % 5 == 0:
            preds_all, dist_scores_all, closed_preds_all, confidences_all = [], [], [], []
            features_all = []
            target_prototypes = torch.zeros_like(source_prototypes).to(device)
            proto_counts = torch.zeros(CLASS_NUM).to(device)
            model.eval()
            with torch.no_grad():
                for i, (test_data, test_label, test_coord) in enumerate(test_loader):
                    pred, target_cset_logit, target_oset_logit, feats = model(
                        None, test_data.to(device), None, None, None, None, None, domain="test")

                    probs = F.softmax(target_cset_logit, dim=1)
                    confs, closed_preds = probs.max(dim=1)  # # 获取最大概率值和对应的类别索引

                    for c in range(CLASS_NUM):
                        mask = (pred == c)
                        n_b = int(mask.sum().item())
                        if n_b == 0:
                            continue
                        feat_b = feats[mask].mean(dim=0)
                        N_prev = int(proto_counts[c].item())
                        if N_prev == 0:
                            target_prototypes[c] = feat_b
                            proto_counts[c] = n_b
                        else:
                            mu_prev = target_prototypes[c]
                            mu_new = (mu_prev * N_prev + feat_b * n_b) / (N_prev + n_b)
                            target_prototypes[c] = mu_new
                            proto_counts[c] = N_prev + n_b

                    feats_norm = F.normalize(feats, dim=1)
                    source_proto_batch = F.normalize(source_prototypes[closed_preds], dim=1)
                    target_proto_batch = F.normalize(target_prototypes[closed_preds], dim=1)

                    d_s = torch.norm(feats_norm - source_proto_batch, dim=1)
                    d_t = torch.norm(feats_norm - target_proto_batch, dim=1)
                    d = (2 * (d_s + 1) * (d_t + 1)) / ((d_s + 1) + (d_t + 1))

                    preds_all.append(torch.from_numpy(pred))
                    closed_preds_all.append(closed_preds.cpu())
                    dist_scores_all.append(-d.cpu())  # 越小越好，取负方便排序
                    confidences_all.append(confs.cpu())
                    features_all.append(feats.cpu())

                preds_all = torch.cat(preds_all)
                closed_preds_all = torch.cat(closed_preds_all)
                dist_scores_all = torch.cat(dist_scores_all)
                confidences_all = torch.cat(confidences_all)
                features_all = torch.cat(features_all, dim=0)

                selected_indices = []

                known_pred_mask = (preds_all >= 0) & (preds_all < CLASS_NUM)
                known_indices_all = torch.where(known_pred_mask)[0]
                preds_known = closed_preds_all[known_indices_all]
                dist_known = dist_scores_all[known_indices_all]
                confs_known = confidences_all[known_indices_all]

                # 每个类分别取高置信 + 小距离样本
                for c in range(CLASS_NUM):
                    class_mask = (preds_known == c)
                    if class_mask.sum() == 0:
                        continue
                    cls_idx = known_indices_all[class_mask]
                    cls_dist = dist_known[class_mask]
                    cls_conf = confs_known[class_mask]

                    # 先根据置信度取前K%
                    k_conf = max(1, int(len(cls_conf) * K))
                    _, conf_top_idx = torch.topk(cls_conf, k_conf)
                    selected_by_conf = cls_idx[conf_top_idx]

                    # 在这些中进一步按距离取前50%
                    cls_dist_sel = cls_dist[conf_top_idx]
                    k_dist = max(1, int(len(cls_dist_sel) * 0.5))
                    _, dist_top_idx = torch.topk(-cls_dist_sel, k_dist)
                    final_sel = selected_by_conf[dist_top_idx]

                    selected_indices.append(final_sel)

                unknown_mask = preds_all == CLASS_NUM
                if unknown_mask.sum() > 0:
                    unknown_idx = torch.where(unknown_mask)[0]
                    unknown_confs = confidences_all[unknown_idx]
                    k_unknown = max(1, int(len(unknown_confs) * K/2))
                    # 取置信度最低的前K%样本
                    _, low_idx = torch.topk(unknown_confs, k_unknown, largest=False)
                    selected_unknown = unknown_idx[low_idx]
                    selected_indices.append(selected_unknown)

                if len(selected_indices) == 0:
                    continue

                selected_indices = torch.cat(selected_indices)

                labels_all = closed_preds_all.clone()
                labels_sel = labels_all[selected_indices].clone()
                unknown_in_sel_mask = (preds_all[selected_indices] == CLASS_NUM)
                if unknown_in_sel_mask.any():
                    labels_sel[unknown_in_sel_mask] = CLASS_NUM

                # 统计已知类与未知类伪标签数量
                known_count = (labels_sel < CLASS_NUM).sum().item()
                unknown_count = (labels_sel == CLASS_NUM).sum().item()

                print(f"Epoch {epoch}: 选取 {len(selected_indices)} 个伪标签样本（已知类 {known_count}，未知类 {unknown_count}）")

                pseudo_data = test_loader.dataset.tensors[0][selected_indices]
                pseudo_labels = labels_sel
                pseudo_feature = features_all[selected_indices]
                pseudo_coords = test_loader.dataset.tensors[2][selected_indices]
                N = pseudo_data.shape[0]
                indices = torch.arange(N)

                pseudo_dataset = torch.utils.data.TensorDataset(indices, pseudo_data, pseudo_labels, pseudo_coords)
                pseudo_loader = torch.utils.data.DataLoader(pseudo_dataset, batch_size=BATCH_SIZE, shuffle=True)

                coord_dim = pseudo_coords.shape[1]
                feature_bank = torch.zeros(N, feat_dim, device=device)
                coord_bank = torch.zeros(N, coord_dim, device=device)
                feature_bank.copy_(pseudo_feature)
                coord_bank.copy_(pseudo_coords)

        model.train()
        for idx, data, label, coord in pseudo_loader:

            loss, metric = model(idx.to(device),
                                 data.to(device),
                                 label.to(device),
                                 coord.to(device).float(),
                                 None,
                                 feature_bank,
                                 coord_bank,
                                 domain="target")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_losses.append(loss.item())
            running_target_cluster_loss.append(metric.get('target_cluster_loss', 0.0))
            running_cset_loss.append(metric.get('target_cset_loss', 0.0))
            running_cc_loss.append(metric.get('cc_loss', 0.0))
            running_neighbor_loss.append(metric.get('neighbor_loss', 0.0))

        print(
            'epoch {:>3d}: cset_loss: {:6.4f}, cluster_loss: {:6.4f}, cc_loss: {:6.4f}, neighbor_loss: {:6.4f}, total_loss: {:6.4f}'.format(
                epoch,
                float(np.mean(running_cset_loss)) if running_cset_loss else 0.0,
                float(np.mean(running_target_cluster_loss)) if running_target_cluster_loss else 0.0,
                float(np.mean(running_cc_loss)) if running_cc_loss else 0.0,
                float(np.mean(running_neighbor_loss)) if running_neighbor_loss else 0.0,
                float(np.mean(running_losses)) if running_losses else 0.0
            ))

        if epoch % epochs == 0:
            # print("Testing ...")
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for test_datas, test_labels, coord in test_loader:
                    batch_size = test_labels.shape[0]
                    pred, target_cset_prob, target_oset_prob, feats = model(
                        None,
                        test_datas.to(device),
                        None,
                        None,
                        None,
                        None,
                        None,
                        domain="test")
                    all_preds.extend(pred)
                    all_labels.extend(test_labels.tolist())

            all_preds = np.array(all_preds)
            print("每类预测样本的数量", np.bincount(all_preds))

            all_labels = np.asarray(all_labels)
            unknown_indices = np.where(all_labels == unknown_class)[0]  # 未知类别的样本索引
            known_indices = np.where(all_labels != unknown_class)[0]  # 已知类别的样本索引
            # 计算未知类别的识别准确率

            unknown_correct = np.where(all_preds[unknown_indices] == unknown_class)[0]
            unk_acc = len(unknown_correct) / len(unknown_indices) if len(unknown_indices) > 0 else 0.
            print("正确预测为未知类的样本数量", len(unknown_correct))
            print("未知类的样本总数量", len(unknown_indices))

            # 计算已知类别的准确率
            known_correct = np.where(all_preds[known_indices] == all_labels[known_indices])[0]
            print("每类正确预测的样本数量", np.bincount(all_labels[known_indices][known_correct], minlength=SHARED_CLASS_NUM))
            print("每类包含的总体样本数量", np.bincount(all_labels[known_indices], minlength=SHARED_CLASS_NUM))
            # 每个已知类的准确率
            per_class_acc = np.bincount(all_labels[known_indices][known_correct], minlength=SHARED_CLASS_NUM) / \
                            np.bincount(all_labels[known_indices], minlength=SHARED_CLASS_NUM)
            per_class_acc_all[iDataSet, :] = per_class_acc
            print("每类准确率：")
            for i, acc in enumerate(per_class_acc, start=1):
                print(f"类{i}：{acc:.4f}")

            knw_acc = np.mean(np.bincount(all_labels[known_indices][known_correct], minlength=SHARED_CLASS_NUM) / \
                                np.bincount(all_labels[known_indices], minlength=SHARED_CLASS_NUM))

            # 计算H-score（调和平均）
            h = 2. * knw_acc * unk_acc / (knw_acc + unk_acc + 1e-8)

            print(f"Known Acc: {knw_acc:.4f}")
            print(f"Unknown Acc: {unk_acc:.4f}")
            print(f"H-Score: {h:.4f}")

            unknown_acc[iDataSet] = unk_acc
            known_acc[iDataSet] = knw_acc
            hscore[iDataSet] = h

print("====== 每轮结果（Per iDataSet） ======")
for i in range(len(known_acc)):
    print(f"iDataSet {i+1:2d}:  Known Acc={known_acc[i]*100:.4f}  "
          f"Unknown Acc={unknown_acc[i]*100:.4f}  "
          f"H-Score={hscore[i]*100:.4f}")

# 计算每类指标的均值与标准差
unknown_acc_Mean = np.mean(unknown_acc)
unknown_acc_Std = np.std(unknown_acc)

known_acc_Mean = np.mean(known_acc)
known_acc_Std = np.std(known_acc)

hscore_Mean = np.mean(hscore)
hscore_Std = np.std(hscore)
per_class_mean = np.mean(per_class_acc_all, axis=0)
per_class_std = np.std(per_class_acc_all, axis=0)

print("\n====== 每类准确率均值 ± 标准差（Across iDataSets）======")
for c in range(SHARED_CLASS_NUM):
    print(f"类 {c:2d}: {per_class_mean[c]*100:.2f}±{per_class_std[c]*100:.2f}")
print(f"Unknown Acc: {unknown_acc_Mean * 100:.2f}±{unknown_acc_Std * 100:.2f}")
print(f"Known Acc: {known_acc_Mean*100:.2f}±{known_acc_Std*100:.2f}")
print(f"H-Score: {hscore_Mean*100:.2f}±{hscore_Std*100:.2f}")
