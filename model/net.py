import random
import torch
import torch.nn as nn
from model.SSFTT import SSFTTnet
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from sklearn.cluster import KMeans
from config.config_Houston import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model (nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.feature_encoder = SSFTTnet()
        in_features = 64
        self.cls = nn.Linear(in_features, self.num_classes, bias=True)

    def forward(self, data):
        output = self.feature_encoder(data)
        logit = self.cls(output)

        return output, logit


class Network (nn.Module):
    def __init__(self, num_classes):
        super(Network, self).__init__()
        self.num_classes = num_classes
        self.feature_encoder = SSFTTnet()
        in_features = 64
        self.cset_fc = nn.Linear(in_features, self.num_classes, bias=True)
        self.oset_fc = nn.Linear(in_features, self.num_classes * 2, bias=True)

        self.neighbor_eps = neighbor_eps
        self.num_samples = num_samples

    def forward(self, target_indices, x, labels, coord, x_2, feature_bank, coord_bank, domain='source'):
        if domain == 'source':
            output = self.feature_encoder(x)
            source_cset_logit = self.cset_fc(output)
            source_cset_prob = F.softmax(source_cset_logit, dim=1)
            source_oset_logit = self.oset_fc(output).view(x.size(0), 2, self.num_classes)
            source_oset_prob = F.softmax(source_oset_logit, dim=1)

            x_unknown = x_2
            output_unknown = self.feature_encoder(x_unknown)
            source_cset_unknown_logit = self.cset_fc(output_unknown)
            source_cset_unknown_prob = F.softmax(source_cset_unknown_logit, dim=1)
            source_oset_unknown_logit = self.oset_fc(output_unknown).view(x_unknown.size(0), 2, self.num_classes)
            source_oset_unknown_prob = F.softmax(source_oset_unknown_logit, dim=1)

            # 源域闭集分类器 交叉熵损失
            source_cset_loss = F.cross_entropy(source_cset_logit, labels)
            metric = {'source_cset_loss': source_cset_loss.item()}
            loss = source_cset_loss*source_loss_weights['source_cset_loss']

            # 一致性损失
            cc_know_loss = -torch.mean(source_cset_prob * source_oset_prob[:, 0, :])
            cc_unknow_loss = -torch.mean(source_cset_unknown_prob * source_oset_unknown_prob[:, 0, :])
            cc_loss = cc_know_loss + cc_unknow_loss
            loss += cc_loss*source_loss_weights['cc_loss']
            metric['cc_loss'] = cc_loss.item()

            # 源域开集分类器损失 已知样本困难样本采样损失
            source_oset_pos_target = torch.zeros_like(source_cset_logit)
            source_oset_pos_target[torch.arange(BATCH_SIZE), labels] = 1
            source_oset_pos_loss = torch.mean(
                torch.sum(-source_oset_pos_target * torch.log(source_oset_prob[:, 0, :] + 1e-8), dim=1))

            source_oset_neg_target = 1 - source_oset_pos_target
            source_oset_neg_loss = torch.mean(
                torch.max(-source_oset_neg_target * torch.log(source_oset_prob[:, 1, :] + 1e-8), dim=1)[0])

            source_oset_known_loss = source_oset_pos_loss + source_oset_neg_loss
            loss += source_oset_known_loss*source_loss_weights['source_oset_known_loss']
            metric['source_oset_known_loss'] = source_oset_known_loss.item()

            # 源域开集分类器损失 未知样本交叉熵损失
            unknown_logits = source_oset_unknown_logit.permute(0, 2, 1).reshape(-1, 2)
            unknown_targets = torch.ones(unknown_logits.size(0), dtype=torch.long, device=unknown_logits.device)
            source_oset_unknown_loss = F.cross_entropy(unknown_logits, unknown_targets)
            loss += source_oset_unknown_loss*source_loss_weights['source_oset_unknown_loss']
            metric['source_oset_unknown_loss'] = source_oset_unknown_loss.item()

            # 闭集分类器给出源域样本的预测类别
            pred = torch.max(source_cset_logit, dim=1)[1]

            return pred, loss, metric

        elif domain == 'target':
            output = self.feature_encoder(x)
            cset_logit = self.cset_fc(output)
            cset_prob = F.softmax(cset_logit, dim=1)
            oset_logit = self.oset_fc(output).view(x.size(0), 2, self.num_classes)
            oset_prob = F.softmax(oset_logit, dim=1)

            target_oset_label = torch.full((x.size(0),), -1, dtype=torch.long, device=device)
            clustered_mask = torch.zeros_like(target_oset_label, dtype=torch.bool)

            target_cset_pred = torch.max(cset_logit, dim=1)[1]
            feats = F.normalize(output, dim=1)
            metric = {}
            loss = 0.0

            for c in range(CLASS_NUM):
                class_idx = (target_cset_pred == c).nonzero(as_tuple=False).squeeze(-1)
                if class_idx.numel() < 30:
                    continue
                oset_prob_cls = oset_prob[class_idx, 1, c]
                unknown_mask = (oset_prob_cls > 0.5)
                unknown_ratio = unknown_mask.float().mean().item()

                if unknown_ratio > UNK_RATIO_THRESH:
                    feats_cls = feats[class_idx]
                    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(feats_cls.detach().cpu().numpy())
                    cluster_labels = torch.tensor(kmeans.labels_, device=device)

                    prob_cluster0 = oset_prob_cls[cluster_labels == 0].mean().item()
                    prob_cluster1 = oset_prob_cls[cluster_labels == 1].mean().item()

                    unknown_cluster = 0 if prob_cluster0 > prob_cluster1 else 1
                    # 找到了类c的已知样本索引和未知样本索引
                    known_idx = class_idx[cluster_labels != unknown_cluster]
                    unknown_idx = class_idx[cluster_labels == unknown_cluster]

                    target_oset_label[known_idx] = 0  # 已知
                    target_oset_label[unknown_idx] = 1  # 未知
                    clustered_mask[class_idx] = True

            valid_idx = clustered_mask.nonzero(as_tuple=False).squeeze(-1)

            if valid_idx.numel() > 0:
                chosen_logits = oset_prob[valid_idx, :, target_cset_pred[valid_idx]]  # [N_valid, 2]
                target_cluster_loss = F.cross_entropy(chosen_logits, target_oset_label[valid_idx])
            else:
                target_cluster_loss = torch.tensor(0.0, device=device)

            # 累加到总损失
            loss += target_cluster_loss*target_loss_weights['target_cluster_loss']   # DL*0.1
            metric['target_cluster_loss'] = target_cluster_loss.item()

            known_mask = labels < CLASS_NUM
            unknown_mask = labels == CLASS_NUM

            if known_mask.sum() > 0:
                # 目标域闭集分类器交叉熵损失
                target_cset_loss = F.cross_entropy(cset_logit[known_mask], labels[known_mask])
                loss += target_cset_loss*target_loss_weights['target_cset_loss']
                metric['target_cset_loss'] = target_cset_loss.item()

            # 目标域开集分类器损失 未知样本损失
            if unknown_mask.sum() > 0:
                oset_logit_unknown = oset_logit[unknown_mask]  # [N_unknown, 2, num_classes]
                oset_logit_unknown_flat = oset_logit_unknown.permute(0, 2, 1).reshape(-1, 2)
                unknown_targets = torch.ones(
                    oset_logit_unknown_flat.size(0),
                    dtype=torch.long,
                    device=oset_logit_unknown.device
                )
                target_oset_unknown_loss = F.cross_entropy(oset_logit_unknown_flat, unknown_targets)

                # 累加进总损失
                loss += target_oset_unknown_loss*target_loss_weights['target_oset_unknown_loss']
                metric['target_oset_unknown_loss'] = target_oset_unknown_loss.item()

            # 一致性损失
            cc_loss = -torch.mean(cset_prob * oset_prob[:, 0, :])
            loss += cc_loss*target_loss_weights['cc_loss']
            metric['cc_loss'] = cc_loss.item()

            # 邻域一致性表达损失
            tt_sim = F.normalize(output, dim=1) @ F.normalize(feature_bank, dim=1).t()
            tt_mask_instance = torch.zeros_like(tt_sim)
            tt_mask_instance[torch.arange(x.size(0)), target_indices] = 1
            tt_mask_instance = tt_mask_instance.bool()
            tt_sim = (tt_sim + 1.) * (~ tt_mask_instance) - 1.

            tt_nearest = torch.max(tt_sim, dim=1, keepdim=True)[0]
            tt_mask_neighbor = tt_sim > (tt_nearest * self.neighbor_eps)
            tt_num_neighbor = torch.sum(tt_mask_neighbor, dim=1) # 样本的邻居数量

            # 空间平方距离
            dist2 = torch.cdist(coord, coord_bank, p=2) ** 2  # [B, M]
            tt_neighbor_score = torch.exp(-dist2)

            tt_sim_exp = torch.exp(tt_sim)
            tt_score = tt_sim_exp / torch.sum(tt_sim_exp, dim=1, keepdim=True)
            tt_neighbor_loss = torch.sum(
                -torch.log(tt_score + 1e-8) * tt_neighbor_score * tt_mask_neighbor,
                dim=1) / tt_num_neighbor
            neighbor_loss = torch.mean(tt_neighbor_loss)  # *0.1
            feature_bank[target_indices] = output.detach()

            loss += neighbor_loss*target_loss_weights['neighbor_loss']
            metric['neighbor_loss'] = neighbor_loss.item()

            return loss, metric

        elif domain == 'test':
            output = self.feature_encoder(x)
            target_cset_logit = self.cset_fc(output)
            target_cset_prob = F.softmax(target_cset_logit, dim=1)
            target_oset_logit = self.oset_fc(output).view(x.size(0), 2, self.num_classes)
            target_oset_prob = F.softmax(target_oset_logit, dim=1)

            target_cset_pred = torch.max(target_cset_logit, dim=1)[1]
            target_oset_pred = target_oset_prob[torch.arange(x.size(0)), 1, target_cset_pred] > 0.5

            cset_preds = np.asarray(target_cset_pred.tolist())
            oset_preds = np.asarray(target_oset_pred.tolist())
            cset_preds[oset_preds] = unknown_class
            pred = cset_preds

            return pred, target_cset_logit, target_oset_logit, output

