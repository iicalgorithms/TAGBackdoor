import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from copy import deepcopy


def defense_prune(poisoned_data, prune_threshold, device, large_graph=True):
    """
    防御方法: 剪枝不相关边 (Prune Unrelated Edges)
    根据节点特征的余弦相似度来修剪图中的边。如果一条边的两个连接节点的特征相似度低于预设阈值，则移除这条边。
    此函数主要用于图数据的预处理或防御机制中，去除那些可能由攻击引入或与图结构不一致的边。

    参数:
    backdoored_data: 投毒后的图数据对象
    large_graph (bool): 是否为大图，用于选择不同的相似度计算策略以优化性能

    返回:
    处理后的图数据对象
    """

    edge_index = poisoned_data.train_edge_index
    x = poisoned_data.x

    # 如果没有边权重，创建全1的权重
    edge_weights = torch.ones(edge_index.shape[1])

    # 选择边权重大于0的边，并将其移动到指定设备
    valid_edge_mask = edge_weights > 0.0
    edge_index = edge_index[:, valid_edge_mask].to(device)
    edge_weights = edge_weights[valid_edge_mask].to(device)
    x = x.to(device)

    # 计算边的相似度
    if large_graph:
        edge_sims = torch.tensor([], dtype=torch.float).cpu()
        N = edge_index.shape[1]
        num_split = 100
        N_split = int(N / num_split)

        for i in range(num_split):
            if i == num_split - 1:
                edge_sim1 = F.cosine_similarity(
                    x[edge_index[0][N_split * i:]],
                    x[edge_index[1][N_split * i:]]
                ).cpu()
            else:
                edge_sim1 = F.cosine_similarity(
                    x[edge_index[0][N_split * i:N_split * (i + 1)]],
                    x[edge_index[1][N_split * i:N_split * (i + 1)]]
                ).cpu()
            edge_sims = torch.cat([edge_sims, edge_sim1])
    else:
        edge_sims = F.cosine_similarity(x[edge_index[0]], x[edge_index[1]])

    # 获取修剪阈值，如果配置中没有则使用默认值

    # 找到不相似的边并移除它们
    keep_mask = edge_sims > prune_threshold
    updated_edge_index = edge_index[:, keep_mask]
    updated_edge_weights = edge_weights[keep_mask]

    # 创建新的数据对象
    updated_data = deepcopy(poisoned_data)
    updated_data.train_edge_index=updated_edge_index
    updated_data.edge_weight=updated_edge_weights

    print(f"Pruning completed. Edges reduced from {edge_index.shape[1]} to {updated_edge_index.shape[1]}")
    return updated_data


def defense_reconstruct(poisoned_data, clean_data, rec_epochs, device):
    """
    防御方法: 基于重构误差修剪不相关边 (Reconstruction-based Edge Pruning)
    此函数结合了节点特征重构和边修剪。它首先训练一个自编码器来学习节点特征的表示，
    然后使用重构误差来识别异常节点。最后，移除那些连接到异常节点的边。
    这是一种更复杂的边修剪策略，旨在识别并移除可能由攻击引入的、与正常数据分布不符的结构。

    参数:
    backdoored_data: 投毒后的图数据对象
    clean_data: 原始清洁的图数据对象

    返回:
    处理后的图数据对象
    """

    class Autoencoder(nn.Module):
        def __init__(self, input_size):
            super(Autoencoder, self).__init__()
            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(input_size, 2 * input_size // 3),
                nn.ReLU(True),
                nn.Linear(2 * input_size // 3, input_size // 3),
                nn.ReLU(True)
            )
            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(input_size // 3, 2 * input_size // 3),
                nn.ReLU(True),
                nn.Linear(2 * input_size // 3, input_size),
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    class MLPAE(nn.Module):
        def __init__(self, ori_x, device, epochs):
            super(MLPAE, self).__init__()
            self.device = device
            self.model = Autoencoder(ori_x.shape[1]).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.criterion = nn.MSELoss()
            self.epochs = epochs
            self.ori_x = ori_x.to(device)

        def fit(self):
            self.model.train()
            for epoch in range(self.epochs):
                output = self.model(self.ori_x)
                loss = self.criterion(output, self.ori_x)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        def inference(self, input_data):
            self.model.eval()
            reconstruction_errors = []
            input_data = input_data.to(self.device)

            with torch.no_grad():
                for i in range(input_data.shape[0]):
                    sample = input_data[i:i + 1]
                    reconstructed = self.model(sample)
                    loss = self.criterion(reconstructed, sample)
                    reconstruction_errors.append(loss)

            reconstruction_errors_tensor = torch.stack(reconstruction_errors)
            return reconstruction_errors_tensor

    # 使用原始清洁数据训练自编码器
    poison_x = poisoned_data.x
    ori_x = clean_data.x

    # 初始化并训练自编码器
    AE = MLPAE(ori_x, device, rec_epochs)
    AE.fit()

    # 对投毒后的节点特征进行推理，得到重构误差分数
    rec_score_ori = AE.inference(poison_x)

    # 计算重构误差分数的第97百分位数作为阈值，用于识别异常节点
    threshold = np.percentile(rec_score_ori.detach().cpu().numpy(), 97)

    # 创建布尔掩码，标记重构误差大于阈值的节点（异常节点）
    mask = rec_score_ori > threshold

    # 获取边信息
    poison_edge_index = poisoned_data.train_edge_index
    if hasattr(poisoned_data, 'edge_weight') and poisoned_data.edge_weight is not None:
        poison_edge_weights = poisoned_data.edge_weight
    else:
        poison_edge_weights = torch.ones(poison_edge_index.shape[1])
    poison_edge_weights.to(device)

    # 保留边的掩码：如果一条边的两个端点都不是异常节点，则保留该边
    keep_edges_mask = ~(mask[poison_edge_index[0]] | mask[poison_edge_index[1]])
    keep_edges_mask.to(device)

    # 根据掩码过滤边索引和边权重
    filtered_poison_edge_index = poison_edge_index[:, keep_edges_mask]
    filtered_poison_edge_weights = poison_edge_weights[keep_edges_mask.to(poison_edge_weights.device)]

    # Create updated data object
    reconstructed_data = deepcopy(poisoned_data)
    reconstructed_data.train_edge_index = filtered_poison_edge_index
    reconstructed_data.edge_weight = filtered_poison_edge_weights

    return reconstructed_data