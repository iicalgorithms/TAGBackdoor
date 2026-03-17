import argparse
import numpy as np
import torch

from torch_geometric.datasets import Planetoid, Flickr, Amazon
from ogb.nodeproppred import PygNodePropPredDataset

# from torch_geometric.loader import DataLoader

from select_sample import select

import numpy as np  # 导入 NumPy 库，用于数值计算
import torch.nn.functional as F  # 导入 PyTorch 的神经网络函数库，例如 cosine_similarity
from torch_geometric.utils import to_dense_adj, dense_to_sparse  # 从 PyTorch Geometric 库导入图相关的工具函数，用于稠密邻接矩阵和稀疏邻接矩阵之间的转换
import torch  # 导入 PyTorch 深度学习框架
import scipy.sparse as sp  # 导入 SciPy 库中的稀疏矩阵模块
from reconstruct import MLPAE  # 从自定义的重构模型中导入 MLPAE (多层感知机自编码器)


def edge_sim_analysis(edge_index, features):  # 定义函数 edge_sim_analysis，用于分析边的相似性
    """
    函数功能说明: edge_sim_analysis (边相似性分析)
    计算图中所有边的余弦相似度，并返回这些相似度值。
    主要用于分析图中边的连接特性，例如边的平均相似度，以及相似度较低的边的比例。

    参数:
    edge_index (Tensor): 边的索引，形状为 [2, num_edges]，表示边的连接关系。
    features (Tensor): 节点的特征矩阵，形状为 [num_nodes, feature_dim]。

    返回:
    numpy.ndarray: 包含所有边余弦相似度的 NumPy 数组。
    """
    sims = []  # 初始化一个空列表，用于存储边的相似度
    for (u, v) in edge_index:  # 遍历每一条边 (u, v)
        sims.append(float(F.cosine_similarity(features[u].unsqueeze(0), features[v].unsqueeze(
            0))))  # 计算节点 u 和节点 v 特征之间的余弦相似度，并添加到列表中. unsqueeze(0) 用于将一维张量变为二维，以符合 cosine_similarity 的输入要求
    sims = np.array(sims)  # 将相似度列表转换为 NumPy 数组
    # print(f"mean: {sims.mean()}, <0.1: {sum(sims<0.1)}/{sims.shape[0]}")  # 打印平均相似度和相似度小于0.1的边的数量及比例 (已注释)
    return sims  # 返回计算得到的边的相似度数组


def prune_unrelated_edge(args, edge_index, edge_weights, x, device,
                         large_graph=True):  # 定义函数 prune_unrelated_edge，用于修剪不相关的边
    """
    函数功能说明: prune_unrelated_edge (修剪不相关边)
    根据节点特征的余弦相似度来修剪图中的边。如果一条边的两个连接节点的特征相似度低于预设阈值 (args.prune_thr)，则移除这条边。
    此函数主要用于图数据的预处理或防御机制中，去除那些可能由攻击引入或与图结构不一致的边。

    参数:
    args (Namespace): 包含各种参数的对象，如 prune_thr (修剪阈值)。
    edge_index (Tensor): 原始边的索引。
    edge_weights (Tensor): 原始边的权重。
    x (Tensor): 节点的特征矩阵。
    device (torch.device): 计算设备 (CPU 或 GPU)。
    large_graph (bool): 是否为大图，用于选择不同的相似度计算策略以优化性能。

    返回:
    tuple: 包含更新后的边索引 (updated_edge_index) 和边权重 (updated_edge_weights) 的元组。
    """
    edge_index = edge_index[:, edge_weights > 0.0].to(device)  # 选择边权重大于0的边，并将其移动到指定设备
    edge_weights = edge_weights[edge_weights > 0.0].to(device)  # 选择边权重大于0的边权重，并将其移动到指定设备
    x = x.to(device)  # 将节点特征移动到指定设备
    # calculate edge simlarity  # 计算边的相似度
    if (large_graph):  # 如果是大图
        edge_sims = torch.tensor([], dtype=float).cpu()  # 初始化一个空的 CPU 张量用于存储边的相似度
        N = edge_index.shape[1]  # 获取边的数量
        num_split = 100  # 定义分割的块数，用于分批计算相似度以节省内存
        N_split = int(N / num_split)  # 计算每块中边的数量
        for i in range(num_split):  # 遍历每一块
            if (i == num_split - 1):  # 如果是最后一块
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:]],
                                                x[edge_index[1][N_split * i:]]).cpu()  # 计算该块内边的余弦相似度，并移至 CPU
            else:  # 如果不是最后一块
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:N_split * (i + 1)]], x[
                    edge_index[1][N_split * i:N_split * (i + 1)]]).cpu()  # 计算该块内边的余弦相似度，并移至 CPU
            # print(edge_sim1)  # 打印当前块的相似度 (已注释)
            edge_sim1 = edge_sim1.cpu()  # 确保相似度在 CPU 上
            edge_sims = torch.cat([edge_sims, edge_sim1])  # 将当前块的相似度拼接到总的相似度张量中
        # edge_sims = edge_sims.to(device)  # 将所有边的相似度移回设备 (已注释，因为后续操作在 CPU 上进行)
    else:  # 如果不是大图
        edge_sims = F.cosine_similarity(x[edge_index[0]], x[edge_index[1]])  # 直接计算所有边的余弦相似度
    # find dissimilar edges and remote them  # 找到不相似的边并移除它们
    # update structure  # 更新图结构
    updated_edge_index = edge_index[:, edge_sims > args.prune_thr]  # 选择相似度大于修剪阈值的边索引
    updated_edge_weights = edge_weights[edge_sims > args.prune_thr]  # 选择相似度大于修剪阈值的边权重
    return updated_edge_index, updated_edge_weights  # 返回更新后的边索引和边权重


def prune_unrelated_edge_isolated(args, edge_index, edge_weights, x, device,
                                  large_graph=True):  # 定义函数 prune_unrelated_edge_isolated，用于修剪不相关的边并找出受影响的节点
    """
    函数功能说明: prune_unrelated_edge_isolated (修剪不相关边并识别孤立节点)
    与 `prune_unrelated_edge` 类似，此函数也根据节点特征的余弦相似度修剪边。
    不同之处在于，它不是直接移除不相似的边，而是将这些边的权重设为0，并返回那些因为边被移除（权重变0）而可能变得“孤立”或连接性减弱的节点列表。

    参数:
    args (Namespace): 包含各种参数的对象，如 prune_thr (修剪阈值)。
    edge_index (Tensor): 原始边的索引。
    edge_weights (Tensor): 原始边的权重。
    x (Tensor): 节点的特征矩阵。
    device (torch.device): 计算设备 (CPU 或 GPU)。
    large_graph (bool): 是否为大图，用于选择不同的相似度计算策略。

    返回:
    tuple: 包含更新后的边索引 (updated_edge_index)、边权重 (updated_edge_weights) 以及因边修剪而受影响的节点列表 (dissim_nodes) 的元组。
    """
    edge_index = edge_index[:, edge_weights > 0.0].to(device)  # 选择边权重大于0的边，并将其移动到指定设备
    edge_weights = edge_weights[edge_weights > 0.0].to(device)  # 选择边权重大于0的边权重，并将其移动到指定设备
    x = x.to(device)  # 将节点特征移动到指定设备
    # calculate edge simlarity  # 计算边的相似度
    if (large_graph):  # 如果是大图
        edge_sims = torch.tensor([], dtype=float).cpu()  # 初始化一个空的 CPU 张量用于存储边的相似度
        N = edge_index.shape[1]  # 获取边的数量
        num_split = 100  # 定义分割的块数
        N_split = int(N / num_split)  # 计算每块中边的数量
        for i in range(num_split):  # 遍历每一块
            if (i == num_split - 1):  # 如果是最后一块
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:]],
                                                x[edge_index[1][N_split * i:]]).cpu()  # 计算该块内边的余弦相似度，并移至 CPU
            else:  # 如果不是最后一块
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:N_split * (i + 1)]], x[
                    edge_index[1][N_split * i:N_split * (i + 1)]]).cpu()  # 计算该块内边的余弦相似度，并移至 CPU
            # print(edge_sim1)  # 打印当前块的相似度 (已注释)
            edge_sim1 = edge_sim1.cpu()  # 确保相似度在 CPU 上
            edge_sims = torch.cat([edge_sims, edge_sim1])  # 将当前块的相似度拼接到总的相似度张量中
        # edge_sims = edge_sims.to(device)  # 将所有边的相似度移回设备 (已注释)
    else:  # 如果不是大图
        # calculate edge simlarity  # 计算边的相似度
        edge_sims = F.cosine_similarity(x[edge_index[0]], x[edge_index[1]])  # 直接计算所有边的余弦相似度
    # find dissimilar edges and remote them  # 找到不相似的边并移除它们
    dissim_edges_index = np.where(edge_sims.cpu() <= args.prune_thr)[0]  # 找到相似度小于等于修剪阈值的边的索引
    edge_weights[dissim_edges_index] = 0  # 将这些不相似边的权重设为0
    # select the nodes between dissimilar edgesy  # 选择这些不相似边连接的节点
    dissim_edges = edge_index[:, dissim_edges_index]  # 获取不相似边的索引，格式为 [[v_1,v_2,...],[u_1,u_2,...]]
    dissim_nodes = torch.cat([dissim_edges[0], dissim_edges[1]]).tolist()  # 将不相似边的两端节点合并并转换为列表
    dissim_nodes = list(set(dissim_nodes))  # 去除重复的节点，得到受影响的节点列表
    # update structure  # 更新图结构
    updated_edge_index = edge_index[:, edge_weights > 0.0]  # 选择边权重大于0的边索引 (即移除了权重为0的边)
    updated_edge_weights = edge_weights[edge_weights > 0.0]  # 选择边权重大于0的边权重
    return updated_edge_index, updated_edge_weights, dissim_nodes  # 返回更新后的边索引、边权重和受影响的节点列表


def select_target_nodes(args, seed, model, features, edge_index, edge_weights, labels, idx_val,
                        idx_test):  # 定义函数 select_target_nodes，用于选择目标节点进行攻击和评估
    """
    函数功能说明: select_target_nodes (选择目标节点)
    该函数用于在给定的验证集和测试集中，根据模型的预测结果和预设参数，选择用于后门攻击的目标测试节点、干净的测试节点以及用于投毒的训练节点。

    参数:
    args (Namespace): 包含各种参数的对象，如 target_class (目标类别), target_test_nodes_num (目标测试节点数量), clean_test_nodes_num (干净测试节点数量), vs_ratio (投毒比例)。
    seed (int): 随机种子，用于保证结果的可复现性。
    model (torch.nn.Module): 预训练的图神经网络模型。
    features (Tensor): 节点特征矩阵。
    edge_index (Tensor): 边索引。
    edge_weights (Tensor): 边权重。
    labels (Tensor): 节点的真实标签。
    idx_val (Tensor): 验证集节点的索引。
    idx_test (Tensor): 测试集节点的索引。

    返回:
    tuple: 包含目标攻击测试节点列表 (atk_test_nodes)、干净测试节点列表 (clean_test_nodes) 和投毒训练节点列表 (poi_train_nodes) 的元组。
    """
    test_ca, test_correct_index = model.test_with_correct_nodes(features, edge_index, edge_weights, labels,
                                                                idx_test)  # 使用模型测试测试集，获取分类正确节点的索引
    test_correct_index = test_correct_index.tolist()  # 将正确索引转换为列表
    '''select target test nodes'''  # 选择目标测试节点
    test_correct_nodes = idx_test[test_correct_index].tolist()  # 获取测试集中被模型正确分类的节点列表
    # filter out the test nodes that are not in target class  # 筛选出不在目标类别中的测试节点
    target_class_nodes_test = [int(nid) for nid in idx_test
                               if labels[nid] == args.target_class]  # 获取测试集中属于目标类别的节点列表
    # get the target test nodes  # 获取目标测试节点
    idx_val, idx_test = idx_val.tolist(), idx_test.tolist()  # 将验证集和测试集索引转换为列表
    rs = np.random.RandomState(seed)  # 初始化随机数生成器，使用指定种子
    cand_atk_test_nodes = list(set(test_correct_nodes) - set(target_class_nodes_test))  # 候选的攻击测试节点：被正确分类且不属于目标类别的测试节点
    atk_test_nodes = rs.choice(cand_atk_test_nodes, args.target_test_nodes_num,
                               replace=False)  # 从候选节点中随机选择指定数量的攻击测试节点 (不重复选择，如果候选不足会报错)
    '''select clean test nodes'''  # 选择干净的测试节点
    cand_clean_test_nodes = list(set(idx_test) - set(atk_test_nodes))  # 候选的干净测试节点：测试集中排除已选为攻击目标的节点
    clean_test_nodes = rs.choice(cand_clean_test_nodes, args.clean_test_nodes_num,
                                 replace=False)  # 从候选节点中随机选择指定数量的干净测试节点 (不重复选择)
    '''select poisoning nodes from unlabeled nodes (assign labels is easier than change, also we can try to select from labeled nodes)'''  # 从无标签节点中选择投毒节点 (注释说明：分配标签比更改标签更容易，也可以尝试从有标签节点中选择)
    N = features.shape[0]  # 获取图中总节点数
    cand_poi_train_nodes = list(
        set(idx_val) - set(atk_test_nodes) - set(clean_test_nodes))  # 候选的投毒训练节点：验证集中排除已选为攻击目标和干净测试的节点
    poison_nodes_num = int(N * args.vs_ratio)  # 计算投毒节点的数量，根据总节点数和投毒比例
    poi_train_nodes = rs.choice(cand_poi_train_nodes, poison_nodes_num, replace=False)  # 从候选节点中随机选择指定数量的投毒训练节点 (不重复选择)

    return atk_test_nodes, clean_test_nodes, poi_train_nodes  # 返回选择的攻击测试节点、干净测试节点和投毒训练节点


def normalize(mx):  # 定义函数 normalize，用于行规范化稀疏矩阵
    """Row-normalize sparse matrix"""  # 函数文档字符串：行规范化稀疏矩阵
    """
    函数功能说明: normalize (行规范化)
    对输入的稀疏矩阵进行行规范化处理。每一行的元素除以该行的元素之和。

    参数:
    mx (scipy.sparse.csr_matrix): 输入的稀疏矩阵。

    返回:
    scipy.sparse.csr_matrix: 行规范化后的稀疏矩阵。
    """
    rowsum = np.array(mx.sum(1))  # 计算每行的元素之和
    r_inv = np.power(rowsum, -1).flatten()  # 计算行和的倒数，并展平为一维数组
    r_inv[np.isinf(r_inv)] = 0.  # 将无穷大的值 (如果某行为0) 替换为0，避免除以0的错误
    r_mat_inv = sp.diags(r_inv)  # 创建一个对角矩阵，对角线元素为 r_inv
    mx = r_mat_inv.dot(mx)  # 左乘对角矩阵，实现行规范化
    return mx  # 返回规范化后的矩阵


def normalize_adj(adj):  # 定义函数 normalize_adj，用于对称规范化邻接矩阵
    """Symmetrically normalize adjacency matrix."""  # 函数文档字符串：对称规范化邻接矩阵
    """
    函数功能说明: normalize_adj (对称规范化邻接矩阵)
    对输入的邻接矩阵进行对称规范化，通常用于图卷积网络 (GCN) 中。
    规范化公式为: D^(-0.5) * A * D^(-0.5)，其中 A 是邻接矩阵，D 是度矩阵。

    参数:
    adj (scipy.sparse.csr_matrix or numpy.ndarray): 输入的邻接矩阵。

    返回:
    scipy.sparse.csr_matrix: 对称规范化后的邻接矩阵 (CSR 格式)。
    """
    adj = sp.coo_matrix(adj)  # 将输入邻接矩阵转换为 COO 稀疏格式，便于计算行和
    rowsum = np.array(adj.sum(1))  # 计算每个节点的度 (行和)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # 计算度的 -0.5 次方，并展平
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # 处理度为0的节点，避免无穷大
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # 创建对角矩阵 D^(-0.5)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(
        d_mat_inv_sqrt).tocsr()  # 执行 A * D^(-0.5)，然后转置，再乘以 D^(-0.5)，最后转换为 CSR 格式返回


def reconstruct_prune_unrelated_edge(args, poison_edge_index, poison_edge_weights, poison_x, ori_x, ori_edge_index,
                                     device, idx,
                                     large_graph=True):  # 定义函数 reconstruct_prune_unrelated_edge，用于基于重构误差修剪边
    """
    函数功能说明: reconstruct_prune_unrelated_edge (基于重构误差修剪不相关边)
    此函数结合了节点特征重构和边修剪。它首先训练一个自编码器 (MLPAE) 来学习节点特征的表示，
    然后使用重构误差来识别异常节点。最后，移除那些连接到异常节点的边。
    这是一种更复杂的边修剪策略，旨在识别并移除可能由攻击引入的、与正常数据分布不符的结构。

    参数:
    args (Namespace): 包含各种参数的对象，如 rec_epochs (重构模型训练轮数)。
    poison_edge_index (Tensor): 投毒后的边索引。
    poison_edge_weights (Tensor): 投毒后的边权重。
    poison_x (Tensor): 投毒后的节点特征。
    ori_x (Tensor): 原始节点特征 (可能用于确定哪些是新增的投毒节点)。
    ori_edge_index (Tensor): 原始边索引 (当前函数中未使用)。
    device (torch.device): 计算设备。
    idx (Tensor or list): 可能用于指示特定节点子集 (当前函数中未使用)。
    large_graph (bool): 是否为大图 (当前函数中未使用，但保留了参数位置)。

    返回:
    tuple: 包含过滤后的边索引 (filtered_poison_edge_index) 和边权重 (filtered_poison_edge_weights) 的元组。
    """
    poison_x = poison_x.to(device)  # 将投毒后的节点特征移动到指定设备
    AE = MLPAE(poison_x, poison_x[len(ori_x):], device,
               args.rec_epochs)  # 初始化多层感知机自编码器 (MLPAE)，可能使用新增的投毒节点特征 (poison_x[len(ori_x):]) 作为某种参考或目标进行训练
    AE.fit()  # 训练自编码器
    rec_score_ori = AE.inference(poison_x)  # 使用训练好的自编码器对所有投毒后的节点特征进行推理，得到重构误差分数
    threshold = np.percentile(rec_score_ori.detach().cpu().numpy(), 97)  # 计算重构误差分数的第97百分位数作为阈值，用于识别异常节点
    mask = rec_score_ori > threshold  # 创建一个布尔掩码，标记重构误差大于阈值的节点 (异常节点)
    keep_edges_mask = ~(mask[poison_edge_index[0]] | mask[
        poison_edge_index[1]])  # 保留边的掩码：如果一条边的两个端点都不是异常节点，则保留该边。 ~(A|B) 表示 非(A或B)，即 A和B都不为True
    filtered_poison_edge_index = poison_edge_index[:, keep_edges_mask]  # 根据掩码过滤边索引
    filtered_poison_edge_weights = poison_edge_weights[keep_edges_mask]  # 根据掩码过滤边权重
    return filtered_poison_edge_index, filtered_poison_edge_weights  # 返回过滤后的边索引和边权重


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
                    default=True, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--model', type=str, default='GCN', help='model',
                    choices=['GCN', 'GAT', 'GraphSage', 'GIN'])
parser.add_argument('--dataset', type=str, default='Cora',
                    help='Dataset',
                    choices=['Cora', 'Pubmed', 'Flickr', 'ogbn-arxiv', 'Computers', 'Photo'])
parser.add_argument('--train_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--thrd', type=float, default=0.5)
parser.add_argument('--target_class', type=int, default=0)
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train benign and backdoor model.')
parser.add_argument('--trojan_epochs', type=int, default=400, help='Number of epochs to train trigger generator.')
parser.add_argument('--inner', type=int, default=1, help='Number of inner')

parser.add_argument('--shadow_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--trojan_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--use_vs_number', action='store_true', default=True,
                    help="if use detailed number to decide Vs")
parser.add_argument('--vs_ratio', type=float, default=0,
                    help="ratio of poisoning nodes relative to the full graph")
parser.add_argument('--vs_number', type=int, default=40,
                    help="number of poisoning nodes relative to the full graph")
parser.add_argument('--defense_mode', type=str, default="prune",
                    choices=['prune', 'none', 'reconstruct'],
                    help="Mode of defense")
parser.add_argument('--prune_thr', type=float, default=0.8,
                    help="Threshold of prunning edges")
parser.add_argument('--target_loss_weight', type=float, default=1,
                    help="Weight of optimize outter trigger generator")
parser.add_argument('--homo_loss_weight', type=float, default=0.1,
                    help="Weight of optimize similarity loss")
parser.add_argument('--dis_weight', type=float, default=1,
                    help="Weight of cluster distance")
parser.add_argument('--test_model', type=str, default='GCN',
                    choices=['GCN', 'GAT', 'GraphSage', 'GNNGuard', 'RobustGCN'],
                    help='Model used to attack')
parser.add_argument('--device_id', type=int, default=0,
                    help="Threshold of prunning edges")

parser.add_argument('--alpha', type=float, default=0.02,
                    help="Ratio of feature dimensions to perturb")
parser.add_argument('--alpha_int', type=int, default=30,
                    help="Number of feature dimensions to perturb")
parser.add_argument('--outter_size', type=int, default=512,
                    help="Number of outter samples")

parser.add_argument('--rec_epochs', type=int, default=100,
                    help='Number of epochs to train benign and backdoor model.')
args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T

# transform = T.Compose([T.NormalizeFeatures()])
#
# if (args.dataset == 'Cora' or args.dataset == 'Citeseer' or args.dataset == 'Pubmed'):
#     dataset = Planetoid(root='./data/', \
#                         name=args.dataset, \
#                         transform=transform)
# elif (args.dataset == 'ogbn-arxiv'):
#     dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./data/')
#     split_idx = dataset.get_idx_split()




data = dataset[0].to(device)

# if (args.dataset == 'ogbn-arxiv'):
#     nNode = data.x.shape[0]
#     setattr(data, 'train_mask', torch.zeros(nNode, dtype=torch.bool).to(device))
#     data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
#     data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
#     data.y = data.y.squeeze(1)

from utils import get_split

data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(args, data, device)

from torch_geometric.utils import to_undirected
from utils import subgraph

data.edge_index = to_undirected(data.edge_index)
train_edge_index, _, edge_mask = subgraph(torch.bitwise_not(data.test_mask), data.edge_index, relabel_nodes=False)
mask_edge_index = data.edge_index[:, torch.bitwise_not(edge_mask)]

from models.backdoor_GCN_cos import Backdoor
from construct import model_construct

unlabeled_idx = (torch.bitwise_not(data.test_mask) & torch.bitwise_not(data.train_mask)).nonzero().flatten()
if (args.use_vs_number):
    size = args.vs_number
else:
    size = int((len(data.test_mask) - data.test_mask.sum()) * args.vs_ratio)
print("Attach Nodes:{}".format(size))
assert size > 0, 'The number of selected trigger nodes must be larger than 0!'
# here is randomly select poison nodes from unlabeled nodes
idx_attach = select(data, args, idx_train, idx_val, device).to(device)

print("idx_attach: {}".format(idx_attach))
unlabeled_idx = torch.tensor(list(set(unlabeled_idx.cpu().numpy()) - set(idx_attach.cpu().numpy()))).to(device)
print(unlabeled_idx)

model = Backdoor(args, device)
model.fit(data.x, train_edge_index, None, data.y, idx_train, idx_attach, unlabeled_idx)
poison_x, poison_edge_index, poison_edge_weights, poison_labels = model.get_poisoned()

if (args.defense_mode == 'prune'):
    poison_edge_index, poison_edge_weights = prune_unrelated_edge(args, poison_edge_index, poison_edge_weights,
                                                                  poison_x, device, large_graph=False)
    bkd_tn_nodes = torch.cat([idx_train, idx_attach]).to(device)
elif (args.defense_mode == 'reconstruct'):
    poison_edge_index, poison_edge_weights = reconstruct_prune_unrelated_edge(args, poison_edge_index,
                                                                              poison_edge_weights, poison_x, data.x,
                                                                              data.edge_index, device, idx_attach,
                                                                              large_graph=True)
    bkd_tn_nodes = torch.cat([idx_train, idx_attach]).to(device)
else:
    bkd_tn_nodes = torch.cat([idx_train, idx_attach]).to(device)
print("Precent of left attach nodes: {:.3f}" \
      .format(len(set(bkd_tn_nodes.tolist()) & set(idx_attach.tolist())) / len(idx_attach)))

import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli

mask = data.y[idx_attach] != args.target_class
mask = mask.to(device)
print('Number of poisoned target nodes', mask.sum())
## only attack those has groud truth labels != target_class ##
idx_attach = idx_attach[(data.y[idx_attach] != args.target_class).nonzero().flatten()]
bkd_tn_nodes = torch.cat([idx_train, idx_attach]).to(device)
known_nodes = torch.cat([idx_train, idx_attach]).to(device)
predictions = []
# edge weight for clean edge_index, may use later #
edge_weight = torch.ones([data.edge_index.shape[1]], device=device, dtype=torch.float)

#### train a backdoored model on poisoned graph ####
test_model = model_construct(args, args.test_model, data, device).to(device)
test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,
               train_iters=args.epochs, verbose=False)
test_model.eval()
clean_acc = test_model.test(poison_x, poison_edge_index, poison_edge_weights, poison_labels, idx_attach)
output_clean = test_model(poison_x, poison_edge_index, poison_edge_weights)
ori_predict = torch.exp(output_clean[known_nodes])
induct_edge_index = torch.cat([poison_edge_index, mask_edge_index], dim=1)
induct_edge_weights = torch.cat(
    [poison_edge_weights, torch.ones([mask_edge_index.shape[1]], dtype=torch.float, device=device)])
induct_x, induct_edge_index, induct_edge_weights = model.inject_trigger(idx_atk, poison_x, induct_edge_index,
                                                                        induct_edge_weights, device)
induct_x, induct_edge_index, induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(), induct_edge_weights.clone().detach()

output = test_model(induct_x, induct_edge_index, induct_edge_weights)

train_attach_rate = (output.argmax(dim=1)[idx_atk] == args.target_class).float().mean()
print("ASR: {:.4f}".format(train_attach_rate))
asr = train_attach_rate
ca = test_model.test(poison_x, induct_edge_index, induct_edge_weights, data.y, idx_clean_test)

print("CA: {:.4f}".format(ca))

###### formal test ########

test_model = model_construct(args, args.test_model, data, device, add_selfloop=False).to(device)
test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,
               train_iters=args.epochs, verbose=False)
test_model.eval()
clean_acc = test_model.test(poison_x, poison_edge_index, poison_edge_weights, poison_labels, idx_attach)
output_clean = test_model(poison_x, poison_edge_index, poison_edge_weights)
ori_predict = torch.exp(output_clean[known_nodes])
print("accuracy on poisoned target nodes: {:.4f}".format(clean_acc))

drop_ratio = 0.5


def sample_noise_all(edge_index, edge_weight, device):
    # Ensure inputs are on the correct device
    edge_index = edge_index.to(device)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=device)
    else:
        edge_weight = edge_weight.to(device)

    # Generate mask for edge dropping
    drop_mask = Bernoulli(1 - drop_ratio).sample(edge_weight.size()).bool()

    # Apply mask to edges
    noisy_edge_index = edge_index[:, drop_mask]
    noisy_edge_weight = edge_weight[drop_mask]

    # Get node degrees
    node_degrees = torch.zeros(edge_index.max() + 1, device=device)
    node_degrees.index_add_(0, noisy_edge_index[0], torch.ones(noisy_edge_index.size(1), device=device))
    # print('degree', node_degrees)

    # Restore edges for isolated nodes
    isolated_nodes = node_degrees == 0
    # print('isolated_nodes', isolated_nodes)
    if isolated_nodes.any():
        potential_restore_edges = isolated_nodes[edge_index[0]]
        # print('potential_restore_edges', potential_restore_edges)
        restore_edges = edge_index[:, potential_restore_edges]
        noisy_edge_index = torch.cat([noisy_edge_index, restore_edges], dim=1)
        restored_weights = torch.ones(restore_edges.size(1), device=device)
        noisy_edge_weight = torch.cat([noisy_edge_weight, restored_weights], dim=0)

    return noisy_edge_index, noisy_edge_weight


predictions = []
K = 20
for i in range(K):
    test_model.eval()
    noisy_poison_edge_index, noisy_poison_edge_weights = sample_noise_all(poison_edge_index, poison_edge_weights,
                                                                          device)
    output = test_model(poison_x, noisy_poison_edge_index, noisy_poison_edge_weights)
    train_attach_rate = (output.argmax(dim=1)[idx_attach] == args.target_class).float().mean()
    train_clean_rate = (output.argmax(dim=1)[idx_train] == data.y[idx_train]).float().mean()
    predictions.append(torch.exp(output[known_nodes]))

epsilon = 1e-8
deviations = []
for sub_pred in predictions:
    sub_pred += epsilon
    deviation = F.kl_div(sub_pred.log(), ori_predict, reduce=False)
    deviations.append(deviation)

summed_deviations = torch.zeros_like(deviations[0]).to(deviations[0].device)
for deviation in deviations:
    ##### summed deviations for each node #####
    summed_deviations += deviation

##### get the index for nodes with less robustness #####

##### args.vs_number is unknown #####
index_of_less_robust = torch.sort(torch.mean(summed_deviations, dim=-1), descending=True)[1]


def find_index(poison_labels, bkd_tn_nodes, index_of_less_robust, target_class):
    # Get the specific list to iterate through
    labels_list = poison_labels[bkd_tn_nodes[index_of_less_robust]]

    # Iterate through the list with index
    for i in range(len(labels_list) - 1):  # -1 to avoid index out of range
        if labels_list[i] != target_class and labels_list[i + 1] != target_class:
            return i - 1

    # Return None if the condition is not met in the loop
    return None


# Example usage:
# Assuming poison_labels, bkd_tn_nodes, index_of_less_robust, and target_class are defined
result_index = find_index(poison_labels, bkd_tn_nodes, index_of_less_robust, args.target_class)
print("Index found:", result_index)
# print(poison_labels[bkd_tn_nodes[index_of_less_robust][:result_index]])

indexs = poison_labels[bkd_tn_nodes[index_of_less_robust][:result_index - 1]]
count = 0
for i in indexs:
    if i == args.target_class:
        count += 1
## correct
correct = count

## fasle
false = len(indexs) - count

test_model = model_construct(args, args.test_model, data, device).to(device)
test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val, train_iters=400,
               verbose=False, finetune=True, attach=bkd_tn_nodes[index_of_less_robust][:result_index])

induct_edge_index = torch.cat([poison_edge_index, mask_edge_index], dim=1)
induct_edge_weights = torch.cat(
    [poison_edge_weights, torch.ones([mask_edge_index.shape[1]], dtype=torch.float, device=device)])
induct_x, induct_edge_index, induct_edge_weights = model.inject_trigger(idx_atk, poison_x, induct_edge_index,
                                                                        induct_edge_weights, device)
# induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(idx_attach,poison_x,induct_edge_index,induct_edge_weights,device)
induct_x, induct_edge_index, induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(), induct_edge_weights.clone().detach()

output = test_model(induct_x, induct_edge_index, induct_edge_weights)
print("****After Defense****")
train_attach_rate = (output.argmax(dim=1)[idx_atk] == args.target_class).float().mean()
print("ASR: {:.4f}".format(train_attach_rate))
asr = train_attach_rate
ca = test_model.test(poison_x, induct_edge_index, induct_edge_weights, data.y, idx_clean_test)
print("CA: {:.4f}".format(ca))