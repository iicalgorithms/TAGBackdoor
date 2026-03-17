import torch
import torch.nn.functional as F
import logging
from torch_geometric.data import Data
from torch import nn, no_grad
import torch.optim as optim
from copy import deepcopy

from src.models import create_model
from src.defence import defense_prune, defense_reconstruct
from config import ExperimentConfig
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.metrics import accuracy_score


class GNNTrojan(nn.Module):
    """
    模块功能描述: GNNTrojan - 基于GAT的简化后门触发器生成器
    
    功能：使用GAT直接聚合子图信息为目标节点生成后门特征
    输入：目标节点特征 + 2跳邻居子图
    输出：目标节点的后门特征
    
    设计思路：
    1. 使用GAT直接聚合子图中的邻居信息
    2. 通过注意力机制自动学习邻居节点的重要性
    3. 生成具有隐蔽性的后门特征
    """
    
    def __init__(self, input_dim, output_dim, hidden_dim, num_heads=4):
        """初始化简化的GNN木马网络
        
        Args:
            input_dim (int): 输入节点特征维度
            output_dim (int): 输出后门特征维度
            num_heads (int): GAT注意力头数，默认
        """
        super(GNNTrojan, self).__init__()

        # 网络参数配置
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # GAT层：直接聚合子图信息
        self.gat_layer = GATConv(
            in_channels=input_dim,
            out_channels=hidden_dim // num_heads,  #128/4=32,由于拼接，还是128
            heads=num_heads,
            dropout=0.1,
            concat=True  # 拼接多头注意力输出
        )
        
        # 后门特征生成器：将GAT输出转换为后门特征
        self.backdoor_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # 确保输出有界，增强隐蔽性
        )
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, target_node_feature, subgraph_node_features, subgraph_edge_index, target_node_idx_in_subgraph):
        """
        前向传播：使用GAT为目标节点生成后门特征
        
        Args:
            target_node_feature (Tensor): 目标节点的原始特征 [1, input_dim]
            subgraph_node_features (Tensor): 2跳子图中所有节点特征 [num_subgraph_nodes, input_dim]
            subgraph_edge_index (Tensor): 子图边索引 [2, num_edges]
            target_node_idx_in_subgraph (int): 目标节点在子图中的索引
            
        Returns:
            Tensor: 目标节点的后门特征 [1, output_dim]
        """
        
        # 处理空子图或单节点情况
        if subgraph_edge_index.size(1) == 0 or subgraph_node_features.size(0) <= 1:
            # 如果没有邻居信息，直接基于目标节点特征生成后门特征
            # 创建一个简单的线性变换作为fallback
            fallback_features = F.relu(target_node_feature @ torch.randn(
                self.input_dim, self.hidden_dim, device=target_node_feature.device
            ) * 0.1)
            return self.backdoor_generator(fallback_features)

        # print(f"target_gat_output.shape:{subgraph_node_features.shape}")
        # 使用GAT聚合子图信息
        gat_output = self.gat_layer(subgraph_node_features, subgraph_edge_index)
        gat_output = F.relu(gat_output)
        
        # 提取目标节点的GAT输出
        target_gat_output = gat_output[target_node_idx_in_subgraph:target_node_idx_in_subgraph+1]  # [1, hidden_dim]


        
        # 生成后门特征
        backdoor_features = self.backdoor_generator(target_gat_output)
        
        return backdoor_features


class TextBackdoor:

    def __init__(self, config: ExperimentConfig, text_autoencoder):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.text_autoencoder = text_autoencoder

        self.trojan = None
        self.shadow_model = None
        self.best_trojan_weights = None
        self.device = config.device



    def _generate_backdoor_features_batch(self, target_node_indices, node_features, edge_index):
        """
        使用简化的GAT架构批量为多个目标节点生成后门特征
        
        Args:
            target_node_indices (Tensor): 目标节点索引列表
            node_features (Tensor): 所有节点特征
            edge_index (Tensor): 图边索引
            
        Returns:
            Tensor: 生成的后门特征 [num_targets, feature_dim]
        """
        backdoor_features = []
        num_nodes = node_features.size(0)
        
        # 调试信息：记录输入参数的基本信息
        self.logger.debug(f"生成后门特征 - 目标节点数: {len(target_node_indices)}, 总节点数: {num_nodes}")
        if len(target_node_indices) > 0:
            self.logger.debug(f"目标节点索引范围: [{target_node_indices.min().item()}, {target_node_indices.max().item()}]")
        self.logger.debug(f"边索引形状: {edge_index.shape}, 节点特征形状: {node_features.shape}")
        
        for target_idx in target_node_indices:
            target_idx_item = target_idx.item()
            
            # 提取2跳子图
            subgraph_nodes, subgraph_edge_index, mapping, _ = k_hop_subgraph(
                node_idx=target_idx_item,
                num_hops=2,
                edge_index=edge_index,
                relabel_nodes=True,  # 重新编号子图节点
                num_nodes=num_nodes
            )

            # 目标节点在重新编号后的子图中的索引总是0
            target_idx_in_subgraph = 0
            
            # 调试信息：记录子图提取结果
            self.logger.debug(f"目标节点 {target_idx_item}: 提取到 {len(subgraph_nodes) if subgraph_nodes is not None else 0} 个子图节点")

            
            if subgraph_nodes is None:
                # 子图提取失败，使用原始特征通过简化的fallback机制
                target_feature = node_features[target_idx:target_idx+1]  # [1, feature_dim]
                
                # 使用简化的fallback机制生成后门特征
                fallback_features = F.relu(target_feature @ torch.randn(
                    self.trojan.input_dim, self.trojan.hidden_dim, device=target_feature.device
                ) * 0.1)
                backdoor_feat = self.trojan.backdoor_generator(fallback_features)
            else:
                # 边界检查：确保所有子图节点索引都在有效范围内
                max_node_idx = subgraph_nodes.max().item()
                min_node_idx = subgraph_nodes.min().item()
                
                self.logger.debug(f"目标节点 {target_idx_item}: 子图节点数={len(subgraph_nodes)}, 索引范围=[{min_node_idx}, {max_node_idx}]")
                
                if max_node_idx >= num_nodes or min_node_idx < 0:
                    # 索引越界，使用fallback机制
                    self.logger.warning(f"子图节点索引越界 - 目标节点: {target_idx_item}, 子图索引范围: [{min_node_idx}, {max_node_idx}], 总节点数: {num_nodes}")
                    target_feature = node_features[target_idx:target_idx+1]  # [1, feature_dim]
                    fallback_features = F.relu(target_feature @ torch.randn(
                        self.trojan.input_dim, self.trojan.hidden_dim, device=target_feature.device
                    ) * 0.1)
                    backdoor_feat = self.trojan.backdoor_generator(fallback_features)
                else:
                    # 正常生成后门特征
                    target_feature = node_features[target_idx:target_idx+1]  # [1, feature_dim]
                    # 使用重新编号前的原始节点索引获取特征
                    subgraph_features = node_features[subgraph_nodes]  # [num_subgraph_nodes, feature_dim]

                    backdoor_feat = self.trojan(
                        target_feature, 
                        subgraph_features, 
                        subgraph_edge_index, 
                        target_idx_in_subgraph
                    )
            
            backdoor_features.append(backdoor_feat)
        
        # 确保有有效的后门特征返回
        if len(backdoor_features) == 0:
            # 如果没有生成任何特征，返回零张量
            return torch.zeros(0, node_features.size(1), device=node_features.device)
        
        return torch.cat(backdoor_features, dim=0)  # [num_targets, feature_dim]



    def _apply_defense_mode(self, poison_data, clean_data):

        self.logger.info(f" defense mode: {self.config.defense_mode}")

        # Apply defense method to backdoored data
        if self.config.defense_mode == 'prune':
            # Record statistics before defense
            original_num_edges = poison_data.train_edge_index.shape[1]
            original_num_nodes = poison_data.x.shape[0]
            self.logger.info(f" Before pruning defense - Nodes: {original_num_nodes}, Edges: {original_num_edges}")

            defensed_data = defense_prune(poison_data, self.config.prune_thr, self.device, large_graph=True)

            # Record statistics after defense
            defended_num_edges = defensed_data.edge_index.shape[1]
            edge_reduction_ratio = (original_num_edges - defended_num_edges) / original_num_edges * 100
            self.logger.info(
                f"After pruning defense - Nodes: {defensed_data.x.shape[0]}, Edges: {defended_num_edges}")
            self.logger.info(
                f"Pruning defense effect - Edge reduction: {edge_reduction_ratio:.2f}% ({original_num_edges} -> {defended_num_edges})")

        elif self.config.defense_mode == 'reconstruct':
            # Record statistics before defense
            original_num_edges = poison_data.train_edge_index.shape[1]
            original_num_nodes = poison_data.x.shape[0]
            self.logger.info(
                f"Before reconstruction defense - Nodes: {original_num_nodes}, Edges: {original_num_edges}")

            defensed_data = defense_reconstruct(poison_data, clean_data, self.config.rec_epochs, self.device)

            # Record statistics after defense
            defended_num_edges = defensed_data.edge_index.shape[1]
            edge_reduction_ratio = (original_num_edges - defended_num_edges) / original_num_edges * 100
            self.logger.info(
                f"After reconstruction defense - Nodes: {defensed_data.x.shape[0]}, Edges: {defended_num_edges}")
            self.logger.info(
                f"Reconstruction defense effect - Edge reduction: {edge_reduction_ratio:.2f}% ({original_num_edges} -> {defended_num_edges})")

        else:
            defensed_data = poison_data

        return defensed_data



    def _train_models(self, data, poison_candidates, poison_mask, sorted_dims):
        """
        训练影子模型和简化的GAT木马网络
        
        Args:
            data: 图数据
            poison_candidates: 中毒候选节点列表
            poison_mask: 中毒节点掩码
        """
        self.logger.info("开始训练触发器生成器...")

        data = data.to(self.device)
        poison_indices = torch.where(poison_mask)[0]

        feature_dim = data.x.shape[1]
        poison_dim_num = min(self.config.poison_dim_num, feature_dim)
        poison_dim_indices= sorted_dims[:poison_dim_num]

        print(f"==feature.dim:{feature_dim}")
        print(f"==poison_dim:{poison_dim_num}")


        # 初始化影子模型
        self.shadow_model = create_model(
            model_name=self.config.shadow_model,
            input_dim=feature_dim,
            hidden_dim=self.config.hidden_dim, #TODO.和target model保持一致
            output_dim=data.num_classes
        ).to(self.device)

        # 初始化简化的木马网络

        self.trojan = GNNTrojan(
            input_dim=feature_dim,
            output_dim=poison_dim_num,
            hidden_dim=self.config.hidden_dim,
        ).to(self.device)

        # 初始化优化器
        shadow_optimizer = optim.Adam(
            self.shadow_model.parameters(), 
            lr=self.config.learning_rate, 
            weight_decay=self.config.weight_decay
        )
        trojan_optimizer = optim.Adam(
            self.trojan.parameters(), 
            lr=self.config.learning_rate, 
            weight_decay=self.config.weight_decay
        )

        # 设置训练掩码和目标标签
        train_mask = data.train_mask
        train_poison_mask = train_mask.clone()  # train + poison

        poison_labels = data.y.clone()
        for candidate in poison_candidates:
            poison_labels[candidate['node_idx']] = candidate['target_label']
            train_poison_mask[candidate['node_idx']] = True
        original_labels = data.y.clone()

        loss_fn = nn.CrossEntropyLoss()
        best_loss = float('inf')
        
        # 主训练循环
        for epoch_trojan in range(self.config.trojan_epochs):

            # 阶段1：训练影子模型
            self.shadow_model.train()
            self.trojan.eval()
            shadow_optimizer.zero_grad()

            # 从影子模型获取嵌入
            with torch.no_grad():
                _, embed = self.shadow_model(data.x, data.train_edge_index, return_embeddings=True)

            print(f"embed.shape:{embed.shape}")


            # 为中毒节点生成后门特征
            poison_indices = torch.where(poison_mask)[0]
            poison_trojan_feat = self._generate_backdoor_features_batch(
                poison_indices, embed, data.train_edge_index
            )

            # 创建中毒特征
            poison_x = data.x.clone().detach()
            poison_x[poison_indices[:,None], poison_dim_indices] = poison_trojan_feat.detach()

            # 通过影子模型前向传播
            output = self.shadow_model(poison_x, data.train_edge_index)
            shadow_loss = loss_fn(output[train_poison_mask], poison_labels[train_poison_mask])
            shadow_loss.backward()
            shadow_optimizer.step()

            # 阶段2：训练木马网络
            self.shadow_model.eval()
            self.trojan.train()
            trojan_optimizer.zero_grad()

            # 随机选择未标记节点作为外部中毒节点
            unlabeled_mask = data.unlabeled_mask
            rest_unlabeled_mask = unlabeled_mask.clone()
            rest_unlabeled_mask = rest_unlabeled_mask & (~poison_mask.to(self.device))

            # 获取剩余未标记节点的索引
            rest_unlabeled_indices = torch.where(rest_unlabeled_mask)[0]
            
            # 从剩余未标记节点中随机选择outer_poison_num个节点
            if len(rest_unlabeled_indices) >= self.config.outer_poison_num:
                selected_indices = rest_unlabeled_indices[
                    torch.randperm(len(rest_unlabeled_indices))[:self.config.outer_poison_num]
                ]
            else:
                selected_indices = rest_unlabeled_indices
                self.logger.warning(
                    f"只有 {len(rest_unlabeled_indices)} 个未标记节点可用，"
                    f"使用全部而非 {self.config.outer_poison_num} 个"
                )
            
            # 创建外部中毒掩码
            outer_poison_mask = torch.zeros_like(unlabeled_mask, dtype=torch.bool)
            outer_poison_mask[selected_indices] = True

            # 从影子模型获取嵌入
            _, embeddings = self.shadow_model(poison_x, data.train_edge_index, return_embeddings=True)
            
            # 为外部中毒节点生成木马特征
            outer_poison_indices = torch.where(outer_poison_mask)[0]
            if len(outer_poison_indices) > 0:
                outer_trojan_features = self._generate_backdoor_features_batch(
                    outer_poison_indices, embeddings, data.train_edge_index
                )
                
                # 创建修改后的特征
                outer_poison_x = data.x.detach().clone()
                poison_x[outer_poison_indices[:, None], poison_dim_indices] = outer_trojan_features.detach()

                # 使用修改后的特征进行前向传播
                outer_output = self.shadow_model(outer_poison_x, data.train_edge_index)
                
                # 目标损失：中毒节点应被分类为目标类别
                target_loss = loss_fn(outer_output[train_poison_mask], poison_labels[train_poison_mask])

                # 相似性损失：修改后的特征应与原始特征相似
                similarity_loss = 1 - F.cosine_similarity(
                    data.x[outer_poison_mask], 
                    outer_poison_x[outer_poison_mask], 
                    dim=1
                ).mean()

                # 组合木马损失
                trojan_loss = (
                    target_loss + 
                    self.config.feature_similarity_weight * similarity_loss
                )
                
                trojan_loss.backward()
                trojan_optimizer.step()
                
                # 保存最佳模型
                if trojan_loss.item() < best_loss:
                    best_loss = trojan_loss.item()
                    self.best_trojan_weights = deepcopy(self.trojan.state_dict())
            else:
                trojan_loss = torch.tensor(0.0, device=self.device)

            
            # 日志记录
            if epoch_trojan % 50 == 0:
                with torch.no_grad():
                    self.shadow_model.eval()

                    pred = output.argmax(dim=1)
                    train_acc = (pred[train_mask] == original_labels[train_mask]).float().mean()
                    poison_acc = (pred[poison_mask] == poison_labels[poison_mask]).float().mean()
                    
                    self.logger.info(
                        f"木马训练轮次 {epoch_trojan}: "
                        f"影子损失: {shadow_loss.item():.4f}, "
                        f"木马损失: {trojan_loss.item():.4f}, "
                        f"训练准确率: {train_acc:.4f}, "
                        f"中毒准确率: {poison_acc:.4f}"
                    )
        
        # 加载最佳木马权重
        if self.best_trojan_weights is not None:
            self.trojan.load_state_dict(self.best_trojan_weights)
            self.logger.info("已加载最佳木马网络权重")

        self.logger.info("基于GAT的简化触发器学习完成")
        return poison_labels, poison_mask, train_poison_mask, poison_dim_indices



    def _create_poisoned_train_dataset(self, data, poison_labels, poison_mask, train_poison_mask, poison_dim_indices):
        """
        使用简化的GAT架构创建中毒训练数据集
        
        Args:
            data: 原始图数据
            poison_labels: 中毒节点标签
            poison_mask: 中毒节点掩码
            train_poison_mask: 训练+中毒节点掩码
            
        Returns:
            Data: 中毒后的图数据
        """
        self.logger.info("创建中毒训练数据集...")
        
        # 创建中毒数据集
        original_x = data.x.detach()
        final_poison_x = original_x.clone()

        poison_indices = torch.where(poison_mask)[0]

        # 设置模型为评估模式
        self.trojan.eval()
        self.shadow_model.eval()
        
        with torch.no_grad():
            # 获取影子模型的嵌入
            _, embed = self.shadow_model(final_poison_x, data.train_edge_index, return_embeddings=True)

            # 批量生成木马特征
            trojan_features = self._generate_backdoor_features_batch(
                poison_indices, embed, data.train_edge_index
            )
            
            # 应用木马特征
            final_poison_x[poison_indices[:,None], poison_dim_indices] = trojan_features.detach()

        # 使用文本自编码器生成新的文本和嵌入
        new_poison_texts = self.text_autoencoder.get_texts(final_poison_x[poison_indices])
        poison_embeddings = self.text_autoencoder.get_embeds(new_poison_texts)

        self.logger.info(f"生成的新中毒文本样例: {new_poison_texts[:3] if len(new_poison_texts) > 3 else new_poison_texts}")

        # 更新节点特征
        new_x = data.x.clone().detach()
        new_x[poison_mask] = poison_embeddings

        # 更新原始文本
        new_raw_texts = list(data.raw_texts)
        for idx, i in enumerate(poison_indices.tolist()):
            new_raw_texts[i] = new_poison_texts[idx]

        # 创建中毒数据对象
        poisoned_data = Data(
            x=new_x,
            y=poison_labels,
            raw_texts=new_raw_texts,
            train_poison_mask=train_poison_mask,  # train + poison
            train_mask=data.train_mask,
            poison_mask=poison_mask,
            val_mask=data.val_mask,
            test_mask=data.test_mask,
            edge_index=data.edge_index,
            train_edge_index=data.train_edge_index,
            test_edge_index=data.test_edge_index,
            num_classes=data.num_classes,
        ).to(self.device)

        # 应用防御模式
        poisoned_data = self._apply_defense_mode(poisoned_data, data)
        
        self.logger.info(f"中毒训练数据集创建完成，包含 {poison_mask.sum().item()} 个中毒节点")
        return poisoned_data


    def _train_target_model(self, data, model_name='GCN'):

        data = data.to(self.device)

        # Create target model
        model = create_model(model_name,
                             input_dim=data.x.shape[1],
                             hidden_dim=self.config.hidden_dim,
                             output_dim=data.num_classes).to(self.device)

        # Set up training parameters
        epochs = self.config.target_epochs
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.target_lr,
                                     weight_decay=self.config.target_weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        # train_mask=data.except_test_mask # 除了test的节点，包含train+poison+val
        train_mask = data.train_mask
        poison_mask = data.poison_mask
        train_poison_mask = data.train_poison_mask  # 包含了train+poison
        val_mask = data.val_mask

        # Early stopping parameters
        patience = self.config.patience  # 默认patience为50
        best_val_acc = 0.0
        best_model_state = None
        epochs_without_improvement = 0

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            # Forward pass
            out = model(data.x, data.train_edge_index)
            loss = criterion(out[train_poison_mask], data.y[train_poison_mask])
            train_loss = loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

            # Evaluation for early stopping
            model.eval()
            with torch.no_grad():
                eval_out = model(data.x, data.train_edge_index)

                # 计算验证集准确率（如果有验证集）或使用训练集准确率
                if val_mask is not None and val_mask.sum() > 0:
                    val_acc = (eval_out[val_mask].argmax(dim=1) == data.y[val_mask]).float().mean().item()
                else:
                    # 如果没有验证集，使用干净训练集准确率作为早停指标
                    val_acc = (eval_out[train_mask].argmax(dim=1) == data.y[train_mask]).float().mean().item()

                # 保存最佳模型权重
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict().copy()
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            if (epoch + 1) % 50 == 0:
                train_acc = (out[train_poison_mask].argmax(dim=1) == data.y[
                    train_poison_mask]).float().sum() / train_poison_mask.sum()
                ac_acc = (out[train_mask].argmax(dim=1) == data.y[train_mask]).float().sum() / train_mask.sum()
                asr_acc = (out[poison_mask].argmax(dim=1) == data.y[poison_mask]).float().sum() / poison_mask.sum()

                print(
                    f" Poisoned data on target model - Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc:{train_acc.item():.4f}, AC Acc:{ac_acc.item():.4f}, ASR Acc:{asr_acc.item():.4f}, Val Acc:{val_acc:.4f}, Best Val Acc:{best_val_acc:.4f}")

            # Early stopping check
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}. Best validation accuracy: {best_val_acc:.4f}")
                break

        # 加载最佳模型权重
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Loaded best model weights with validation accuracy: {best_val_acc:.4f}")

        return model
    def _get_poisoned_test_data(self, poison_train_data, target_model, poison_dim_indices):
        """
        使用简化的GAT架构生成中毒测试数据，通过注入后门触发器
        
        Args:
            poison_train_data: 中毒训练数据
            target_model: 目标模型
            poison_dim_indices: 中毒维度索引（可选，保持兼容性）
            
        Returns:
            Data: 包含中毒测试数据的图对象
        """
        self.logger.info("生成中毒测试数据...")

        test_mask = poison_train_data.test_mask
        test_node_indices = torch.where(test_mask)[0]

        # 使用目标模型预测测试节点
        with torch.no_grad():
            output = target_model(poison_train_data.x, poison_train_data.test_edge_index)
        probs = F.softmax(output, dim=1)
        predictions = output.argmax(dim=1)

        # 计算不确定性
        if self.config.uncertainty_method == "entropy":
            uncertainty = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        elif self.config.uncertainty_method == "variance":
            uncertainty = torch.var(probs, dim=1)
        else:
            raise ValueError(f"未知的不确定性方法: {self.config.uncertainty_method}")

        test_uncertainty = uncertainty[test_node_indices]

        # 根据不确定性排序并分割测试节点
        sorted_indices = torch.argsort(test_uncertainty, descending=True)
        test_node_indices = test_node_indices[sorted_indices]

        num_test_nodes = len(test_node_indices)
        split_point = num_test_nodes // 2

        test_poison_nodes = test_node_indices[:split_point]
        test_clean_nodes = test_node_indices[split_point:]

        # 创建测试掩码
        test_poison_mask = torch.zeros_like(test_mask, dtype=torch.bool)
        test_clean_mask = torch.zeros_like(test_mask, dtype=torch.bool)

        test_poison_mask[test_poison_nodes] = True
        test_clean_mask[test_clean_nodes] = True

        self.logger.info(f"测试节点总数: {len(test_node_indices)}")
        self.logger.info(f"中毒测试节点数: {len(test_poison_nodes)}")
        self.logger.info(f"干净测试节点数: {len(test_clean_nodes)}")

        # 设置中毒节点的目标标签
        poison_y = poison_train_data.y.clone()
        test_poison_node_indices = torch.where(test_poison_mask)[0]

        for node_idx in test_poison_node_indices:
            target_label = self.config.fixed_target_class
            poison_y[node_idx] = target_label

        # 为测试节点注入触发器
        with torch.no_grad():
            _, embed = target_model(poison_train_data.x, poison_train_data.test_edge_index, return_embeddings=True)

            print(f"embed.shape:{embed.shape}")
            
            # 批量生成测试节点的后门特征
            trojan_feat = self._generate_backdoor_features_batch(
                test_poison_node_indices, embed, poison_train_data.test_edge_index
            )

        # 创建中毒特征
        poison_x = poison_train_data.x.detach().clone()
        poison_x[test_poison_node_indices[:, None], poison_dim_indices] = trojan_feat.detach()

        # 创建中毒测试数据对象
        poisoned_data = Data(
            x=poison_x,
            y=poison_y,
            raw_texts=poison_train_data.raw_texts,
            train_mask=poison_train_data.train_mask,
            val_mask=poison_train_data.val_mask,
            test_mask=poison_train_data.test_mask,
            edge_index=poison_train_data.edge_index,
            train_edge_index=poison_train_data.train_edge_index,
            test_edge_index=poison_train_data.test_edge_index,
            test_poison_mask=test_poison_mask,
            test_clean_mask=test_clean_mask,
            num_classes=poison_train_data.num_classes,
        ).to(self.device)

        self.logger.info(f"生成的中毒数据包含 {poisoned_data.test_poison_mask.sum().item()} 个中毒节点")
        return poisoned_data

    def _evaluate_clean_accuracy(self, target_model, data):
        """评估模型在干净测试数据上的准确率
        
        Args:
            target_model: 要评估的目标模型
            data: 包含测试数据的图对象
            
        Returns:
            Dictionary: 包含干净数据准确率指标的字典
        """
        # 使用test_clean_mask识别干净测试节点
        test_mask = data.test_clean_mask

        self.logger.info(f"评估干净数据准确率，测试节点数: {torch.sum(test_mask).item()}...")

        # 获取预测结果
        data.to(self.device)
        with torch.no_grad():
            logits = target_model(data.x, data.test_edge_index)
            probs = F.softmax(logits, dim=1)

        predict_labels = logits.argmax(dim=1)[test_mask]
        true_labels = data.y[test_mask]

        # 计算指标
        accuracy = accuracy_score(true_labels.cpu().numpy(), predict_labels.cpu().numpy())

        # 每类准确率
        unique_classes = int(data.y.max().item()) + 1
        per_class_acc = {}

        for class_idx in range(unique_classes):
            class_mask = true_labels == class_idx
            if class_mask.sum() > 0:
                class_acc = (predict_labels[class_mask] == true_labels[class_mask]).float().mean()
                per_class_acc[class_idx] = class_acc.item()

        results = {
            'overall_accuracy': accuracy,
            'per_class_accuracy': per_class_acc,
            'num_test_samples': test_mask.sum().item()
        }

        self.logger.info(f"干净数据准确率评估结果: {results}")
        return results

    def _evaluate_attack_success_rate(self, target_model, data):
        """评估攻击成功率
        
        Args:
            target_model: 要评估的目标模型
            data: 包含中毒测试数据的图对象

        Returns:
            Dictionary: 包含攻击成功率指标的字典
        """
        test_mask = data.test_poison_mask
        x = data.x.detach()
        self.logger.info(f"评估攻击成功率，中毒测试节点数: {torch.sum(test_mask).item()}...")

        with torch.no_grad():
            logits = target_model(x, data.test_edge_index)
        predictions = logits.argmax(dim=1)
        targets = data.y
        predictions = predictions.to(targets.device)

        # 计算攻击成功率
        successful_attacks = (predictions[test_mask] == targets[test_mask]).float()
        attack_success_rate = successful_attacks.mean().item()

        # 每个目标类别的成功率
        unique_targets = torch.unique(targets[test_mask])
        per_target_asr = {}

        for target_class in unique_targets:
            target_mask = (targets[test_mask] == target_class)
            if target_mask.sum() > 0:
                target_asr = successful_attacks[target_mask].mean().item()
                per_target_asr[target_class.item()] = target_asr

        results = {
            'attack_success_rate': attack_success_rate,
            'per_target_asr': per_target_asr,
            'num_attack_samples': test_mask.sum().item(),
            'successful_attacks': successful_attacks.sum().item()
        }

        self.logger.info(f"攻击成功率评估结果: {results}")
        return results



    def run_trigger_generation_pipeline(self, data, poison_candidates, poison_mask, sorted_dims):
        """运行基于简化GAT架构的完整触发器生成流水线

        该方法协调整个后门攻击过程，包括：
        1. 训练影子模型和简化的GAT木马网络
        2. 使用GAT生成后门触发器
        3. 创建中毒数据
        """
        self.logger.info("开始基于GAT的简化触发器生成流水线...")

        poison_labels, poison_mask, train_poison_mask, poison_dim_indices= self._train_models(data, poison_candidates, poison_mask, sorted_dims)

        poisoned_data = self._create_poisoned_train_dataset(data, poison_labels, poison_mask, train_poison_mask, poison_dim_indices)

        evaluation_results={}
        for target_model_name in self.config.target_models:
            # 在中毒训练数据上训练目标模型
            target_model = self._train_target_model(poisoned_data, target_model_name)
            target_model.eval()

            # 生成用于攻击的中毒测试数据
            poisoned_test_data = self._get_poisoned_test_data(poisoned_data, target_model, poison_dim_indices)

            # 使用test_clean_mask评估干净数据准确率
            clean_accuracy_results = self._evaluate_clean_accuracy(target_model, poisoned_test_data)

            # 使用test_poison_mask评估攻击成功率
            attack_success_results = self._evaluate_attack_success_rate(target_model, poisoned_test_data)

            print(f"=========target model:{target_model}")
            self.logger.info(f"流水线完成 干净数据准确率: {clean_accuracy_results['overall_accuracy']:.4f}, 攻击成功率: {attack_success_results['attack_success_rate']:.4f}")
            # return poisoned_data, self.trojan, self.shadow_model, [i for i in range(data.x.shape[1])]
    
            # Compile comprehensive results
            evaluation_result = {
                'model_name': target_model_name,
                'clean_accuracy': clean_accuracy_results,
                'attack_success': attack_success_results,
                # 'text_similarity': all_similarity_score,
                # 'text_details': each_text_results,
                'experiment_config': {
                    'poison_num': self.config.poison_node_num,
                    'dataset_name': self.config.dataset_name,
                }
            }

            # Log summary
            self.logger.info(f"Evaluation summary for {target_model_name}:")
            self.logger.info(f"  Clean Accuracy: {clean_accuracy_results['overall_accuracy']:.4f}")
            self.logger.info(f"  Attack Success Rate: {attack_success_results['attack_success_rate']:.4f}")
            # self.logger.info(f"  Text Similarity: {all_similarity_score:.4f}")

            evaluation_results[target_model_name] = evaluation_result

        return evaluation_results
                






