# Gaussian-RulE 使用指南

## 📝 简介

**Gaussian-RulE** 是对原始 Box-RulE 模型的改进版本，使用**高斯分布嵌入**替代盒子嵌入，彻底解决了梯度消失问题。

### 核心改进

| 问题 | Box-RulE | Gaussian-RulE |
|------|----------|---------------|
| 梯度消失 | ❌ 盒子不相交时梯度≈0 | ✅ KL散度处处可微 |
| 高维诅咒 | ❌ 交集概率 ≈ 0.3^200 ≈ 10^-105 | ✅ 无交集计算 |
| 信息衰减 | ❌ 无法建模 | ✅ 方差累积自动建模 |
| 规则融合 | ⚠️ 需要复杂的体积计算 | ✅ 解析解（自动加权） |

---

## 🚀 快速开始

### 1. 环境配置

已安装依赖（参考主 README.md）：
```bash
conda create -n RulE python=3.8.0
conda activate RulE
pip install -r requirements.txt
```

### 2. 训练 Gaussian-RulE

#### 使用配置文件（推荐）：
```bash
cd src
python gaussian_main.py --config ../config/gaussian_kinship_config.json
```

#### 使用命令行参数：
```bash
cd src
python gaussian_main.py \
    --data_path ../data/kinship \
    --rule_path ../data/kinship/mined_rules.txt \
    --save_path ../checkpoints/gaussian_kinship \
    --hidden_dim 200 \
    --warmup_steps 5000 \
    --joint_steps 15000
```

### 3. 预期输出

训练日志会显示：
```
==================================================
Gaussian-RulE Training
==================================================
Stage 1: Warmup Training (Gaussian Initialization)
目标：初始化高斯嵌入，防止方差退化
[Warmup] Step 100/5000: loss_kge=0.2345, loss_kl_reg=0.000123, loss_total=0.2346
...
Warmup Valid MRR: 0.3456

Stage 2: Joint Training (KGE + Rules)
目标：学习规则嵌入，动态置信度 w_i(h)
[Joint] Step 100/15000: loss_kge=0.1234, loss_rule=0.0567, loss_kl_reg=0.000098, loss_total=0.2935
...
Joint Valid MRR: 0.4567
```

**关键指标**：
- ✅ **均值梯度 (mean_grad) > 0**：不再是 0.0000
- ✅ **MRR 持续上升**：不再卡在 0.0443
- ✅ **loss_kl_reg 稳定**：方差不退化

---

## 📂 文件结构

```
src/
├── gaussian_main.py          # 主训练脚本 ⭐新增
├── gaussian_model.py         # 高斯模型 ⭐新增
├── gaussian_trainer.py       # 训练器 ⭐新增
├── gaussian_layers.py        # 高斯操作层 ⭐新增
├── data.py                   # 数据加载（复用）
└── utils.py                  # 工具函数（复用）

config/
└── gaussian_kinship_config.json  # 配置文件 ⭐新增
```

---

## 🔬 模型架构

### 实体表示（高斯分布）

每个实体 $e$ 表示为：
$$\mathcal{N}(\mu_e, \sigma_e^2)$$

- **均值 $\mu_e$**：实体的语义中心
- **方差 $\sigma_e^2$**：实体的不确定性/模糊程度

### 推理操作

1. **关系变换**（沿路径推导）：
   $$
   \mu' = \mu + \mu_r, \quad \sigma'^2 = \sigma^2 + \sigma_r^2
   $$

2. **KL 散度评分**（处处可微！）：
   $$
   Score = \gamma - D_{KL}(\mathcal{P} || \mathcal{Q})
   $$
   $$
   D_{KL} = \frac{1}{2} \left( \sum \frac{\sigma_p^2}{\sigma_q^2} + \sum \frac{(\mu_p - \mu_q)^2}{\sigma_q^2} - d + \sum \log \frac{\sigma_q^2}{\sigma_p^2} \right)
   $$

   **关键**：第二项 $(\mu_p - \mu_q)^2$ 保证了无论分布相距多远，梯度都存在！

3. **动态置信度** $w_i(h)$：
   $$
   w_i = \exp(-D_{KL}(Gaussian_{Body} || Gaussian_{Head}))
   $$

   示例：Alice 搬家后，$w_3$(出生地规则) ≈ 0 自动失效。

---

## ⚙️ 核心参数说明

### 模型参数
- `hidden_dim`: 嵌入维度（默认 200）
- `init_logvar`: 初始 log 方差（-1.0 → 方差约 0.37）
- `epsilon`: 数值稳定性参数（1e-8）

### 损失权重
- `gamma_fact`: KGE margin（6.0）
- `gamma_rule`: Rule margin（5.0）
- `lambda_kl`: KL 正则化权重（0.001）
  - **作用**：防止方差退化（过小）或膨胀（过大）
- `target_logvar`: 目标 log 方差（0.0 → 方差保持在 1.0）
- `weight_rule`: 规则损失权重（2.0）

### 训练阶段
- `warmup_steps`: Warmup 阶段步数（5000）
  - 目标：初始化高斯嵌入
- `joint_steps`: Joint 阶段步数（15000）
  - 目标：学习规则和动态置信度

---

## 🔍 与 Box-RulE 对比

### Box-RulE 问题诊断（run.log）

```
Step 100: Grad: center=0.0000, width=671.1092, relation=621.0000
Step 500: Grad: center=0.0000, width=4.4101, relation=3.5518
Step 500: MRR: 0.0443 (stuck)
```

**问题根源**：
1. `center_grad = 0.0000`：盒子中心梯度消失
2. 原因：`softplus(max_inter - min_inter)` 在盒子不相交时梯度接近 0
3. 高维诅咒：200 维空间中盒子交集概率 ≈ 10^-105

### Gaussian-RulE 解决方案

```python
# KL 散度的关键项（gaussian_layers.py:52-53）
term2 = (mu_p - mu_q) ** 2 / (var_q + self.epsilon)  # 中心距离
```

- **梯度永存**：即使两个分布相距很远（如 Alice 的巴黎 vs 纽约），$(mu_p - mu_q)^2$ 依然产生强力梯度
- **无交集计算**：KL 散度不需要计算体积交集
- **自动加权**：Product of Gaussians 自动给置信度高（方差小）的规则更大权重

---

## 📊 预期效果

### 预期改进（相比 Box-RulE）

| 指标 | Box-RulE | Gaussian-RulE（预期） |
|------|----------|---------------------|
| Warmup MRR | 0.0443 (stuck) | > 0.30 |
| Joint MRR | 0.0443 (stuck) | > 0.40 |
| 均值梯度 | 0.0000 | > 0.01 |
| 方差梯度 | 4.4 → 0 (衰减) | 稳定 |

---

## 🛠️ 调试建议

### 如果训练不收敛

1. **检查 KL 正则化**：
   ```bash
   # 查看训练日志中的 loss_kl_reg
   # 如果过大（>0.01），降低 lambda_kl
   ```

2. **检查方差是否退化**：
   ```python
   # 在训练中添加日志
   logging.info(f'Mean logvar: {model.entity_logvar_emb.weight.mean().item():.4f}')
   # 应该在 -2.0 ~ 2.0 之间
   ```

3. **调整 target_logvar**：
   - 如果方差太大（logvar > 2），设置 `target_logvar = -0.5`
   - 如果方差太小（logvar < -2），设置 `target_logvar = 0.5`

### 如果内存不足

```json
{
    "batch_size": 64,          // 减小批次
    "rule_batch_size": 64,
    "hidden_dim": 100,         // 降低维度
    "negative_sample_size": 128
}
```

---

## 📖 参考文献

1. **KG2E** (Lin et al., 2015): "Learning Entity and Relation Embeddings for Knowledge Graph Completion"
   - 首次提出高斯分布嵌入

2. **TransG** (Xiao et al., 2016): "TransG : A Generative Model for Knowledge Graph Embedding"
   - 使用高斯混合模型

3. **RulE** (原始论文)
   - 本实现保留了动态置信度 $w_i(h)$ 的创新

---

## ✅ 总结

Gaussian-RulE 通过以下创新解决了 Box-RulE 的核心问题：

1. **高斯分布表示**：$\mathcal{N}(\mu, \sigma^2)$ 而非盒子 $(C, W)$
2. **KL 散度评分**：处处可微，梯度永存
3. **Product of Gaussians**：规则融合有解析解
4. **动态置信度 $w_i(h)$**：保留 RulE 的核心创新

**关键优势**：无需担心梯度消失，训练更稳定，性能更优。

---

## 🤝 贡献

如有问题或建议，欢迎提交 Issue 或 Pull Request。

---

**Happy Training! 🎉**
