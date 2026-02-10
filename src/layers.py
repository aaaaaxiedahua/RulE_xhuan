
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, short_cut=False, batch_norm=False, activation="relu", dropout=0):
        super(MLP, self).__init__()

        self.dims = [input_dim] + hidden_dims
        self.short_cut = short_cut

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 2):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))
        else:
            self.batch_norms = None

    def forward(self, input):
        layer_input = input

        for i, layer in enumerate(self.layers):
            hidden = layer(layer_input)
            if i < len(self.layers) - 1:
                if self.batch_norms:
                    x = hidden.flatten(0, -2)
                    hidden = self.batch_norms[i](x).view_as(hidden)
                hidden = self.activation(hidden)
                if self.dropout:
                    hidden = self.dropout(hidden)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden

        return hidden

class FuncToNodeSum(nn.Module):
    def __init__(self, vector_dim):
        super(FuncToNodeSum, self).__init__()

        self.vector_dim = vector_dim
        self.layer_norm = nn.LayerNorm(self.vector_dim)
        self.add_model = MLP(self.vector_dim, [self.vector_dim])
        # for param in self.add_model.parameters():
        #     param.requires_grad = False
        
    
    def forward(self, A_fn, x_f, mlp_rule_feature):
        
        weight = torch.transpose(A_fn, 0, 1).unsqueeze(-1)
        message = x_f.unsqueeze(0)

        feature = torch.transpose((message * weight), 1, 2)
        weighted_features = torch.matmul(feature, mlp_rule_feature)
        weighted_features_norm = self.layer_norm(weighted_features)
        weighted_features_relu = torch.relu(weighted_features_norm)
        output = weighted_features_relu.mean(1)
        
        return output


    # def forward(self, A_fn, x_f):

    #     # batch_size = b_n.max().item() + 1


    #     weight = torch.transpose(A_fn, 0, 1).unsqueeze(-1)
    #     message = x_f.unsqueeze(0)


    #     features = (message * weight).sum(1)
    #     # features = (message * weight).mean(1)

    #     # features = (message * weight).max(1)[0]

    #     output = self.add_model(features)
    #     output = self.layer_norm(output)
    #     output = torch.relu(output)

    #     return output


class GatedRelationComposition(nn.Module):
    """
    门控关系组合：将规则体中的关系按顺序逐步融合，生成查询感知的规则表示。
    参考 RUN-GNN (EMNLP 2023) 的查询相关融合门控单元。

    先将 hidden_dim 投影到 gate_dim 以减少参数量，再在 gate_dim 空间做门控融合。
    """

    def __init__(self, hidden_dim, gate_dim, mlp_rule_dim):
        super(GatedRelationComposition, self).__init__()
        self.rel_proj = nn.Linear(hidden_dim, gate_dim)
        self.W_gate = nn.Linear(3 * gate_dim, gate_dim)
        self.W_hidden = nn.Linear(gate_dim, gate_dim)
        self.out_proj = nn.Linear(gate_dim, mlp_rule_dim)

    def forward(self, body_emb, body_mask, query_emb):
        """
        Args:
            body_emb: [N_rules, max_body_len, hidden_dim] 规则体关系 embedding（已含逆关系 flag）
            body_mask: [N_rules, max_body_len] bool，True 为有效位置
            query_emb: [hidden_dim] 查询关系 embedding（已含逆关系 flag）
        Returns:
            [N_rules, mlp_rule_dim]
        """
        body_proj = self.rel_proj(body_emb)                              # [N, max_len, gate_dim]
        query_proj = self.rel_proj(query_emb).unsqueeze(0).expand(
            body_emb.size(0), -1
        )                                                                 # [N, gate_dim]

        h = query_proj                                                    # [N, gate_dim]

        for t in range(body_emb.size(1)):
            r_t = body_proj[:, t, :]                                      # [N, gate_dim]
            m_t = body_mask[:, t].unsqueeze(-1).float()                   # [N, 1]

            gate_input = torch.cat([h, r_t, query_proj], dim=-1)          # [N, 3*gate_dim]
            gate = torch.sigmoid(self.W_gate(gate_input))                 # [N, gate_dim]
            h_new = gate * torch.tanh(self.W_hidden(r_t)) + (1 - gate) * h
            h = h_new * m_t + h * (1 - m_t)                              # padding 位置不更新

        return self.out_proj(h)                                           # [N, mlp_rule_dim]


class InterRuleAttention(nn.Module):
    """
    规则间多头自注意力：建模同一查询下多条规则的交互关系。
    参考 KnowFormer (ICML 2024) 的结构感知注意力思想。

    结构：MultiHeadAttention + Residual + LayerNorm
    """

    def __init__(self, mlp_rule_dim, attn_heads, use_structure_bias=True):
        super(InterRuleAttention, self).__init__()
        self.attn = nn.MultiheadAttention(mlp_rule_dim, attn_heads, batch_first=True)
        self.norm = nn.LayerNorm(mlp_rule_dim)
        self.use_structure_bias = use_structure_bias

    def forward(self, features, structure_bias=None):
        """
        Args:
            features: [N_rules, mlp_rule_dim]
            structure_bias: [N_rules, N_rules] 可选，加性注意力偏置
        Returns:
            [N_rules, mlp_rule_dim]
        """
        x = features.unsqueeze(0)                                         # [1, N, dim]

        attn_mask = None
        if self.use_structure_bias and structure_bias is not None:
            attn_mask = structure_bias                                     # [N, N] 2D additive mask

        attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask)
        output = self.norm(x + attn_output)                               # residual + LN

        return output.squeeze(0)                                          # [N, dim]
