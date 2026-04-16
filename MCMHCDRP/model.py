import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, global_mean_pool
from torch.nn.init import xavier_uniform_


# -------------------------- 超图嵌入分支 --------------------------
class HyperGraphEmbedding(nn.Module):
    def __init__(self, in_dim, hidden_dim, global_drug_emb=None):
        super().__init__()
        self.global_drug_emb = global_drug_emb if global_drug_emb is not None else torch.empty(0)
        self.conv1 = nn.Linear(in_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, in_dim)
        self.attn_mlp = nn.Sequential(nn.Linear(in_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 2))
        self.hyper_edge_extractor = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                                                  nn.Linear(hidden_dim, in_dim))
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(in_dim)

    def set_global_drug_emb(self, global_drug_emb):
        self.global_drug_emb = torch.nan_to_num(global_drug_emb, nan=0.0)

    def forward(self, drug_emb, hyper_adj_vec):
        hyper_adj_vec = torch.nan_to_num(hyper_adj_vec, nan=0.0)
        hyper_adj_vec = hyper_adj_vec - torch.max(hyper_adj_vec, dim=1, keepdim=True)[0]
        weights = F.softmax(hyper_adj_vec, dim=1)
        h_feat = self.hyper_edge_extractor(torch.matmul(weights, self.global_drug_emb))
        h = F.relu(self.conv1(weights @ (weights.transpose(0, 1) @ h_feat)))
        h = self.dropout(h);
        h = self.conv2(h)
        w = F.softmax(self.attn_mlp(torch.cat([drug_emb, h], dim=-1)), dim=1)
        out = self.norm(w[:, 0:1] * drug_emb + w[:, 1:2] * h + drug_emb)
        return torch.clamp(out, -10.0, 10.0)


# -------------------------- 药物图嵌入分支 --------------------------
class DrugGraphEmbeddingChebNet(nn.Module):
    def __init__(self, in_channels, out_dim, K=3):
        super().__init__()
        h = [256, 512, 512, 512]
        self.convs = nn.ModuleList([
            ChebConv(in_channels, h[0], K), ChebConv(h[0], h[1], K),
            ChebConv(h[1], h[2], K), ChebConv(h[2], h[3], K)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(dim) for dim in h])
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(h[3], out_dim)

    def forward(self, graph):
        x, edge_index = graph.x, graph.edge_index
        if edge_index.numel() == 0:
            x = torch.zeros((x.shape[0], 512), device=x.device)
        else:
            for conv, bn in zip(self.convs, self.bns):
                x = F.relu(bn(conv(x, edge_index)))
                x = self.dropout(x)
        batch = graph.batch if hasattr(graph, "batch") else torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        return self.fc(global_mean_pool(x, batch))


# -------------------------- 细胞 Perceiver 分支 --------------------------
class PerceiverCellEncoder(nn.Module):
    def __init__(self, exp_dim, mut_dim, meth_dim, path_dim, out_dim):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Linear(exp_dim, out_dim), nn.Linear(mut_dim, out_dim),
            nn.Linear(meth_dim, out_dim), nn.Linear(path_dim, out_dim)
        ])
        self.latents = nn.Parameter(torch.randn(1, 8, out_dim))
        self.cross_attn = nn.MultiheadAttention(out_dim, 8, batch_first=True)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, exp, mut, meth, path):
        inputs = [p(d.squeeze(1) if d.dim() == 3 else d) for p, d in zip(self.projs, [exp, mut, meth, path])]
        kv = torch.stack(inputs, dim=1)
        q = self.latents.repeat(kv.size(0), 1, 1)
        out, _ = self.cross_attn(q, kv, kv)
        return self.norm(out + q)


# -------------------------- 融合预测模块 --------------------------
class GatedBilinearFusion(nn.Module):
    def __init__(self, drug_dim, cell_dim, out_dim):
        super().__init__()
        self.drug_gate = nn.Linear(drug_dim, drug_dim)
        self.cell_gate = nn.Linear(cell_dim, cell_dim)
        self.fusion = nn.Linear(drug_dim + cell_dim, out_dim)

    def forward(self, d_vec, c_tokens):
        c_vec = c_tokens.mean(dim=1)
        d_g = torch.sigmoid(self.drug_gate(d_vec))
        c_g = torch.sigmoid(self.cell_gate(c_vec))
        return self.fusion(torch.cat([d_vec * d_g, c_vec * c_g], dim=1)), None


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, out_dim)
        )

    def forward(self, x): return self.net(x)


# -------------------------- 主模型 --------------------------
class MCMHCDRP(nn.Module):
    def __init__(self, cell_exp_dim, cell_mut_dim, cell_meth_dim, cell_path_dim, global_drug_emb, cfg):
        super().__init__()
        self.cfg = cfg
        self.drug_embedding = DrugGraphEmbeddingChebNet(6, cfg.drug.drug_out_dim)
        self.cell_embedding = PerceiverCellEncoder(cell_exp_dim, cell_mut_dim, cell_meth_dim, cell_path_dim,
                                                   cfg.cell.cell_out_dim)
        self.hyper_embedding = HyperGraphEmbedding(cfg.drug.drug_out_dim, cfg.hyper.hyper_hidden_dim, global_drug_emb)

        self.cl_proj_g = nn.Sequential(nn.Linear(cfg.drug.drug_out_dim, cfg.drug.drug_out_dim),
                                       nn.LayerNorm(cfg.drug.drug_out_dim))
        self.cl_proj_h = nn.Sequential(nn.Linear(cfg.drug.drug_out_dim, cfg.drug.drug_out_dim),
                                       nn.LayerNorm(cfg.drug.drug_out_dim))

        self.fusion = GatedBilinearFusion(cfg.drug.drug_out_dim, cfg.cell.cell_out_dim, cfg.mlp.mlp_in_dim)
        self.mlp = MLP(cfg.mlp.mlp_in_dim)

    def set_global_drug_emb(self, emb): self.hyper_embedding.set_global_drug_emb(emb)

    def forward(self, drug_graph, cell_data, hyper_adj_vec=None, mode='normal', hyper_v2=None):
        z_g1 = self.drug_embedding(drug_graph)
        if mode == 'contrastive':
            z_g2 = self.drug_embedding(drug_graph)
            z_h1 = self.hyper_embedding(z_g1, hyper_adj_vec)
            z_h2 = self.hyper_embedding(z_g1, hyper_v2)
            return F.normalize(self.cl_proj_g(z_g1), dim=1), F.normalize(self.cl_proj_g(z_g2), dim=1), \
                F.normalize(self.cl_proj_h(z_h1), dim=1), F.normalize(self.cl_proj_h(z_h2), dim=1)

        v_h = self.hyper_embedding(z_g1, hyper_adj_vec) if hyper_adj_vec is not None else None
        v_d = (z_g1 + v_h) if v_h is not None else z_g1
        v_c = self.cell_embedding(*cell_data)
        f, att = self.fusion(v_d, v_c)
        return self.mlp(f).squeeze(-1), att