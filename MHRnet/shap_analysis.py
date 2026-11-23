# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import torch
import pandas as pd
import seaborn as sns

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import shap
import inspect

from config import get_cfg_defaults
from data_load import dataload
from data_process import data_process
from model import MHRnet

# ===================== 配置与路径 =====================
BEST_MODEL_PATH = r"./output_dir/best_model.pt"
OUT_DIR = r"./shap_final_output"

# ===================== 控制参数 =====================
SEED = 2020
TOP_VAR_GENES = 2000  # 初筛基因数
TOP_SHOW = 10  # 提取前10个基因
NSAMPLES_A = 80  # 提高采样精度（解决SSX2散乱问题）
N_SUMMARY_TOTAL = 300  # 总绘图点数
CMAP_NAME = "plasma"  # 指定颜色
CLIP_VAL = 3.0  # 特征截断值（解决SSX2问题的关键）


# ===================== 工具函数 =====================
def set_seed_all(seed=2020):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def to_np_1d(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().view(-1).numpy().astype(np.float32)


def save_tiff400(fig, save_path):
    fig.savefig(save_path, dpi=400, format="tiff", bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


@torch.no_grad()
def recompute_global_drug_emb(model, aligned_drug_graphs, device):
    model.eval()
    embs = []
    for g in aligned_drug_graphs:
        g = g.to(device)
        emb = model.drug_embedding(g)
        embs.append(emb.squeeze())
    return torch.stack(embs, dim=0)


# ===================== 修正后的 Wrapper (包含数据清洗逻辑) =====================
class WrapperExpSubsetFixedContext:
    def __init__(self, model, device, v_d_fixed, exp_baseline, exp_sub_idx,
                 mut_fixed, meth_fixed, path_fixed, label_mean, label_std):
        self.model = model
        self.device = device
        self.v_d_fixed = v_d_fixed
        self.exp0_full = exp_baseline
        self.exp_sub_idx = np.array(exp_sub_idx, dtype=np.int64)
        self.mut_fixed = mut_fixed
        self.meth_fixed = meth_fixed
        self.path_fixed = path_fixed
        self.label_mean = float(label_mean)
        self.label_std = float(label_std)

    def predict(self, X_sub):
        X_sub = np.asarray(X_sub, dtype=np.float32)
        # ✅ 核心改进：对 SSX2 等离群特征进行截断，平滑 SHAP 贡献
        X_sub = np.clip(X_sub, -CLIP_VAL, CLIP_VAL)

        if X_sub.ndim == 1: X_sub = X_sub.reshape(1, -1)
        n = X_sub.shape[0]

        # 恢复完整 exp
        base = self.exp0_full.detach().cpu().numpy().astype(np.float32)
        exp_np = np.repeat(base, n, axis=0)
        exp_np[:, self.exp_sub_idx] = X_sub
        exp_t = torch.from_numpy(exp_np).to(self.device)

        # 固定其他
        mut_t = self.mut_fixed.expand(n, -1)
        meth_t = self.meth_fixed.expand(n, -1)
        path_t = self.path_fixed.expand(n, -1)

        with torch.no_grad():
            v_c = self.model.cell_embedding(exp_t, mut_t, meth_t, path_t)
            v_d = self.v_d_fixed.expand(n, -1)
            f, _ = self.model.fusion(v_d, v_c)
            pred_norm = self.model.mlp(f).squeeze(-1)

        return (pred_norm * self.label_std + self.label_mean).detach().cpu().numpy()


# ===================== 主程序 =====================
def main():
    set_seed_all(SEED)
    ensure_dir(OUT_DIR)
    cfg = get_cfg_defaults()
    device = torch.device(f"cuda:{cfg.model.cuda_id}" if torch.cuda.is_available() else "cpu")

    # 1. 加载数据 (适配你 13 个返回值的 data_process)
    data = dataload(**cfg)
    processed = data_process(data[0], data[1], data[2], data[3], data[4],
                             data[5], data[7], data[8], data[9], cfg)

    (train_set, test_set, val_set, full_data_set, Hd_tensor, _,
     _, _, _, aligned_drug_graphs, valid_drug_ids_A, l_mean, l_std) = processed

    # 2. 模型初始化
    model = MHRnet(data[1].shape[1], data[0].shape[1], data[2].shape[1], data[3].shape[1],
                     torch.zeros((len(valid_drug_ids_A), cfg.drug.drug_out_dim), device=device), cfg).to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()

    global_emb = recompute_global_drug_emb(model, aligned_drug_graphs, device)
    model.set_global_drug_emb(global_emb)

    full_ds = full_data_set.dataset
    gene_names_all = [str(c) for c in pd.read_csv(cfg.path.expression, index_col=0).columns]

    # 3. 筛选 Top 10 (使用方差初筛 + 快速 SHAP)
    expr_all = full_ds.cell_expression.cpu().numpy()
    cand_idx = np.argsort(-np.var(expr_all, axis=0))[:TOP_VAR_GENES]

    # 获取用于解释的样本和背景
    all_pos = list(range(len(full_ds.position)))
    random.shuffle(all_pos)
    ex_ids = all_pos[:15]  # 用少量样本定 Top 10，节省时间
    bg_ids = all_pos[15:20]

    # 固定上下文 (取均值)
    exp0_t = torch.from_numpy(np.mean([to_np_1d(full_ds[j][1]) for j in bg_ids], axis=0)).to(device).view(1, -1)
    mut0_t = torch.from_numpy(to_np_1d(full_ds[bg_ids[0]][2])).to(device).view(1, -1)
    meth0_t = torch.from_numpy(to_np_1d(full_ds[bg_ids[0]][3])).to(device).view(1, -1)
    path0_t = torch.from_numpy(to_np_1d(full_ds[bg_ids[0]][4])).to(device).view(1, -1)

    # 默认选第一个药物做展示背景
    d_idx = int(full_ds.pair[full_ds.position[ex_ids[0]]][1])
    with torch.no_grad():
        vg = global_emb[d_idx].view(1, -1).to(device)
        hv = Hd_tensor[d_idx].view(1, -1).to(device)
        vh = model.hyper_embedding(vg, hv)
        vd = (vg + vh)

    print("Stage 1: Ranking genes...")
    X_bg = np.stack([to_np_1d(full_ds[j][1]) for j in bg_ids])[:, cand_idx]
    X_ex = np.stack([to_np_1d(full_ds[j][1]) for j in ex_ids])[:, cand_idx]
    wrapper_rank = WrapperExpSubsetFixedContext(model, device, vd, exp0_t, cand_idx, mut0_t, meth0_t, path0_t, l_mean,
                                                l_std)
    explainer_rank = shap.KernelExplainer(wrapper_rank.predict, X_bg)
    sv_rank = explainer_rank.shap_values(X_ex, nsamples=40)

    top10_local = np.argsort(-np.mean(np.abs(sv_rank), axis=0))[:TOP_SHOW]
    top10_gene_idx = cand_idx[top10_local]
    top10_names = [gene_names_all[i] for i in top10_gene_idx]

    # 4. 阶段 2：高质量 SHAP 计算
    print(f"Stage 2: High-res SHAP for Top 10: {top10_names}")
    summary_ids = all_pos[:min(N_SUMMARY_TOTAL, len(all_pos))]
    X_ex_final = np.stack([to_np_1d(full_ds[j][1]) for j in summary_ids])[:, top10_gene_idx]
    # 对特征进行平滑截断，用于绘图显示（对应 SSX2 问题）
    X_ex_final_clipped = np.clip(X_ex_final, -CLIP_VAL, CLIP_VAL)

    X_bg_final = np.mean(np.stack([to_np_1d(full_ds[j][1]) for j in bg_ids]), axis=0)[top10_gene_idx].reshape(1, -1)

    wrapper_final = WrapperExpSubsetFixedContext(model, device, vd, exp0_t, top10_gene_idx, mut0_t, meth0_t, path0_t,
                                                 l_mean, l_std)
    explainer_final = shap.KernelExplainer(wrapper_final.predict, X_bg_final)
    sv_final = explainer_final.shap_values(X_ex_final_clipped, nsamples=NSAMPLES_A)

    # 5. 绘图 1: Beeswarm Plot (第一张形状 + 第二张颜色)
    plt.style.use("ggplot")
    fig1 = plt.figure(figsize=(9, 7))
    exp_obj = shap.Explanation(values=sv_final, data=X_ex_final_clipped, feature_names=top10_names)
    shap.plots.beeswarm(exp_obj, max_display=TOP_SHOW, color=plt.get_cmap(CMAP_NAME), show=False)
    plt.title("Top 10 Feature Contributions (Smoothed)", fontsize=14)
    plt.xlabel("SHAP Value (Impact on IC50)")
    save_tiff400(fig1, os.path.join(OUT_DIR, "Fig1_Beeswarm_Plasma.tiff"))

    # 6. 绘图 2: Interaction Heatmap (十个基因之间的关联分析)
    print("Stage 3: Calculating Interaction Heatmap...")
    # 计算 SHAP 值的 Pearson 相关性，代表特征间贡献的相关性（协同作用）
    corr_matrix = np.corrcoef(sv_final.T)

    fig2, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                xticklabels=top10_names, yticklabels=top10_names, square=True, ax=ax)
    plt.title("Gene-Gene SHAP Interaction Landscape", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    save_tiff400(fig2, os.path.join(OUT_DIR, "Fig2_Interaction_Heatmap.tiff"))

    print(f"Success! Analysis complete. Check files in {OUT_DIR}")


if __name__ == "__main__":
    main()