import random
import re  # 必须导入：用于ID提取
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn.functional as F
import networkx as nx
from prettytable import PrettyTable
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from torch_geometric.data import Data
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def save_predictions(predictions, save_path):
    pd.DataFrame(predictions, columns=["cell", "drug", "true_ic50", "predicted_ic50"]).to_csv(save_path, index=False)

def augment_hyper_adj_vec(hyper_adj_vec: torch.Tensor, drop_prob=0.2, noise_std=0.05):
    x = hyper_adj_vec
    if drop_prob > 0:
        mask = (torch.rand_like(x) > drop_prob).float()
        x = x * mask
    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std

    # ✅ 防止这里产生 NaN / inf
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def info_nce_loss(z1, z2, temperature=0.5, hard_negative_ratio=0.1):
    batch_size = z1.size(0)
    eps = 1e-8
    z1 = torch.nan_to_num(z1, nan=0.0, posinf=0.0, neginf=0.0)
    z2 = torch.nan_to_num(z2, nan=0.0, posinf=0.0, neginf=0.0)
    z1 = z1 + eps
    z2 = z2 + eps
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z1 = torch.nan_to_num(z1, nan=0.0, posinf=0.0, neginf=0.0)
    z2 = torch.nan_to_num(z2, nan=0.0, posinf=0.0, neginf=0.0)
    similarity = torch.matmul(z1, z2.T) / max(eps, float(temperature))
    similarity = torch.nan_to_num(similarity, nan=0.0, posinf=0.0, neginf=0.0)
    pos_sim = torch.diag(similarity)
    eye_mask = ~torch.eye(batch_size, dtype=bool, device=z1.device)
    neg_sim = similarity[eye_mask].view(batch_size, batch_size - 1)
    k = int(batch_size * hard_negative_ratio)
    if k > 0 and k < neg_sim.size(1):
        neg_sim, _ = torch.topk(neg_sim, k=k, dim=1, largest=True)
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
    labels = torch.zeros(batch_size, dtype=torch.long, device=z1.device)
    return F.cross_entropy(logits, labels)

def eval_predict(y_label, y_pred, hyper_attention=None):
    mae = mean_absolute_error(y_label, y_pred)
    mse = mean_squared_error(y_label, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_label, y_pred)
    pearson, pearson_p = pearsonr(y_label, y_pred)
    spearman, spearman_p = spearmanr(y_label, y_pred)
    hyper_contrib = None
    if hyper_attention is not None and hyper_attention.numel() > 0:
        hyper_contrib = np.mean(hyper_attention.cpu().detach().numpy())
    return mse, rmse, mae, r2, pearson, pearson_p, spearman, spearman_p, hyper_contrib

# 下面保存/绘图/超图统计函数：你原来的可直接保留不动（省略可粘回原版）
# save_output / save_test_output / draw_loss_curve / print_table / print_hypergraph_stats / plot_hypergraph_structure / plot_hyper_adj_heatmap ...
def save_output(train_loss, test_loss, eval_result, savedir, prefix, best_model_eval=None,
                hyper_stats=None, contrastive_loss=None):
    os.makedirs(savedir, exist_ok=True)
    columns = ['Epoch', 'Train Loss', 'Test Loss', 'Contrastive Loss',
               'MSE', 'RMSE', 'MAE', 'R2', 'Pearson', 'Pearson p-value',
               'Spearman', 'Spearman p-value', 'Hyper Contribution',
               'Hyperedge Num', 'Avg Drugs per Hyperedge']

    data = []
    for epoch in range(len(train_loss)):
        cl_val = contrastive_loss[epoch] if contrastive_loss and epoch < len(contrastive_loss) else np.nan
        eval_row = eval_result[epoch] if epoch < len(eval_result) else [np.nan] * 13
        hyper_contrib = eval_row[8] if len(eval_row) > 8 else np.nan

        base_row = [epoch, train_loss[epoch], test_loss[epoch], cl_val,
                    eval_row[4] if len(eval_row) > 4 else np.nan,
                    eval_row[5] if len(eval_row) > 5 else np.nan,
                    eval_row[6] if len(eval_row) > 6 else np.nan,
                    eval_row[7] if len(eval_row) > 7 else np.nan,
                    eval_row[8] if len(eval_row) > 8 else np.nan,
                    eval_row[9] if len(eval_row) > 9 else np.nan,
                    eval_row[10] if len(eval_row) > 10 else np.nan,
                    eval_row[11] if len(eval_row) > 11 else np.nan,
                    hyper_contrib]

        if hyper_stats:
            base_row += [hyper_stats['hyperedge_num'], hyper_stats['avg_drugs_per_edge']]
        else:
            base_row += [np.nan, np.nan]
        data.append(base_row)

    pd.DataFrame(data, columns=columns).to_csv(
        os.path.join(savedir, f'{prefix}_results_with_hyper.csv'), index=False
    )

    if best_model_eval is not None:
        with open(os.path.join(savedir, 'best_model.txt'), 'w') as f:
            f.write("Best Model Evaluation:\n")
            f.write(str(best_model_eval) + '\n')
            if hyper_stats:
                f.write("\nHypergraph Stats:\n")
                for k, v in hyper_stats.items():
                    f.write(f"{k}: {v:.2f}\n")

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Train Loss (Main)', color='#1f77b4')
    plt.plot(test_loss, label='Test Loss (Main)', color='#ff7f0e')
    if contrastive_loss:
        plt.plot(contrastive_loss, label='Contrastive Loss (Hyper)', color='#2ca02c')
    plt.xlabel('Epoch'), plt.ylabel('Loss'), plt.title(f'{prefix} Loss (with Hyper)')
    plt.legend(), plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(savedir, f'{prefix}_loss_curve_with_hyper.jpg'), dpi=300, bbox_inches='tight')
    plt.close()


def save_test_output(test_loss, best_eval, savedir, prefix, hyper_stats=None):
    os.makedirs(savedir, exist_ok=True)
    np.save(os.path.join(savedir, f'{prefix}_test_loss.npy'), test_loss)
    with open(os.path.join(savedir, f'best_model_{prefix}.txt'), 'w') as f:
        f.write("Best Test Evaluation:\n")
        f.write(str(best_eval) + '\n')
        if hyper_stats:
            f.write("\nHypergraph Stats:\n")
            for k, v in hyper_stats.items():
                f.write(f"{k}: {v:.2f}\n")


def draw_loss_curve(train_loss, test_loss, savedir, prefix, contrastive_loss=None):
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Train Loss (Main)', color='#1f77b4')
    plt.plot(test_loss, label='Test Loss (Main)', color='#ff7f0e')
    if contrastive_loss:
        plt.plot(contrastive_loss, label='Contrastive Loss (Hyper)', color='#2ca02c')
    plt.xlabel('Epochs'), plt.ylabel('Loss'), plt.title(f'{prefix} Loss (with Hyper)')
    plt.legend(), plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(savedir, f'{prefix}_loss_curve_hyper.jpg'), dpi=300, bbox_inches='tight')
    plt.close()


def print_table(data, only_test=False, title=None, hyper_contribution=None):
    table = PrettyTable()
    if title:
        table.title = title
    if not only_test:
        table.field_names = ['epoch', 'train_loss', 'test_loss', 'mse', 'rmse', 'mae', 'r2',
                             'pearson', 'pcc-p', 'spearman', 'scc-p', 'hyper_contribution']
        data = list(data) + [f"{hyper_contribution:.4f}" if hyper_contribution is not None else "-"]
    else:
        table.title = 'Test Set Evaluation (with Hyper)'
        table.field_names = ['mse', 'rmse', 'mae', 'r2', 'pearson', 'pcc-p',
                             'spearman', 'scc-p', 'hyper_contribution']
        data = list(data) + [f"{hyper_contribution:.4f}" if hyper_contribution is not None else "-"]
    table.add_row(data)
    return table


# -------------------------- 超图专用工具（修复反向查找错误） --------------------------
def print_hypergraph_stats(hyperedges, hyper_adj, drug_dict, prefix=""):
    n_drug = len(drug_dict)
    n_hyperedge = len(hyperedges)
    drugs_per_edge = [len(e) for e in hyperedges] if hyperedges else []
    avg_drugs = np.mean(drugs_per_edge) if drugs_per_edge else 0.0
    isolated = sum(1 for i in range(n_drug) if hyper_adj[i].sum() == 0) if n_drug > 0 else 0
    non_zero_ratio = (hyper_adj > 0).sum() / (n_drug * n_drug) * 100 if n_drug > 0 else 0.0

    return {
        'drug_num': n_drug, 'hyperedge_num': n_hyperedge,
        'avg_drugs_per_edge': avg_drugs, 'isolated_drug_num': isolated,
        'adj_non_zero_ratio': non_zero_ratio
    }


def plot_hypergraph_structure(hyperedges, drug_dict, savedir, prefix="", top_k_edges=20):
    # 修复：drug_dict反向查找（key=int_id, value=连续索引）
    drug_idx_to_int_id = {v: k for k, v in drug_dict.items()}
    sorted_edges = sorted(hyperedges, key=lambda x: len(x), reverse=True)[:top_k_edges] if hyperedges else []
    G = nx.Graph()

    for i, edge in enumerate(sorted_edges):
        he_node = f"HE_{i}"
        G.add_node(he_node, type='hyperedge', color='#ff7f0e')
        for drug_idx in edge:
            # 安全查找药物int_id
            drug_int_id = drug_idx_to_int_id.get(drug_idx, f"drug_{drug_idx}")
            G.add_node(drug_int_id, type='drug', color='#1f77b4')
            G.add_edge(drug_int_id, he_node)

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=2020, k=3)  # 增大k，避免节点重叠

    drug_nodes = [n for n, a in G.nodes(data=True) if a['type'] == 'drug']
    he_nodes = [n for n, a in G.nodes(data=True) if a['type'] == 'hyperedge']

    nx.draw_networkx_nodes(
        G, pos, nodelist=drug_nodes, node_size=300, node_color='#1f77b4', alpha=0.8, label='Drug'
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=he_nodes, node_size=500, node_color='#ff7f0e', alpha=0.8, label='Hyperedge (Virtual)'
    )
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=1)
    nx.draw_networkx_labels(G, pos, labels={n: n for n in drug_nodes}, font_size=8)

    plt.legend(fontsize=12, loc='upper right')
    plt.title(f'{prefix} Hypergraph Structure (Top-{top_k_edges} Hyperedges)', fontsize=14)
    plt.axis('off')

    save_path = os.path.join(savedir, f'{prefix}_hypergraph_structure.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_hyper_adj_heatmap(hyper_adj, drug_dict, savedir, prefix="", top_n_drugs=50):
    drug_idx_to_int_id = {v: k for k, v in drug_dict.items()}
    top_n_drugs = min(top_n_drugs, hyper_adj.shape[0])
    top_idx = np.argsort(hyper_adj.sum(axis=1))[::-1][:top_n_drugs]
    hyper_adj_top = hyper_adj[top_idx][:, top_idx]
    top_drug_ids = [drug_idx_to_int_id.get(idx, f"drug_{idx}") for idx in top_idx]

    plt.figure(figsize=(12, 10))
    cmap = mcolors.LinearSegmentedColormap.from_list("custom", ["white", "#1f77b4", "#ff7f0e"])
    im = plt.imshow(hyper_adj_top, cmap=cmap)
    plt.xticks(range(len(top_drug_ids)), top_drug_ids, rotation=90, fontsize=8)
    plt.yticks(range(len(top_drug_ids)), top_drug_ids, fontsize=8)
    plt.colorbar(im, label='Association Strength')
    plt.title(f'{prefix} Hypergraph Adjacency (Top-{top_n_drugs} Drugs)')
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, f'{prefix}_hyper_adj_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
