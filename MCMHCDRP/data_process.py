import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch_geometric.data import Batch
import re
from rdkit import Chem
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
from typing import Tuple, Optional
from collections import defaultdict

SKLEARN_AVAILABLE = True

def extract_int_id(id_str):
    if isinstance(id_str, int):
        return id_str
    id_str = str(id_str).strip()
    m = re.search(r'(\d+)', id_str)
    return int(m.group()) if m else hash(id_str) % 1000000

def collate_fn(batch):
    drug_graphs = [item[0] for item in batch]
    other_features = [item[1:] for item in batch]
    batched_graph = Batch.from_data_list(drug_graphs)

    batched_others = []
    for i in range(len(other_features[0])):
        feat_list = [item[i] for item in other_features]
        if isinstance(feat_list[0], torch.Tensor):
            batched_others.append(torch.stack(feat_list))
        else:
            batched_others.append(torch.tensor(feat_list, dtype=torch.float))
    return (batched_graph,) + tuple(batched_others)

# ---- drug fp / Sd ----
def smiles_to_ecfp4(smiles: str, radius: int = 2, n_bits: int = 1024) -> Optional[np.ndarray]:
    if pd.isna(smiles) or str(smiles).strip() == "":
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits, includeChirality=True)
        bv = gen.GetFingerprint(mol)
    except Exception:
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits, useChirality=True)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr

def build_drug_fingerprint_matrix(drug_smiles_df: pd.DataFrame, smiles_col="SMILES", drug_id_col="drug_id",
                                  radius=2, n_bits=1024) -> Tuple[np.ndarray, list]:
    fingerprint_matrix = []
    valid_drug_ids = []
    for _, row in tqdm(drug_smiles_df.iterrows(), total=len(drug_smiles_df), disable=True):
        did = extract_int_id(row[drug_id_col])
        fp = smiles_to_ecfp4(row[smiles_col], radius=radius, n_bits=n_bits)
        if fp is not None:
            fingerprint_matrix.append(fp)
            valid_drug_ids.append(did)
    fingerprint_matrix = np.vstack(fingerprint_matrix)
    return fingerprint_matrix, valid_drug_ids

def compute_drug_similarity_matrix(D, valid_drug_ids, K=10):
    n = D.shape[0]
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            inter = np.logical_and(D[i], D[j]).sum()
            union = np.logical_or(D[i], D[j]).sum()
            s = inter / union if union != 0 else 0
            sim[i, j] = s
            sim[j, i] = s

    adj = np.zeros((n, n))
    for i in range(n):
        s = sim[i].copy()
        s[i] = -1
        topk = np.argsort(s)[-K:]
        for j in topk:
            if s[j] > 0:
                adj[i, j] = s[j]
    return adj, valid_drug_ids

# ---- Sc ----
def normalize_gene_expression(X):
    X = np.asarray(X)
    if np.any(np.isnan(X)):
        X = SimpleImputer(strategy='mean').fit_transform(X)
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit_transform(X)

def generate_cell_similarity_matrix_Sc(gene_exp_df, K=10):
    cell_int_ids = gene_exp_df.index.tolist()
    norm_exp = normalize_gene_expression(gene_exp_df.values)
    sim = cosine_similarity(norm_exp) if SKLEARN_AVAILABLE else (norm_exp @ norm_exp.T)
    np.fill_diagonal(sim, 0)
    adj = np.zeros_like(sim)
    for i in range(sim.shape[0]):
        topk = np.argsort(sim[i])[-K:]
        for j in topk:
            if sim[i, j] > 0:
                adj[i, j] = sim[i, j]
                adj[j, i] = sim[i, j]
    return adj, cell_int_ids

# ---- A align ----
def get_aligned_response_matrix(gdsc_path: str, sc_cell_lines: list, sd_drugs: list,
                                threshold_ic50: float = 1.0):
    """
    从 GDSC2_IC50.csv 读取响应矩阵，并与 Sc / Sd 对齐。

    - 首列为细胞系名 -> 作为 index
    - 首行药物 id -> 作为列名
    - 内部有 NaN -> 先删“全 NaN 行/列”，再用列均值填充剩余 NaN
    """
    import pandas as pd
    from data_process import extract_int_id   # 或把 extract_int_id 放到公共 utils 里

    # 1. 正确读取：首列作为 index
    df = pd.read_csv(gdsc_path, index_col=0)

    # 2. index / columns 转成 int id（和其它模块统一）
    df.index = [extract_int_id(i) for i in df.index]
    df.columns = [extract_int_id(c) for c in df.columns]

    # 3. 只保留在 Sc / Sd 里出现的细胞和药物（交集）
    sc_cell_set = set(sc_cell_lines)
    sd_drug_set = set(sd_drugs)

    common_cells = [cid for cid in df.index if cid in sc_cell_set]
    common_drugs = [did for did in df.columns if did in sd_drug_set]

    df = df.loc[common_cells, common_drugs]

    # 4. 删掉“整行/整列全是 NaN”的细胞 / 药物（你这个文件其实没有，但保险起见）
    df = df.dropna(axis=0, how="all")   # 删全 NaN 的细胞系
    df = df.dropna(axis=1, how="all")   # 删全 NaN 的药物

    # 重新确定有效细胞/药物 id
    valid_cell_ids = df.index.tolist()
    valid_drug_ids = df.columns.tolist()

    # 5. 用“每列均值”填充 NaN，如果某列均值也是 NaN，再用 0 兜底
    col_means = df.mean(axis=0)
    df = df.fillna(col_means)
    df = df.fillna(0.0)

    # 6. 得到 IC50 连续矩阵 + 二分类响应矩阵
    A_ic50 = df.values.astype("float32")
    A_response = (A_ic50 <= threshold_ic50).astype("int32")

    return A_ic50, A_response, valid_cell_ids, valid_drug_ids


# ---- Hd build (keep your logic) ----
def extract_drug_hyperedges(Hd, Sd, valid_drug_ids, min_size=2, similarity_threshold=None):
    n = Hd.shape[0]
    hyperedges, hyperedge_set = [], set()
    hyperedge_map = defaultdict(list)

    for i in range(n):
        did_i = valid_drug_ids[i]
        total = Hd[i] + Sd[i]
        k = min(5, n - 1)
        idx = np.argsort(total)[-k-1:-1] if k > 0 else []
        if len(idx) == 0:
            continue
        nodes = sorted(list(set([did_i] + [valid_drug_ids[j] for j in idx])))
        if len(nodes) >= min_size:
            t = tuple(nodes)
            if t not in hyperedge_set:
                hyperedge_set.add(t)
                he_idx = len(hyperedges)
                hyperedges.append(nodes)
                for node in nodes:
                    hyperedge_map[node].append(he_idx)
    return hyperedges, hyperedge_map

def build_drug_hypergraph(Sd, Sc, D, A, valid_drug_ids):
    Hd1 = Sd @ Sd * Sd
    A_T_A = A.T @ A
    Hd2 = A_T_A * Sd
    Sd_A_T = A.T @ Sc @ A
    Sd_A_T_d = Sd_A_T * Sd
    Hd3 = Sd_A_T_d + Sd_A_T_d.T
    Sd_AT = Sd @ A.T
    Hd4 = Sd_AT @ Sd_AT.T

    Hd =Hd1 + Hd2 + Hd3 + Hd4
    if Hd.max() != 0:
        Hd = Hd / Hd.max()
    hyperedges, hyperedge_map = extract_drug_hyperedges(Hd, Sd, valid_drug_ids)
    return Hd, hyperedges, hyperedge_map

class my_dataloader(Dataset):
    def __init__(self, drug_graphs_aligned, cell_expression, cell_mutation, cell_methylation,
                 cell_pathway, pair, position, drug_hypergraph_adj=None,
                 label_mean=0.0, label_std=1.0, normalize_label=True):
        self.drug_graphs = drug_graphs_aligned
        self.cell_expression = cell_expression
        self.cell_mutation = cell_mutation
        self.cell_methylation = cell_methylation
        self.cell_pathway = cell_pathway
        self.pair = pair
        self.position = position
        self.drug_hypergraph_adj = drug_hypergraph_adj

        self.label_mean = float(label_mean)
        self.label_std = float(label_std)
        self.normalize_label = normalize_label

    def __len__(self):
        return len(self.position)

    def __getitem__(self, index):
        pair_idx = self.position[index]
        cell_idx = int(self.pair[pair_idx][0])
        drug_idx = int(self.pair[pair_idx][1])
        ic50_value = float(self.pair[pair_idx][2])

        # ✅ label 标准化
        if self.normalize_label:
            ic50_value = (ic50_value - self.label_mean) / (self.label_std + 1e-8)

        drug_graph = self.drug_graphs[drug_idx]
        exp = self.cell_expression[cell_idx]
        device = exp.device
        drug_graph = drug_graph.to(device)

        mut = self.cell_mutation[cell_idx].to(device)
        meth = self.cell_methylation[cell_idx].to(device)
        path = self.cell_pathway[cell_idx].to(device)

        # ✅ 返回 label 为 tensor，避免后面 dtype/shape 混乱
        label = torch.tensor(ic50_value, dtype=torch.float32, device=device)

        if self.drug_hypergraph_adj is not None:
            hyper_adj_vec = self.drug_hypergraph_adj[drug_idx].to(device)
            return drug_graph, exp, mut, meth, path, label, hyper_adj_vec

        return drug_graph, exp, mut, meth, path, label


def data_process(mut_feature, exp_feature, methy_feature, pathway_feature, pair,
                 drug_graph_map, drug_brics_fragments_map, all_brics_fragments, drug_id_mapping, cfg):
    # 1) Sd
    smiles_df = pd.read_csv(cfg.path.smiles)
    D, valid_drug_ids = build_drug_fingerprint_matrix(smiles_df)
    Sd, valid_drug_ids = compute_drug_similarity_matrix(D, valid_drug_ids, K=10)

    # 2) Sc
    Sc_original, cell_int_ids_original = generate_cell_similarity_matrix_Sc(exp_feature, K=10)

    # 3) A align
    A_ic50, A, valid_cell_ids, valid_drug_ids_A = get_aligned_response_matrix(
        gdsc_path=cfg.path.response,
        sc_cell_lines=cell_int_ids_original,
        sd_drugs=valid_drug_ids,
        threshold_ic50=1.0
    )

    # 4) align Sc/Sd
    valid_sc_idx = [cell_int_ids_original.index(cid) for cid in valid_cell_ids]
    Sc = Sc_original[valid_sc_idx, :][:, valid_sc_idx]

    valid_sd_idx = [valid_drug_ids.index(did) for did in valid_drug_ids_A]
    Sd = Sd[valid_sd_idx, :][:, valid_sd_idx]
    D = D[valid_sd_idx, :]

    # 5) Hd
    Hd, hyperedges, hyperedge_map = build_drug_hypergraph(Sd, Sc, D, A, valid_drug_ids_A)
    Hd_tensor = torch.from_numpy(Hd.astype("float32"))

    # ✅ 6) 药物图对齐：按 valid_drug_ids_A 顺序组装 aligned_drug_graphs
    aligned_drug_graphs = []
    missing = 0
    for did in valid_drug_ids_A:
        g = drug_graph_map.get(did, None)
        if g is None:
            missing += 1
            # 缺失就用“空图特征兜底”（尽量不断训练）
            from torch_geometric.data import Data
            g = Data(x=torch.zeros((1, 6), dtype=torch.float),
                     edge_index=torch.empty((2, 0), dtype=torch.long),
                     edge_attr=torch.empty((0, 1), dtype=torch.float))
        aligned_drug_graphs.append(g)

    # ✅ 7) drug_dict 用最终对齐顺序
    drug_dict = {did: idx for idx, did in enumerate(valid_drug_ids_A)}
    cell_dict = {cid: idx for idx, cid in enumerate(valid_cell_ids)}

    # 8) 过滤 pair -> (cell_idx, drug_idx, ic50)
    all_pairs = []
    valid_cell_set = set(valid_cell_ids)
    valid_drug_set = set(valid_drug_ids_A)
    for cell_int_id, drug_int_id, ic50 in pair:
        cell_int_id = extract_int_id(cell_int_id)
        drug_int_id = extract_int_id(drug_int_id)
        if cell_int_id in valid_cell_set and drug_int_id in valid_drug_set:
            all_pairs.append([cell_dict[cell_int_id], drug_dict[drug_int_id], ic50])
    all_pairs = np.array(all_pairs, dtype=np.float32)

    # ===== Label 标准化参数（仅使用训练集统计）=====

    if len(all_pairs) == 0:
        raise ValueError("无有效配对数据")

    # 9) cell tensors（这里先保持你原样；你之后可再加标准化）
    valid_cell_indices = [exp_feature.index.tolist().index(cid) for cid in valid_cell_ids]
    mutation_np = mut_feature.values[valid_cell_indices].astype("float32")
    expression_np = exp_feature.values[valid_cell_indices].astype("float32")
    methylation_np = methy_feature.values[valid_cell_indices].astype("float32")
    pathway_np = pathway_feature.values[valid_cell_indices].astype("float32")
    mutation_np = np.nan_to_num(mutation_np, nan=0.0, posinf=0.0, neginf=0.0)
    expression_np = np.nan_to_num(expression_np, nan=0.0, posinf=0.0, neginf=0.0)
    methylation_np = np.nan_to_num(methylation_np, nan=0.0, posinf=0.0, neginf=0.0)
    pathway_np = np.nan_to_num(pathway_np, nan=0.0, posinf=0.0, neginf=0.0)

    # 10) split + loaders
    train_index, temp_index = train_test_split(range(len(all_pairs)), test_size=0.2, random_state=42)
    val_index, test_index = train_test_split(temp_index, test_size=0.5, random_state=42)

    y_train = all_pairs[train_index, 2].astype(np.float32)
    label_mean = float(y_train.mean())
    label_std = float(y_train.std() + 1e-8)

    # ===== 使用训练集细胞统计对四路细胞特征进行标准化 =====
    train_cell_ids = sorted(set(all_pairs[train_index, 0].astype(int).tolist()))
    def std_norm(X, train_ids):
        if len(train_ids) == 0:
            return X
        m = X[train_ids].mean(axis=0)
        s = X[train_ids].std(axis=0) + 1e-8
        return (X - m) / s
    expression_np = std_norm(expression_np, train_cell_ids)
    mutation_np = std_norm(mutation_np, train_cell_ids)
    methylation_np = std_norm(methylation_np, train_cell_ids)
    pathway_np = std_norm(pathway_np, train_cell_ids)

    # 转为 tensor
    mutation = torch.from_numpy(mutation_np)
    expression = torch.from_numpy(expression_np)
    methylation = torch.from_numpy(methylation_np)
    pathway = torch.from_numpy(pathway_np)

    def build_loader(position, shuffle_flag):
        return DataLoader(
            my_dataloader(
                drug_graphs_aligned=aligned_drug_graphs,
                cell_expression=expression,
                cell_mutation=mutation,
                cell_methylation=methylation,
                cell_pathway=pathway,
                pair=all_pairs,
                position=position,
                drug_hypergraph_adj=Hd_tensor,
                label_mean=label_mean,
                label_std=label_std,
                normalize_label=True
            ),
            batch_size=cfg.model.batch_size,
            shuffle=shuffle_flag,
            collate_fn=collate_fn,
            num_workers=0,
            drop_last=False
        )

    train_set = build_loader(train_index, True)
    val_set = build_loader(val_index, False)
    test_set = build_loader(test_index, False)
    full_data_set = build_loader(range(len(all_pairs)), False)

    return (
        train_set, test_set, val_set, full_data_set,
        Hd_tensor, hyperedges, drug_dict, cell_dict, valid_cell_ids,
        aligned_drug_graphs, valid_drug_ids_A,label_mean,label_std  # ✅ 额外返回：对齐药物图 + 对齐药物顺序
    )
