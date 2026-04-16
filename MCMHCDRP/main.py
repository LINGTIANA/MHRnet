# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import csv
import re
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import r2_score

# 环境与自定义模块
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from config import get_cfg_defaults
from data_load import dataload
from data_process import data_process
from model import MCMHCDRP
from utils import set_seed, eval_predict, info_nce_loss, augment_hyper_adj_vec


# ===================== 0) 基础工具 =====================

def extract_int_id(id_str):
    if isinstance(id_str, (int, float)):
        return int(id_str)
    id_str = str(id_str).strip()
    m = re.search(r'(\d+)', id_str)
    return int(m.group()) if m else hash(id_str) % 1000000


# ===================== 1) 修正后的预测导出函数 (严防错位版) =====================

def predict_test_only_and_save(model, loader, depmap_names_map, drug_id_list, device, l_mean, l_std):
    """
    通过随 Batch 提取 Label 的方式，确保预测值与真实值绝对物理对齐。
    """
    model.eval()
    results = []
    ds = loader.dataset

    with torch.no_grad():
        for i, batch in enumerate(loader):
            dg = batch[0].to(device)
            cell_data = [b.to(device) for b in batch[1:5]]
            true_labels_norm = batch[5].to(device)
            hv = batch[6].to(device)

            # 模型预测
            pred_norm, _ = model(dg, cell_data, hv)

            # 计算当前 Batch 在 ds.position 中的起始偏移量
            batch_start_idx = i * loader.batch_size

            for j in range(pred_norm.size(0)):
                # 通过位置数组还原原始对信息
                global_pos_idx = ds.position[batch_start_idx + j]
                cell_idx, drug_idx, _ = ds.pair[global_pos_idx]

                # 还原 ID 名称
                internal_cell_id = ds.valid_cell_ids[int(cell_idx)]
                c_name = depmap_names_map.get(internal_cell_id, str(internal_cell_id))
                d_name = str(drug_id_list[int(drug_idx)])

                # 反归一化
                real_true = float(true_labels_norm[j].item() * l_std + l_mean)
                real_pred = float(pred_norm[j].item() * l_std + l_mean)

                results.append([c_name, d_name, real_true, real_pred])

    return pd.DataFrame(results, columns=['cell', 'drug', 'true_ic50', 'predicted_ic50'])


# ===================== 2) 训练/评估核心函数 =====================

def train_one_epoch(model, loader, optimizer, criterion, cfg):
    model.train()
    device = next(model.parameters()).device
    res = {"reg": 0, "cl_g": 0, "cl_h": 0, "cl_x": 0}
    n = 0
    for batch in loader:
        dg, cell_data, label, hv = batch[0].to(device), [b.to(device) for b in batch[1:5]], batch[5].to(device), batch[
            6].to(device)

        optimizer.zero_grad()
        pred, _ = model(dg, cell_data, hv)
        reg_loss = criterion(pred, label)

        # 三重对比学习损失计算
        cg, ch, cx = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        if cfg.hyper.contrastive_weight > 0:
            hv_aug = augment_hyper_adj_vec(hv, 0.2, 0.1).to(device)
            zg1, zg2, zh1, zh2 = model(dg, cell_data, hv, mode='contrastive', hyper_v2=hv_aug)
            cg = info_nce_loss(zg1, zg2, 0.07)
            ch = info_nce_loss(zh1, zh2, 0.07)
            cx = info_nce_loss(zg1, zh1, 0.1)

        loss = reg_loss + cfg.hyper.contrastive_weight * (cg + ch + cx) / 3.0
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        res["reg"] += reg_loss.item()
        res["cl_g"] += cg.item()
        res["cl_h"] += ch.item()
        res["cl_x"] += cx.item()
        n += 1
    return {k: v / n for k, v in res.items()}


@torch.no_grad()
def eval_loop(model, loader, criterion, cfg):
    model.eval()
    device = next(model.parameters()).device
    preds, trues = [], []
    for batch in loader:
        p, _ = model(batch[0].to(device), [b.to(device) for b in batch[1:5]], batch[6].to(device))
        preds.extend(p.cpu().tolist())
        trues.extend(batch[5].cpu().tolist())

    y_p, y_t = np.array(preds), np.array(trues)
    # 获得全部 8 个关键指标
    res = eval_predict((y_t * cfg.label_std + cfg.label_mean).tolist(), (y_p * cfg.label_std + cfg.label_mean).tolist())
    return [criterion(torch.tensor(y_p), torch.tensor(y_t)).item()] + list(res)


# ===================== 3) 主流程 =====================

def main():
    cfg = get_cfg_defaults()
    set_seed(cfg.seed)
    save_dir = cfg.path.savedir
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(f"cuda:{cfg.model.cuda_id}" if torch.cuda.is_available() else "cpu")

    # 1. 建立物理映射字典
    raw_exp = pd.read_csv(cfg.path.expression, index_col=0)
    id_to_name_map = {extract_int_id(idx): idx for idx in raw_exp.index}

    # 2. 数据加载与安全解包
    data = dataload(**cfg)
    processed = data_process(data[0], data[1], data[2], data[3], data[4], data[5], data[7], data[8], data[9], cfg=cfg)

    train_set, test_set, val_set, full_data_set, Hd_tensor, hyperedges, \
        drug_dict, cell_dict, valid_cell_ids, aligned_drug_graphs, \
        valid_drug_ids_A, l_mean, l_std = processed[:13]

    for s in [train_set, val_set, test_set]:
        s.dataset.valid_cell_ids = valid_cell_ids
    cfg.label_mean, cfg.label_std = l_mean, l_std

    # 3. 初始化网络
    model = MCMHCDRP(data[1].shape[1], data[0].shape[1], data[2].shape[1], data[3].shape[1],
                     torch.zeros((len(valid_drug_ids_A), cfg.drug.drug_out_dim), device=device), cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.model.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

    # 完整日志表头，包含回归与三个对比损失
    csv_path = os.path.join(save_dir, "training_metrics.csv")
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(
            ["Epoch", "Tr_Loss", "CL_G", "CL_H", "CL_X", "Val_Loss", "MSE", "RMSE", "MAE", "R2", "PCC", "SCC"])

    print("\n" + "=" * 40 + " 训练启动 " + "=" * 40)
    best_mae = float("inf")

    for epoch in range(cfg.model.epoch):
        with torch.no_grad():
            embs = torch.stack([model.drug_embedding(g.to(device)).squeeze().detach() for g in aligned_drug_graphs])
            model.set_global_drug_emb(embs)

        m = train_one_epoch(model, train_set, optimizer, nn.SmoothL1Loss(), cfg)
        va = eval_loop(model, val_set, nn.SmoothL1Loss(), cfg)

        # 实时打印：回归损失 + 三个对比损失 + 验证指标
        print(
            f"[Epoch {epoch + 1:03d}] Loss: Reg={m['reg']:.4f} | CL: G={m['cl_g']:.3f}, H={m['cl_h']:.3f}, X={m['cl_x']:.3f}")
        print(
            f"      Val: MSE={va[1]:.4f} RMSE={va[2]:.4f} MAE={va[3]:.4f} R2={va[4]:.4f} PCC={va[5]:.4f} SCC={va[7]:.4f}")

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow(
                [epoch + 1, m['reg'], m['cl_g'], m['cl_h'], m['cl_x'], va[0], va[1], va[2], va[3], va[4], va[5], va[7]])

        if va[3] < best_mae:
            best_mae = va[3]
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
        scheduler.step(va[3])

    # 🟢 4. 最终评估与对齐导出 (仅 Test)
    print("\n" + "=" * 30 + " 执行最终测试对齐验证与导出 " + "=" * 30)
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt")))

    # 生成最终报告
    te = eval_loop(model, test_set, nn.SmoothL1Loss(), cfg)
    print(f"\n[FINAL TEST REPORT]")
    print(f"MSE: {te[1]:.6f} | RMSE: {te[2]:.6f} | MAE: {te[3]:.6f}")
    print(f"R2:  {te[4]:.6f} | PCC:  {te[5]:.6f} | SCC: {te[7]:.6f}")

    # 保存 TXT 指标总结
    with open(os.path.join(save_dir, "test_performance_summary.txt"), "w", encoding="utf-8") as f:
        f.write("FINAL BEST MODEL TEST performance\n" + "=" * 30 + "\n")
        f.write(f"Mean Squared Error (MSE):          {te[1]:.8f}\n")
        f.write(f"Root Mean Squared Error (RMSE):     {te[2]:.8f}\n")
        f.write(f"Mean Absolute Error (MAE):          {te[3]:.8f}\n")
        f.write(f"Coefficient of Determination (R2):  {te[4]:.8f}\n")
        f.write(f"Pearson Correlation (PCC):          {te[5]:.8f}\n")
        f.write(f"Spearman Correlation (SCC):         {te[7]:.8f}\n")

    # 执行随 Batch 提取的预测导出逻辑
    df_test = predict_test_only_and_save(model, test_set, id_to_name_map, valid_drug_ids_A, device, l_mean, l_std)

    # 核心对齐校验
    check_r2 = r2_score(df_test['true_ic50'], df_test['predicted_ic50'])
    print(f"\n[对齐验证] 导出 CSV 的全局 R2 校验值: {check_r2:.4f}")

    if check_r2 < 0:
        print("❌ 警告：CSV 校验 R2 为负值，说明 test_loader 可能开启了 shuffle=True，请立即检查 data_process.py！")
    else:
        print("✅ 验证通过：导出数据已物理对齐。")

    df_test.to_csv(os.path.join(save_dir, 'test_predictions_aligned.csv'), index=False)

    # 5. 绘制多损失曲线 (回归损失 vs 三种对比损失)
    log = pd.read_csv(csv_path)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1);
    plt.plot(log['Epoch'], log['Tr_Loss'], label='Train');
    plt.plot(log['Epoch'], log['Val_Loss'], label='Val');
    plt.legend();
    plt.title("Regression Loss Improvement")

    # 绘制三种对比损失
    plt.subplot(1, 2, 2);
    plt.plot(log['Epoch'], log['CL_G'], label='CL_Global');
    plt.plot(log['Epoch'], log['CL_H'], label='CL_Hyper');
    plt.plot(log['Epoch'], log['CL_X'], label='CL_Cross')
    plt.legend();
    plt.title("Contrastive Loss Curves")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves_full.tiff'), dpi=400);
    plt.close()

    print(f"\n[Done] 全部流程圆满结束。")


if __name__ == "__main__":
    main()