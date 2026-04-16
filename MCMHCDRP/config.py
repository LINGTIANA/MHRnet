from yacs.config import CfgNode as CN
import os

_C = CN()
_C.seed = 2020

_C.path = CN()
rootpath = os.path.dirname(os.path.realpath(os.path.abspath(__file__)))
_C.path.savedir = os.path.join(rootpath, "output_dir")
_C.path.response = os.path.join(rootpath, "data", "GDSC2_IC50.csv")
_C.path.pathway = os.path.join(rootpath, "data", "CELL", "pathway_cosmic.csv")
_C.path.mutation = os.path.join(rootpath, "data", "CELL", "geo_mutation_cosmic.csv")
_C.path.methylation = os.path.join(rootpath, "data", "CELL", "geo_methylation_cosmic.csv")
_C.path.expression = os.path.join(rootpath, "data", "CELL", "geo_expression_cosmic.csv")
_C.path.smiles = os.path.join(rootpath, "data", "cid_smiles.csv")

_C.model = CN()
_C.model.lr = 1e-4
_C.model.weight_decay = 1e-4
_C.model.epoch = 200
_C.model.cuda_id = 0
_C.model.batch_size = 128

_C.drug = CN()
_C.drug.drug_out_dim = 128

_C.cell = CN()
_C.cell.cell_out_dim = 128


_C.mlp = CN()
_C.mlp.mlp_in_dim = 256
_C.mlp.mlp_hidden_dim = [512, 128]

_C.hyper = CN()
_C.hyper.hyper_hidden_dim = 128
_C.hyper.drug_fusion_type = "attention"
_C.hyper.contrastive_weight = 0.10          # ✅ 降低对比权重（建议0.05~0.1）
_C.hyper.contrastive_temperature = 0.5
_C.hyper.hard_negative_ratio = 0.1

# ✅ 超图视图增强（超图-超图对比用）
_C.hyper.hyper_view_drop_prob = 0.2         # 超图向量 dropout
_C.hyper.hyper_view_noise_std = 0.05        # 超图向量噪声

def get_cfg_defaults():
    return _C.clone()

if __name__ == "__main__":
    cfg = get_cfg_defaults()
    os.makedirs(cfg.path.savedir, exist_ok=True)
    print(cfg)
