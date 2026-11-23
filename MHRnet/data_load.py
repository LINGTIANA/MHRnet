import numpy as np
import pandas as pd
import re
from rdkit import Chem
from rdkit.Chem import BRICS
from torch_geometric.data import Data
import torch

def smiles_to_mol(smiles):
    return Chem.MolFromSmiles(smiles)

def mol_to_graph(mol):
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            atom.GetNumImplicitHs(),
            1 if atom.GetIsAromatic() else 0
        ])
    x = torch.tensor(atom_features, dtype=torch.float)

    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    edge_attr = []
    for bond in mol.GetBonds():
        edge_attr.append([bond.GetBondTypeAsDouble()])
    edge_attr = torch.tensor(edge_attr, dtype=torch.float) if len(edge_attr) else torch.empty((0, 1), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def extract_int_id(id_str):
    id_str = str(id_str)
    m = re.search(r"\d+", id_str)
    return int(m.group()) if m else hash(id_str) % 1000000

def mol_to_brics_fragments(mol):
    if not mol:
        return []
    frags = list(BRICS.BRICSDecompose(mol, returnMols=True))
    out = []
    for frag in frags:
        if frag and frag.GetNumAtoms() >= 2:
            s = Chem.MolToSmiles(frag)
            if len(s) >= 2 and "*" not in s:
                out.append(s)
    return list(set(out))

def dataload(**cfg):
    response_path = cfg['path']['response']
    mutation_path = cfg['path']['mutation']
    methylation_path = cfg['path']['methylation']
    expression_path = cfg['path']['expression']
    pathway_path = cfg['path']['pathway']
    smiles_path = cfg['path']['smiles']

    response_df = pd.read_csv(response_path, index_col=0)
    drug_ids = response_df.columns.values
    cell_ids = response_df.index.values

    valid_pairs = []
    for cell_id in cell_ids:
        cell_response = response_df.loc[cell_id]
        cell_int_id = extract_int_id(cell_id)
        for drug_id in drug_ids:
            ic50 = cell_response[drug_id]
            if not np.isnan(ic50):
                drug_int_id = extract_int_id(drug_id)
                valid_pairs.append([cell_int_id, drug_int_id, ic50])

    mut_feature = pd.read_csv(mutation_path, index_col=0)
    exp_feature = pd.read_csv(expression_path, index_col=0)
    methy_feature = pd.read_csv(methylation_path, index_col=0)
    pathway_df = pd.read_csv(pathway_path, index_col=0)

    def convert_index_to_int(df):
        df.index = [extract_int_id(idx) for idx in df.index]
        return df

    mut_feature = convert_index_to_int(mut_feature)
    exp_feature = convert_index_to_int(exp_feature)
    methy_feature = convert_index_to_int(methy_feature)
    pathway_df = convert_index_to_int(pathway_df)

    smiles_df = pd.read_csv(smiles_path)
    assert all(c in smiles_df.columns for c in ['drug_id', 'SMILES'])

    drug_id_mapping = {}  # original drug_id -> drug_int_id
    invalid_smiles = []

    # ✅ 核心：建立 drug_int_id -> graph 的映射（后面按对齐顺序取）
    drug_graph_map = {}
    drug_smiles_map = {}
    drug_brics_fragments_map = {}

    all_brics_fragments = set()

    for _, row in smiles_df.iterrows():
        original_drug_id = row['drug_id']
        drug_int_id = extract_int_id(original_drug_id)
        drug_id_mapping[original_drug_id] = drug_int_id

        smiles = row['SMILES']
        mol = smiles_to_mol(smiles)
        if not mol:
            invalid_smiles.append((original_drug_id, smiles))
            continue

        graph = mol_to_graph(mol)
        frags = mol_to_brics_fragments(mol)

        drug_graph_map[drug_int_id] = graph
        drug_smiles_map[drug_int_id] = smiles
        drug_brics_fragments_map[drug_int_id] = frags
        all_brics_fragments.update(frags)

    return (
        mut_feature, exp_feature, methy_feature, pathway_df, valid_pairs,
        drug_graph_map, drug_smiles_map, drug_brics_fragments_map, all_brics_fragments,
        drug_id_mapping
    )
