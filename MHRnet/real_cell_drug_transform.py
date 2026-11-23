import pandas as pd

# 1. 加载数据
cell_info = pd.read_csv("data/CELL/cell_line_info.csv")  # 细胞系数据
drug_info = pd.read_csv("data/drug_info/drug_info.csv")  # 药物数据
ic50_data = pd.read_csv("output_dir/test_predictions_aligned.csv")  # IC50数据

# 2. 创建细胞ID到细胞名称的映射
cell_mapping = dict(zip(cell_info['DepMap_ID'], cell_info['stripped_cell_line_name']))

# 3. 创建药物ID到药物名称的映射
drug_mapping = dict(zip(drug_info['pubchem_id'], drug_info['drug_name']))

# 4. 替换IC50数据中的细胞ID和药物ID
ic50_data['cell_name'] = ic50_data['cell'].map(cell_mapping)
ic50_data['drug_name'] = ic50_data['drug'].map(drug_mapping)

# 5. 生成新的文件，包含真实的细胞名称和药物名称
ic50_data.to_csv('output_dir/final_all_predictions_real_name.csv', index=False)

print("文件已生成，包含真实细胞和药物名称。")
