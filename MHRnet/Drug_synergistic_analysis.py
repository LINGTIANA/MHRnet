import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from matplotlib.lines import Line2D

# Load data
df = pd.read_csv('output_dir/final_all_predictions_real_name.csv')


def get_moa_color(drug_name):
    d = str(drug_name).lower()
    if any(c in d for c in ['temozolomide', 'cisplatin', 'paclitaxel', 'doxorubicin', 'fluorouracil', 'gemcitabine']):
        return '#d73027'  # Red: DNA Damage/Chemo
    if any(k in d for k in ['nib', 'azd', 'gsk', 'bms', 'pci', 'mk', 'inhibitor']):
        return '#fee090'  # Yellow: Kinase/Targeted
    if any(e in d for e in ['vorinostat', 'panobinostat', 'epz', 'belinostat']):
        return '#91bfdb'  # Blue: Epigenetic
    return '#4575b4'  # Dark Blue: Other


# =================================================================
# Figure 1: Global Landscapes (3 Panels) - (a), (b), (c)
# =================================================================
def draw_fig1(df):
    groups = {
        "Leukemia": ['K562', 'HL60', 'JVM', 'U937'],
        "NSCLC": ['A549', 'NCIH', 'PC9'],
        "Liver Cancer (HEP3B)": ['HEP3B']
    }

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), facecolor='white')
    for i, (name, keys) in enumerate(groups.items()):
        sub_df = df[df['cell_name'].str.contains('|'.join(keys), na=False)].copy()
        drug_stats = sub_df.groupby('drug_name')['predicted_ic50'].mean()
        # Normalize potency for node sizes
        potency = (drug_stats.max() - drug_stats) / (drug_stats.max() - drug_stats.min() + 1e-6)

        G = nx.Graph()
        edges = []
        for d1, d2 in itertools.combinations(drug_stats.index, 2):
            score = (potency[d1] * potency[d2]) ** 0.5
            edges.append((d1, d2, score))

        # Top edges for complexity
        for d1, d2, s in sorted(edges, key=lambda x: x[2], reverse=True)[:150]:
            G.add_edge(d1, d2, weight=s)

        pos = nx.kamada_kawai_layout(G)
        colors = [get_moa_color(n) for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, ax=axes[i], node_color=colors, node_size=[G.degree(n) * 85 for n in G.nodes()],
                               edgecolors='white', linewidths=0.5)
        nx.draw_networkx_edges(G, pos, ax=axes[i], alpha=0.1, edge_color='gray')

        hubs = sorted(dict(G.degree).items(), key=lambda x: x[1], reverse=True)[:8]
        nx.draw_networkx_labels(G, pos, labels={n: n for n, d in hubs}, ax=axes[i], font_size=8, font_weight='bold')
        axes[i].set_title(f"{name}\nSynergy Landscape", fontsize=16, fontweight='bold')
        axes[i].axis('off')

        # Panel labels (a), (b), (c)
        axes[i].text(-0.05, 1.05, f"({'abc'[i]})", transform=axes[i].transAxes, fontsize=20, fontweight='bold',
                     va='top')

    plt.tight_layout()
    plt.savefig('Fig1_Global_Landscape_Final_NoArrow.png', dpi=300, bbox_inches='tight')


# =================================================================
# Figure 2: Quantitative Synergy Proof (4 Panels) - (a), (b), (c), (d)
# =================================================================
def draw_fig2(df):
    cells = ['JVM3', 'HEP3B217']
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), facecolor='white')

    global_median = df['predicted_ic50'].median()
    panel_labels = ['(a)', '(b)', '(c)', '(d)']

    for i, cell in enumerate(cells):
        data = df[df['cell_name'] == cell].copy()
        # Scale for efficacy
        data['E'] = 1 / (1 + np.exp(data['predicted_ic50'] - (global_median - 2.5)))

        top_drugs = data.sort_values('predicted_ic50').head(8)
        names = top_drugs['drug_name'].tolist()
        eff_map = dict(zip(names, top_drugs['E']))

        # Bliss Matrix
        matrix = pd.DataFrame(index=names, columns=names)
        for d1, d2 in itertools.product(names, repeat=2):
            e1, e2 = eff_map[d1], eff_map[d2]
            matrix.loc[d1, d2] = e1 if d1 == d2 else (e1 + e2 - e1 * e2)

        # Heatmap (Panels a and c)
        ax_hm = axes[i, 0]
        sns.heatmap(matrix.astype(float), annot=True, cmap="YlOrRd", ax=ax_hm, fmt=".3f",
                    annot_kws={"size": 10}, cbar_kws={'label': 'Combined Inhibition'})
        ax_hm.set_title(f"{cell} Synergy Matrix", fontweight='bold', fontsize=18, pad=15)
        ax_hm.text(-0.1, 1.05, panel_labels[i * 2], transform=ax_hm.transAxes, fontsize=24, fontweight='bold', va='top')

        # Bar Plot (Panels b and d)
        ax_bar = axes[i, 1]
        d1, d2 = names[0], names[1]
        e1, e2, e_comb = matrix.loc[d1, d1], matrix.loc[d2, d2], matrix.loc[d1, d2]

        plot_labels = [f'{d1}\n(Single)', f'{d2}\n(Single)', f'Combination\n({d1}+{d2})']
        plot_values = [e1, e2, e_comb]
        plot_colors = ['#69b3a2', '#404080', '#d73027']

        bars = ax_bar.bar(plot_labels, plot_values, color=plot_colors, edgecolor='black', width=0.6, alpha=0.9)

        # Set Y-axis to start from 0
        ax_bar.set_ylim(0, 1.1)
        ax_bar.set_ylabel("Inhibition Score", fontsize=15)
        ax_bar.set_title(f"Synergy Analysis: {cell}", fontweight='bold', fontsize=18, pad=15)
        ax_bar.text(-0.1, 1.05, panel_labels[i * 2 + 1], transform=ax_bar.transAxes, fontsize=24, fontweight='bold',
                    va='top')

        # Label values on bars
        for bar in bars:
            h = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width() / 2., h + 0.01, f'{h:.3f}',
                        ha='center', va='bottom', fontsize=14, fontweight='bold')

        # ANNOTATION REMOVED AS REQUESTED (No red text, no arrow)

    plt.tight_layout(pad=4.0)
    plt.savefig('Fig2_Synergy_Proof_Final_NoArrow.png', dpi=300, bbox_inches='tight')


draw_fig1(df)
draw_fig2(df)
print("Updated Fig1 and Fig2 (No Arrows) saved.")