import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
import pandas as pd
import scanpy as sc
sicle = '151672'
dataset = 'DLPFC'
results_file = f'output/DLPFC/{sicle}'
input_dir = os.path.join('data', dataset, sicle)

adata = sc.read_visium(path=input_dir, count_file=sicle + '_filtered_feature_bc_matrix.h5')

adata.var_names_make_unique()
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)
sc.pp.pca(adata, n_comps=50)


sc.pp.neighbors(adata, use_rep='X_pca')
sc.tl.louvain(adata, key_added='clusters', resolution=0.6)

#ari nmi
if dataset == 'DLPFC':
    label = pd.read_csv(os.path.join('data',dataset, sicle, sicle+'_truth.txt'), sep='\t', header=None, index_col=0)
    label.columns = ['Ground Truth']
    y = pd.factorize(label["Ground Truth"].astype("category"))[0]
    adata.obs['Ground Truth'] = label
else:
    label = pd.read_csv(os.path.join('data', dataset, sicle,'label_truth.txt'), sep='\t', header=None,
                                index_col=0)
    label.columns = ['over','Ground Truth']
    y = pd.factorize(label["Ground Truth"].astype("category"))[0]
    adata.obs['Ground Truth'] = label.iloc[:,1]


y_pred = adata.obs['clusters']
ARI = adjusted_rand_score(y, y_pred)
NMI = normalized_mutual_info_score(y, y_pred)
print('===== Project: {} ARI score: {:.4f}'.format(dataset, ARI))
print('===== Project: {} NMI score: {:.4f}'.format(dataset, NMI))

#spatial plot
plt.rcParams["figure.figsize"] = (4, 3)
sc.pl.spatial(adata, color="clusters", title=['SCANPY(ARI=%.2f)'%ARI], legend_fontsize=10,
                  na_in_legend=False)
plt.savefig(f'{results_file}/SCANPY_CLUSTER_plot.jpg')

#umap
sc.tl.umap(adata)
plt.rcParams["figure.figsize"] = (4, 3)
sc.pl.umap(adata, color=["clusters", "Ground Truth"],na_in_legend = False, legend_fontsize=10,wspace=0.25,
               title=['SCANPY', 'Ground_Truth'], legend_loc='right margin')
# sc.pl.umap(adata, color=["clusters", "Ground Truth"],
#                title=['SCANPY', 'Ground_Truth'], legend_loc='right margin')
plt.savefig(f'{results_file}/SCANPY_UMAP_plot.jpg')

#PAGA
used_adata = adata[adata.obs['Ground Truth'].notna()]
sc.tl.paga(used_adata, groups="Ground Truth")
plt.rcParams["figure.figsize"] = (4, 3)
sc.pl.paga_compare(used_adata, legend_fontsize=10, frameon=True, size=20,
                       title='SCANPY', legend_fontoutline=2, show=True)
plt.savefig(f'{results_file}/SCANPY_paga_plot.jpg')

