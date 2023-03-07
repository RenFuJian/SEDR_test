import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
import pandas as pd
import scanpy as sc
sicle = 'Mouse embryo data'
dataset = 'Mouse embryo data'
results_file = f'output/{dataset}/'
input_dir = os.path.join('data', dataset)





# adata = sc.read_visium(path=input_dir, count_file=sicle + '_filtered_feature_bc_matrix.h5')


count_path = os.path.join('./data',dataset,'count.csv')
count = pd.read_csv(count_path, sep=',', header=None)
adata = sc.AnnData(count)
pos = pd.read_csv('./data/'+dataset+'/pos.csv',sep=',', header=None)
x_array = pos.iloc[:,0].tolist()
y_array = pos.iloc[:,1].tolist()
adata.obs['array_row'] = x_array
adata.obs['array_col'] = y_array

if dataset == 'osmFISH':
    label = pd.read_csv('data/'+dataset+'/labeltruth.txt',sep=',', header=None)
    label.columns = ['order','Ground Truth']  #mouse embryo 24类  osmFISH 7类  MERFISH 15类
    y = pd.factorize(label["Ground Truth"].astype("category"))[0]
    adata.obs['Ground Truth'] = label.iloc[:,1].tolist()
else:
    label = pd.read_csv(os.path.join('data', dataset, 'labeltruth.txt'), sep='\t')
    label.columns = ['Ground Truth']
    y = pd.factorize(label["Ground Truth"].astype("category"))[0]
    adata.obs['Ground Truth'] = label.iloc[:, 0].tolist()


adata.var_names_make_unique()
sc.pp.filter_genes(adata, min_cells=3)
# sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)
sc.pp.pca(adata, n_comps=50)


sc.pp.neighbors(adata, use_rep='X_pca')
sc.tl.louvain(adata, key_added='clusters', resolution=1.3)

#ari nmi

y_pred = adata.obs['clusters']
ARI = adjusted_rand_score(y, y_pred)
NMI = normalized_mutual_info_score(y, y_pred)
print('===== Project: {} ARI score: {:.4f}'.format(dataset, ARI))
print('===== Project: {} NMI score: {:.4f}'.format(dataset, NMI))
y_pred.to_csv(results_file+"/y_pred.csv")

#spatial plot
# plt.rcParams["figure.figsize"] = (6, 3)
# sc.pl.spatial(adata, color="clusters", title=['SCANPY(ARI=%.2f)'%ARI], legend_fontsize=10,
#                   na_in_legend=False)
# plt.savefig(f'{results_file}/SCANPY_CLUSTER_plot.jpg')

#umap
sc.tl.umap(adata)
plt.rcParams["figure.figsize"] = (6, 3)
sc.pl.umap(adata, color=["clusters","Ground Truth"],na_in_legend = False, legend_fontsize=6,
               title=["SCANPY",'Ground_Truth'])
plt.rcParams["figure.figsize"] = (6, 3)
sc.pl.umap(adata, color=["clusters"],na_in_legend = False, legend_fontsize=6,legend_fontoutline=2,
               title=['SCANPY'])
sc.pl.umap(adata, color=["Ground Truth"],na_in_legend = False, legend_fontsize=6,legend_fontoutline=2,
               title=['SCANPY'], legend_loc='on data')
plt.savefig(f'{results_file}/SCANPY_UMAP_plot.jpg')

#PAGA
used_adata = adata[adata.obs['Ground Truth'].notna()]
sc.tl.paga(used_adata, groups="Ground Truth")
plt.rcParams["figure.figsize"] = (6, 3)
sc.pl.paga_compare(used_adata, legend_fontsize=6, frameon=True, size=15,
                       title='SCANPY', legend_fontoutline=2, show=True)
plt.savefig(f'{results_file}/SCANPY_paga_plot.jpg')

