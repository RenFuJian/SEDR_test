import os

import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, \
                            homogeneity_completeness_v_measure
from sklearn.metrics.cluster import contingency_matrix
import numpy as np
import scanpy as sc
import stlearn as st
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import sys
import matplotlib.pyplot as plt

dataset = "osmFISH"
TILE_PATH = Path("/tmp/{}_tiles".format(dataset))
TILE_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = Path(f"./output/osmFISH/{dataset}/stLearn")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

count_patch = os.path.join('./data',dataset,'count.csv')
count = pd.read_csv(count_patch, sep=',', header=None)
adata_h5=sc.AnnData(count)


ground_truth_df = pd.read_csv('./data/'+dataset+'/labeltruth.txt',sep=',', header=None)
if dataset == 'osmFISH':
    ground_truth_df.columns = ['order','ground_truth']  #mouse embryo 24类  osmFISH 7类  MERFISH 15类
    adata_h5.obs['ground_truth'] = ground_truth_df.iloc[:,1]
elif dataset == 'Mouse embryo data':
    ground_truth_df.columns = ['ground_truth']  # mouse embryo 24类  osmFISH 7类  MERFISH 15类
    adata_h5.obs['ground_truth'] = ground_truth_df

le = LabelEncoder()     #LabelEncoder就是把n个类别值编码为0~n-1之间的整数，建立起1-1映射 这两行代码 都是
ground_truth_le = le.fit_transform(list(ground_truth_df["ground_truth"].values))
n_cluster = len((set(ground_truth_df["ground_truth"]))) - 1
ground_truth_df["ground_truth_le"] = ground_truth_le


pos = pd.read_csv('./data/'+dataset+'/pos.csv',sep=',', header=None)
x_array = pos.iloc[:,0]
y_array = pos.iloc[:,1]
coo = [x_array,y_array]
coo = np.array(coo).transpose()



# pre-processing for gene count table
st.pp.filter_genes(adata_h5,min_cells=1)
st.pp.normalize_total(adata_h5)
st.pp.log1p(adata_h5)

# run PCA for gene expression data
st.em.run_pca(adata_h5,n_comps=15)

# pre-processing for spot image
st.pp.tiling(adata_h5, TILE_PATH)

# this step uses deep learning model to extract high-level features from tile images
# may need few minutes to be completed
st.pp.extract_feature(adata_h5)

def calculate_clustering_matrix(pred, gt, sample, methods_):
    df = pd.DataFrame(columns=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"])

    pca_ari = adjusted_rand_score(pred, gt)
    df = df.append(pd.Series([sample, pca_ari, "pca", methods_, "Adjusted_Rand_Score"],
                             index=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"]), ignore_index=True)

    pca_nmi = normalized_mutual_info_score(pred, gt)
    df = df.append(pd.Series([sample, pca_nmi, "pca", methods_, "Normalized_Mutual_Info_Score"],
                             index=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"]), ignore_index=True)

    pca_purity = purity_score(pred, gt)
    df = df.append(pd.Series([sample, pca_purity, "pca", methods_, "Purity_Score"],
                             index=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"]), ignore_index=True)

    pca_homogeneity, pca_completeness, pca_v_measure = homogeneity_completeness_v_measure(pred, gt)

    df = df.append(pd.Series([sample, pca_homogeneity, "pca", methods_, "Homogeneity_Score"],
                             index=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"]), ignore_index=True)


    df = df.append(pd.Series([sample, pca_completeness, "pca", methods_, "Completeness_Score"],
                             index=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"]), ignore_index=True)

    df = df.append(pd.Series([sample, pca_v_measure, "pca", methods_, "V_Measure_Score"],
                             index=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"]), ignore_index=True)
    return df

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    cm = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)


# run stSME clustering
st.spatial.SME.SME_normalize(adata_h5, use_data="raw", weights="physical_distance")
data_ = adata_h5.copy()
data_.X = data_.obsm['raw_SME_normalized']
st.pp.scale(data_)
st.em.run_pca(data_,n_comps=30)
st.tl.clustering.kmeans(data_, n_clusters=n_cluster, use_data="X_pca", key_added="X_pca_kmeans")
ARI = adjusted_rand_score(data_.obs["X_pca_kmeans"],ground_truth_le)
NMI = normalized_mutual_info_score(data_.obs["X_pca_kmeans"],ground_truth_le)
st.pl.cluster_plot(data_, use_label="X_pca_kmeans",title =['stlearn (ARI=%.3f)'%ARI],show_plot = True)

#save result
methods_ = "stSME_disk"
results_df = calculate_clustering_matrix(data_.obs["X_pca_kmeans"], ground_truth_le, dataset, methods_)
results_df.to_csv(OUTPUT_PATH/'cluster_result.csv')
plt.savefig(OUTPUT_PATH / 'cluster.png')
print(['stlearn (ARI=%.4f)'%ARI])
print(['stlearn (ARI=%.4f)'%NMI])
print(results_df)
