#
import os

import torch
import argparse
import warnings
import numpy as np
import pandas as pd
from src.graph_func import graph_construction
from src.utils_func import mk_dir, adata_preprocess, load_ST_file
import anndata
from src.SEDR_train import SEDR_Train
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.pyplot as plt
import scanpy as sc

warnings.filterwarnings('ignore')
torch.cuda.cudnn_enabled = False
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print('===== Using device: ' + device)



# ################ Parameter setting
parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=10, help='parameter k in spatial graph')
parser.add_argument('--knn_distanceType', type=str, default='euclidean',
                    help='graph distance type: euclidean/cosine/correlation')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--cell_feat_dim', type=int, default=50, help='Dim of PCA')
parser.add_argument('--feat_hidden1', type=int, default=100, help='Dim of DNN hidden 1-layer.')
parser.add_argument('--feat_hidden2', type=int, default=20, help='Dim of DNN hidden 2-layer.')
parser.add_argument('--gcn_hidden1', type=int, default=32, help='Dim of GCN hidden 1-layer.')
parser.add_argument('--gcn_hidden2', type=int, default=8, help='Dim of GCN hidden 2-layer.')
parser.add_argument('--p_drop', type=float, default=0.2, help='Dropout rate.')
parser.add_argument('--using_dec', type=bool, default=True, help='Using DEC loss.')
parser.add_argument('--using_mask', type=bool, default=False, help='Using mask for multi-dataset.')
parser.add_argument('--feat_w', type=float, default=10, help='Weight of DNN loss.')
parser.add_argument('--gcn_w', type=float, default=0.1, help='Weight of GCN loss.')
parser.add_argument('--dec_kl_w', type=float, default=10, help='Weight of DEC loss.')
parser.add_argument('--gcn_lr', type=float, default=0.01, help='Initial GNN learning rate.')
parser.add_argument('--gcn_decay', type=float, default=0.01, help='Initial decay rate.')
parser.add_argument('--dec_cluster_n', type=int, default=10, help='DEC cluster number.')
parser.add_argument('--dec_interval', type=int, default=20, help='DEC interval nnumber.')
parser.add_argument('--dec_tol', type=float, default=0.00, help='DEC tol.')
# ______________ Eval clustering Setting ______________
parser.add_argument('--eval_resolution', type=int, default=1, help='Eval cluster number.')
parser.add_argument('--eval_graph_n', type=int, default=20, help='Eval graph kN tol.')



params = parser.parse_args()
params.device = device
dataset = "Mouse embryo data"

# ################ Path setting
data_root = './data/'+dataset

# set saving result path
save_root = './output/'+dataset+'/'


def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    '''
        arg1(adata)[AnnData matrix]
        arg2(fixed_clus_count)[int]
        
        return:
            resolution[int]
    '''
    for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        if count_unique_leiden == fixed_clus_count:
            break
    return res

ARI_list = []
NMI_list = []

count_patch = os.path.join('./data',dataset,'count.csv')
count = pd.read_csv(count_patch, sep=',', header=None)
adata_h5=sc.AnnData(count)
adata_X = adata_preprocess(adata_h5, min_cells=5, pca_n_comps=params.cell_feat_dim)

pos = pd.read_csv('./data/'+dataset+'/pos.csv',sep=',', header=None)
x_array = pos.iloc[:,0]
y_array = pos.iloc[:,1]
coo = [x_array,y_array]
coo = np.array(coo).transpose()
graph_dict = graph_construction(coo, adata_h5.shape[0], params)
params.save_path = mk_dir(f'{save_root}/SEDR')

params.cell_num = adata_h5.shape[0]
print('==== Graph Construction Finished')


for seed in range(0,50):
    ################### Model training
    sedr_net = SEDR_Train(adata_X, graph_dict, params)
    if params.using_dec:
            sedr_net.train_with_dec()
    else:
        sedr_net.train_without_dec()
    sedr_feat, _, _, _ = sedr_net.process()

    np.savez(f'{params.save_path}/SEDR_result.npz', sedr_feat=sedr_feat, params=params)
    # ################## Result plot
    adata_sedr = anndata.AnnData(sedr_feat)
    # adata_sedr.uns['spatial'] = adata_h5.uns['spatial']
    # adata_sedr.obsm['spatial'] = adata_h5.obsm['spatial']

    sc.pp.neighbors(adata_sedr, n_neighbors=params.eval_graph_n)
    sc.tl.umap(adata_sedr)

    n_clusters = 7
    eval_resolution = res_search_fixed_clus(adata_sedr, n_clusters)
    sc.tl.leiden(adata_sedr, key_added="SEDR_leiden", resolution=eval_resolution)

    label = pd.read_csv('./data/'+dataset+'/labeltruth.txt',sep=',', header=None)
    if dataset == 'osmFISH':
        label.columns = ['order','Ground Truth']  #mouse embryo 24类  osmFISH 7类  MERFISH 15类
        y = pd.factorize(label["Ground Truth"].astype("category"))[0]
        adata_h5.obs['Ground Truth'] = label.iloc[:,1]
    elif dataset == 'Mouse embryo data':
        label.columns = ['Ground Truth']  # mouse embryo 24类  osmFISH 7类  MERFISH 15类
        y = pd.factorize(label["Ground Truth"].astype("category"))[0]
        adata_h5.obs['Ground Truth'] = label

    y_pred =  adata_sedr.obs['SEDR_leiden'].tolist()
    adata_sedr.obs['SEDR'] = y_pred

    ARI = adjusted_rand_score(y, y_pred)
    NMI = normalized_mutual_info_score(y, y_pred)
    print('===== Project: {} ARI score: {:.4f}'.format(dataset, ARI))
    print('===== Project: {} NMI score: {:.4f}'.format(dataset, NMI))
    ARI_list.append(ARI)
    NMI_list.append(NMI)


    print('===== Project: AVG ARI score: {:.4f}'.format(np.mean(ARI_list)))
    print('===== Project: AVG NMI score: {:.4f}'.format(np.mean(NMI_list)))





