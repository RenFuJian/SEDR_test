import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.colors import ListedColormap
dataset = 'Mouse embryo data'

df = pd.read_csv('./data/'+dataset+'/pos.csv',sep=',', header=None)
x_array = df.iloc[:,0].tolist()
y_array = df.iloc[:,1].tolist()
y_pred = pd.read_csv(f'output/{dataset}/y_pred.csv')
df['cluster'] = y_pred['clusters']
df.columns=['x','y','cluster']


colors = ['#1f77b4', '#ff7f0e', '#279e68', '#d62728', '#aa40fc', '#8c564b', '#e377c2', '#b5bd61', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#dbdb8d', '#9edae5', '#ad494a', '#8c6d31','#023fa5','#7d87b9','#bec1d4','#d6bcc0']
df['c'] = df.cluster.map({0:colors[0], 1:colors[1], 2:colors[2], 3:colors[3], 4:colors[4], 5:colors[5], 6:colors[6],7:colors[7],8:colors[8],9:colors[9],10:colors[10],11:colors[11],12:colors[12],13:colors[13],14:colors[14]
                          ,15:colors[15],16:colors[16],17:colors[17],18:colors[18],19:colors[19],20:colors[20],21:colors[21],22:colors[22],23:colors[23]})
scatter = plt.scatter(x=df.x,y=df.y,c=df.c, alpha = 0.7, s=5)
# legend1 = ax.legend(*scatter.legend_elements(),
#                     loc="lower left", title="Classes")
# ax.add_artist(legend1)
# handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
plt.show()


label = pd.read_csv(os.path.join('data',dataset, 'labeltruth.txt'), sep='\t', header=None, index_col=0)
label.columns = ['Ground Truth']
df['truelabel'] = label['Ground Truth']
scatter = plt.scatter(x=df.x,y=df.y,c=df.c, alpha = 0.7, s=5)
