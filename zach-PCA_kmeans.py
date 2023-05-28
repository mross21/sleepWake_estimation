#%%

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import svd
from scipy.linalg import cholesky
from scipy.linalg import inv
from scipy.io import loadmat
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sksparse import cholmod
import networkx as nx


def regularized_svd(X, B, rank, alpha, as_sparse=False):
    """
    Perform graph regularization SVD as defined in
    Vidar & Alvindia (2013).

    Parameters
    ----------
    X : numpy array
        d x n data matrix.

    B : numpy array
        n x n graph Laplacian of nearest neighborhood graph of data.

    rank : int
        Rank of matrix to approximate.

    alpha : float
        Scaling factor.

    as_sparse : bool
        If True, use sparse matrix operations. Default is False.

    Returns
    -------
    H_star : numpy array
        d x r matrix (Eq 15).

    W_star : numpy array
        r x n matrix (Eq 15).
    """
    if as_sparse:
        # Use sparse matrix operations to reduce memory
        I = sp.lil_matrix(B.shape)
        I.setdiag(1)
        C = I + (alpha * B)
        print('Computing Cholesky decomposition')
        factor = cholmod.cholesky(C)
        D = factor.L()
        print('Computing inverse of D.T')
        invDt = sp.linalg.inv(D.T)
        # Eq 11
        print('Computing randomized SVD')
        E, S, Fh = randomized_svd(X @ invDt,
                                  n_components=rank,
                                  random_state=123)
        E_tilde = E[:, :rank]  # rank-r approximation; H_star = E_tilde (Eq 15)
        H_star = E_tilde  # Eq 15
        W_star = E_tilde.T @ X @ sp.linalg.inv(C)  # Eq 15

    else:
        # Eq 11
        I = np.eye(B.shape[0])
        C = I + (alpha * B)
        D = cholesky(C)
        E, S, Fh = svd(X @ inv(D.T))
        E_tilde = E[:, :rank]  # rank-r approximation; H_star = E_tilde (Eq 15)
        H_star = E_tilde  # Eq 15
        W_star = E_tilde.T @ X @ inv(C)  # Eq 15
    return H_star, W_star


# sns.set_theme(style='ticks', font='Helvetica')

# rows = hours, columns = days
X = np.genfromtxt('/home/mindy/Desktop/BiAffect-iOS/UnMASCK/graph_regularized_SVD/U20-matrix.csv', delimiter=',')
n_hours = X.shape[0]
n_days = X.shape[1]

# Heatmap of original data
plt.figure(figsize=(8, 6))
sns.heatmap(X, square=True, cmap='viridis')
plt.xlabel('Day')
plt.ylabel('Hour')

# Reshape data into observations x features
# Columns (features): [day, hour, keypresses]
# Rows (observations): 726
df = pd.DataFrame(X.T)
df = df.melt(var_name='hour', value_name='keypresses')
df['day'] = (pd.Series(np.arange(n_days))
             .repeat(n_hours).reset_index(drop=True))
df = df[['day', 'hour', 'keypresses']]  # rearrange columns

# Original data space
f, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.scatter(df['hour'], df['day'], df['keypresses'])
ax.set(xlabel='Hour', ylabel='Day', zlabel='# keypresses')

# Something simple for first approach: PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df.to_numpy())
kmeans = KMeans(n_clusters=2, random_state=123).fit(X_pca)
pca_df = pd.DataFrame({
    'pca_x': X_pca[:, 0],
    'pca_y': X_pca[:, 1],
    'cluster': kmeans.labels_
})


# Visualize k-means clusters in PCA embedding
f, ax = plt.subplots()
sns.scatterplot(data=pca_df, x='pca_x', y='pca_y', hue='cluster', ax=ax)

# Visualize original data heatmap and heatmap with k-means cluster labels
f, ax = plt.subplots(ncols=2, sharex=True, sharey=True,
                     figsize=(14.41, 4.34))
sns.heatmap(X, cmap='viridis', square=True, ax=ax[0],
            cbar_kws={'label': '# keypresses', 'fraction': 0.043})
# Reshape k-means cluster labels from 726d vector to 22x33
cluster_mat = pca_df['cluster'].to_numpy().reshape(X.shape)
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom',
    colors=sns.color_palette('colorblind')[:2],
    N=2
)
sns.heatmap(cluster_mat, ax=ax[1], square=True, cmap=cmap,
            cbar_kws={'fraction': 0.043})
colorbar = ax[1].collections[0].colorbar
colorbar.set_ticks([0.25, 0.75])
colorbar.set_ticklabels(['0', '1'])
colorbar.set_label('Cluster')
ax[0].set(title='Data', xlabel='Day', ylabel='Hour')
ax[1].set(title='k-means clustering from PCA', xlabel='Day', ylabel='Hour')
f.tight_layout()
f.show()
# f.savefig('keypress_heatmaps.png')
# %%
