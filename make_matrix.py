#%%
import re
import glob
import pandas as pd 
from scipy.sparse import csgraph
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import f_oneway
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib as mpl

# sort files numerically
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return(parts)

def day_weight(d1, d2, d3=None, d4=None):
    # calculate how the pattern that day compares with all other days
    if d3 is not None:
        avgNeighbors = (d2 + d3)/2
        return np.sqrt(np.sum((d1 - avgNeighbors) ** 2)) # euclidean distance
    else:
        return np.sqrt(np.sum((d1 - d2) ** 2)) # euclidean distance

    # if (d3 is None):
    #     F, _ = f_oneway(d1, d2) # F-test
    # elif (d4 is None):
    #     F, _ = f_oneway(d1, d2, d3) # F-test
    # else:
    #     F, _ = f_oneway(d1, d2, d3, d4) # F-test
    # return F

def regularized_svd(X, B, rank, alpha, as_sparse=False):
    """
    Perform graph regularized SVD as defined in
    Vidar & Alvindia (2013).

    Parameters
    ----------
    X : numpy array
        m x n data matrix.

    B : numpy array
        n x n graph Laplacian of nearest neighborhood graph of data.

    W : numpy array
        n x n weighted adjacency matrix.

    rank : int
        Rank of matrix to approximate.

    alpha : float
        Scaling factor.

    as_sparse : bool
        If True, use sparse matrix operations. Default is False.

    Returns
    -------
    H_star : numpy array
        m x r matrix (Eq 15).

    W_star : numpy array
        r x n matrix (Eq 15).
    """
    import numpy as np
    from scipy.linalg import svd
    # from scipy.linalg import cholesky
    from numpy.linalg import cholesky
    from scipy.linalg import inv
    import scipy.sparse as sp
    from sklearn.utils.extmath import randomized_svd
    from sksparse import cholmod

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



pathIn = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/keypress/'
pathOut = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/graph_regularized_SVD/matrices/'

# list of user accel files
all_files = sorted(glob.glob(pathIn + "*.csv"), key = numericalSort)

for file in all_files:

    df = pd.read_csv(file, index_col=False)

    user = int(df['userID'].unique())
    print('user: {}'.format(user))

    # if user != 10:
    #     continue

    # if user not in [89]: #89,96,102
    #     continue

    df['hour'] = pd.to_datetime(df['keypressTimestampLocal']).dt.hour

    M = df.groupby(['dayNumber','hour'],as_index = False).size().pivot('dayNumber','hour').fillna(0)

    if M.shape[0] < 25:
        print('not enough days')
        continue

    missingHours = [h for h in range(24) if h not in list(M['size'].columns)] #M.index

    M.columns = M.columns.droplevel(0)

    for h in missingHours:
        # M.loc[h] = [0]*M.shape[1] # to add row
        M.insert(h,h,[0]*M.shape[0])

    M = M.sort_index(ascending=True)
    M_copy = M

    avgMax = M.max().mean()
    if avgMax < 200:
        print('not enough kp per day')
        continue
    
    #######################################################################################

    # ADJACENCY MATRIX OF SIZE DAYS X DAYS

    # days = rows
    # hours = columns

    M = np.array(M_copy)

    # hard code adjacency matrix
    W = np.zeros((M.shape[0], M.shape[0]))

    for i in range(M.shape[0]):
        # print('row: {}'.format(i))
        for j in range(M.shape[0]):
            # print('col: {}'.format(j))
            # diagonals
            if i == j:
                W[i,j] = 0
                # print('diagonal')
            
            # previous days
            elif (i-j) == 1:
                # print('day connection')
                W[i,j] = day_weight(M[i,:], M[j,:], M[j-1,:]) #, M[j-2,:])
            # future days
            elif (i-j) == -1:
                # print('day connection')
                if j == M.shape[0]-1:
                    W[i,j] = day_weight(M[i,:], M[j,:])
                # elif j == M.shape[0]-2:
                #     W[i,j] = day_weight(M[i,:], M[j,:], M[j+1,:])
                else:
                    W[i,j] = day_weight(M[i,:], M[j,:], M[j+1,:]) #, M[j+2,:])

            else:
                W[i,j] = 0
    #             print('no connection')

    #         print('-----------------')

    #     print('==========================================')

            
    # print('finish with weighted adjacency matrix')


    # testM = np.array([[1,2,3,4],
    #              [5,6,7,8],
    #              [9,10,11,12],
    #              [13,14,15,16]])

    # test2=np.ones((5,5))

    

    B = csgraph.laplacian(W)
    M = M_copy.T

    # H_star, W_star = regularized_svd(M, B, rank=1, alpha=0.1, as_sparse=False)
    try:
        H_star, W_star = regularized_svd(M, B, rank=1, alpha=0.1, as_sparse=False)
    except:
        print('PD error')
        continue
    out=H_star@W_star
    out = out.T

    # fig, axs = plt.subplots(1, 2)
    # fig.suptitle('Typing Activity by Hour and Day')
    # axs[0].imshow(M_copy, cmap='viridis', 
    #                          vmin = 0,
    #                          vmax = 200)
    # axs[0].set_xlabel('Hour')
    # axs[0].set_ylabel('Day')
    # axs[0].set_title('Original')
    # axs[1].imshow(out, cmap='viridis', 
    #                          vmin = 0,
    #                          vmax = 200)
    # axs[1].set_title('Graph Regularized SVD')
    # fig.colorbar(plt.imshow(out, cmap='viridis', 
    #                          vmin = 0,
    #                          vmax = 200), ax=fig.get_axes())
    # axs[1].set_xlabel('Hour')
    # # axs[1].set_ylabel('Day')
    # plt.show()
    # # plt.savefig(pathOut+'user_{}.png'.format(user))

    ##############################################################
    
    clip_amount = out.max().mean()/4
    cutoff = np.clip(out, 0, clip_amount)

    # Reshape data into observations x features
    # Columns (features): [day, hour, keypresses]
    # Rows (observations)

    X = cutoff.T
    n_hours = X.shape[0]
    n_days = X.shape[1]

    # # Heatmap of original data
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(X, square=True, cmap='viridis')
    # plt.xlabel('Day')
    # plt.ylabel('Hour')

    # Reshape data into observations x features
    # Columns (features): [day, hour, keypresses]
    # Rows (observations): 726
    df = pd.DataFrame(X.T)
    df = df.melt(var_name='hour', value_name='keypresses')
    df['day'] = (pd.Series(np.arange(n_days))
                .repeat(n_hours).reset_index(drop=True))
    df = df[['day', 'hour', 'keypresses']]  # rearrange columns

    # # Original data space
    # f, ax = plt.subplots(subplot_kw={'projection': '3d'})
    # ax.scatter(df['hour'], df['day'], df['keypresses'])
    # ax.set(xlabel='Hour', ylabel='Day', zlabel='# keypresses')

    # Something simple for first approach: PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df.to_numpy())
    kmeans = KMeans(n_clusters=2, random_state=123).fit(X_pca)
    pca_df = pd.DataFrame({
        'pca_x': X_pca[:, 0],
        'pca_y': X_pca[:, 1],
        'cluster': kmeans.labels_})


    # # Visualize k-means clusters in PCA embedding
    # f, ax = plt.subplots()
    # sns.scatterplot(data=pca_df, x='pca_x', y='pca_y', hue='cluster', ax=ax)

    # Visualize original data heatmap and heatmap with k-means cluster labels
    f, ax = plt.subplots(nrows=2,ncols=2, sharex=False, sharey=True,
                        figsize=(10,10))
    sns.heatmap(M.T, cmap='viridis', ax=ax[0,0], vmin=0, vmax=500,
                cbar_kws={'label': '# keypresses', 'fraction': 0.043})
    sns.heatmap(out, cmap='viridis', ax=ax[0,1], vmin=0, vmax=500,
                cbar_kws={'label': '# keypresses', 'fraction': 0.043})
    sns.heatmap(cutoff, cmap='viridis', ax=ax[1,0], vmin=0, vmax=clip_amount,
                cbar_kws={'label': '# keypresses', 'fraction': 0.043})
    # Reshape k-means cluster labels from 726d vector to 22x33
    cluster_mat = pca_df['cluster'].to_numpy().reshape(X.shape)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom',
        colors=['#de8f05', '#0173b2'], #sns.color_palette('colorblind')[:2],
        N=2
    )
    sns.heatmap(cluster_mat.T, ax=ax[1,1], cmap=cmap,
                cbar_kws={'fraction': 0.043})
    colorbar = ax[1,1].collections[0].colorbar
    colorbar.set_ticks([0.25, 0.75])
    colorbar.set_ticklabels(['0', '1'])
    colorbar.set_label('Cluster')
    ax[0,0].set(title='Original', xlabel='Hour', ylabel='Day')
    ax[0,1].set(title='Graph Reg. SVD', xlabel='Hour', ylabel='Day')
    ax[1,0].set(title='Truncated Graph Reg. SVD', xlabel='Hour', ylabel='Day')
    ax[1,1].set(title='K-Means Clustering from PCA', xlabel='Hour', ylabel='Day')
    f.tight_layout()
    # plt.show(f)
    f.savefig(pathOut+'user_{}_svd_PCA-kmeans-v7.png'.format(user))
    plt.close(f)
    

    print('==========================================')
    
print('finish')

#%%

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

# ## COPY

# # hard code adjacency matrix
# W = np.zeros((M.shape[0], M.shape[0]))

# for i in range(M.shape[0]): # rows = days
#     print('row: {}'.format(i))

#     for j in range(M.shape[0]): # columns = hours
#         print('col: {}'.format(j))
#         # loop through hours within day
#         # connect hours within one day(inside borders)
#         if j != M.shape[1]-1: 
#             # start with hours 
#             if j == i:
#                 W[i,j] = 9999 # same cell not adjacent to itself
#              # cell to left
#             elif j == i-1:
#                 W[i,j] = hour_weight(M[i,j],M[i,j+1]) # cell and cell to the left
#             # cell to right
#             elif j == i+1:
#                 W[i,j] = hour_weight(M[i,j],M[i,j+1]) # cell and cell to the right
#             # nonadjacent cells
#             else:
#                 W[i,j] = 0        
#         # connect 23hr and 00hr
#         else:
#             print('at border')
#             if i != j:
#                 if i != M.shape[0]-1:
#                     W[i,j] = hour_weight(M[i,j],M[i+1,0])
#                 else:
#                     W[i,j] = 1111
#             else: # for last cell of data
#                 W[i,j] = 9999 # diagonals

#%%
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

# # REGULAR SVD

# # days = rows
# # hours = columns

# import numpy as np
# import matplotlib.pyplot as plt

# print('before')
# plt.imshow(M)
# plt.show()

# # sig = (2/(1 + np.exp(-M)))-1
# # cutoff = np.clip(M, 0, 100)
# # print('after transformation')
# # plt.imshow(cutoff)
# # plt.show()

# # u, s, vt  = np.linalg.svd(M)
# # S = np.diag(s)
# # r=1
# # svd_M = u[:, :r] @ S[0:r, :r] @ vt[:r, :]
# # print('after')
# # plt.imshow(svd_M)
# # plt.show()
        
# # %%

# import numpy as np

# # days = rows
# # hours = columns

# def day_weight(d1,d2):
#     return 1#((d1+d2)/2) + 1
# def hour_weight(h1,h2):
#     return 1#((h1+h2)/4) + 1

# M = np.array(M_copy)

# # M = np.array([[1,2,3,4],
# #              [5,6,7,8],
# #              [9,10,11,12],
# #              [13,14,15,16],
# #              [17,18,19,20]])

# # hard code adjacency matrix
# W = np.zeros((M.size, M.size))

# for i in range(M.size):
#     # print('row: {}'.format(i))
#     for j in range(M.size):
#         # print('col: {}'.format(j))

#         # iterate across hours of each day then across days
#         # d1h1, d1h2, d1h3, d1h4...d2h1, d2h2, d3h3...
#         i_Mi = i//M.shape[1]
#         i_Mj = i%M.shape[1]
#         # print('i: M[{},{}]'.format(i_Mi,i_Mj))
#         # print('val: {}'.format(M[i_Mi,i_Mj]))
#         j_Mi = j//M.shape[1]
#         j_Mj = j%M.shape[1]
#         # print('j: M[{},{}]'.format(j_Mi,j_Mj))
#         # print('val: {}'.format(M[j_Mi,j_Mj]))

#         # diagonals
#         if i == j:
#             W[i,j] = 0
#             # print('diagonal')

#         # # # # avoid points at boundaries
#         # # # elif (i_Mj != M.shape[1]-1):

#         # if abs(subtraction of col indices) == 1 & subtraction of row indices == 0:
#         elif (abs(j_Mj-i_Mj) == 1) & ((j_Mi-i_Mi) == 0):
#             W[i,j] = hour_weight(M[i_Mi,i_Mj],M[j_Mi,j_Mj])
#             # print('hour connection')
        
#         # if abs(subtraction of row indices) == 1 & subtraction of col indices == 0:
#         elif (abs(j_Mi-i_Mi) == 1) & ((j_Mj-i_Mj) == 0):
#             W[i,j] = day_weight(M[i_Mi,i_Mj],M[j_Mi,j_Mj])
#             # print('day connection')
        
#         # connect 23hr with 00hr
#         elif (i_Mj == M.shape[1]-1) & ((j_Mi-i_Mi) == 1) & (j_Mj == 0):
#             W[i,j] = hour_weight(M[i_Mi,i_Mj],M[i_Mi+1,0])
#             # print('boundary hour connection')

#         else:
#             W[i,j] = 0
#             # print('no connection')

#         # print('-----------------')
            

#         # # # # connect 23hr with 00hr
#         # # # # if point is at boundary of row (23hr)
#         # # # elif (i_Mj == M.shape[1]-1):
            
#         # # #     if ((j_Mj-i_Mj) == 1) & ((j_Mi-i_Mi) == 0):
#         # # #         W[i,j] = W[i,j] = hour_weight(M[i_Mi,i_Mj],M[i_Mi+1,0])
#         # # #         print('boundary hour connection')

#         # # #     if ((j_Mj-i_Mj) == -1) & ((j_Mi-i_Mi) == 0):
#         # # #         W[i,j] = W[i,j] = hour_weight(M[i_Mi,i_Mj],M[i_Mi+1,0])
#         # # #         print('non-boundary hour connection')

#         # # #     # if abs(subtraction of row indices) == 1 & subtraction of col indices == 0:
#         # # #     elif (abs(j_Mi-i_Mi) == 1) & ((j_Mj-i_Mj) == 0):
#         # # #         W[i,j] = day_weight(M[i_Mi,i_Mj],M[j_Mi,j_Mj])
#         # # #         print('boundary day connection')

#     # print('==========================================')


        
# print('finish with weighted adjacency matrix')

# # #%%
# # pd.DataFrame(W).to_csv(pathOut+'adj_matrix_test.csv',index=False,header=False)

# #%%
# from scipy.sparse import csgraph

# # testM = np.array([[1,2,3,4],
# #              [5,6,7,8],
# #              [9,10,11,12],
# #              [13,14,15,16]])

# # test2=np.ones((5,5))

# B = csgraph.laplacian(W)

# import numpy as np
# from scipy.linalg import svd
# from scipy.linalg import cholesky
# from scipy.linalg import inv
# import scipy.sparse as sp
# from sklearn.utils.extmath import randomized_svd
# from sksparse import cholmod

# # H_star, W_star = regularized_svd(M, B, rank=1, alpha=0.1, as_sparse=False)

# alpha=0.1
# rank=1

# I = np.eye(B.shape[0])
# C = I + (alpha * B)
# D = cholesky(C)
# E, S, Fh = svd(M @ inv(D.T))
# E_tilde = E[:, :rank]  # rank-r approximation; H_star = E_tilde (Eq 15)
# H_star = E_tilde  # Eq 15
# W_star = E_tilde.T @ M @ inv(C)  # Eq 15