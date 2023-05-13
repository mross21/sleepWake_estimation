#%%
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
    from scipy.linalg import cholesky
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

#%%
import pandas as pd 

file = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/keypress/User_71_keypressDataMetrics.csv'
pathOut = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/graph_regularized_SVD/'

df = pd.read_csv(file, index_col=False)

df['hour'] = pd.to_datetime(df['keypressTimestampLocal']).dt.hour

M = df.groupby(['dayNumber','hour'],as_index = False).size().pivot('dayNumber','hour').fillna(0)

missingHours = [h for h in range(24) if h not in M.index]

for h in missingHours:
    M.loc[h] = [0]*M.shape[1]

M = M.sort_index(ascending=True)
M_copy = M

# #%%
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


# #%%
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


# ADJACENCY MATRIX OF SIZE DAYS X DAYS

import numpy as np

# days = rows
# hours = columns

def day_weight(d1,d2):
    # calculate how the pattern that day compares with all other days
    return np.sqrt(np.sum((d1 - d2) ** 2))
# def hour_weight(h1,h2):
#     return 1 #((h1+h2)/4) + 1

M = np.array(M_copy)

# M = np.array([[1,2,3,4],
#              [5,6,7,8],
#              [9,10,11,12],
#              [13,14,15,16],
#              [17,18,19,20],
#              [21,22,23,24],
#              [25,26,27,28]])

# hard code adjacency matrix
W = np.zeros((M.shape[0], M.shape[0]))

for i in range(M.shape[0]):
    print('row: {}'.format(i))
    for j in range(M.shape[0]):
        print('col: {}'.format(j))

        # # iterate across hours of each day then across days
        # # d1h1, d1h2, d1h3, d1h4...d2h1, d2h2, d3h3...
        # i_Mi = i//M.shape[1]
        # i_Mj = i%M.shape[1]
        # print('i: M[{},{}]'.format(i_Mi,i_Mj))
        # print('val: {}'.format(M[i_Mi,i_Mj]))
        # j_Mi = j//M.shape[1]
        # j_Mj = j%M.shape[1]
        # print('j: M[{},{}]'.format(j_Mi,j_Mj))
        # print('val: {}'.format(M[j_Mi,j_Mj]))

        # diagonals
        if i == j:
            W[i,j] = 0
            print('diagonal')
        
        # days
        elif abs(i-j) == 1:
            print('day connection')
            # if i >= M.shape[1]:
            #     print('bad i')
            #     print('+++++++++++++++++++++++++++++++++++++++++++++++++')
            # elif j >= M.shape[1]:
            #     print('bad j')
            #     print('+++++++++++++++++++++++++++++++++++++++++++++++++')
            # else:
            W[i,j] = day_weight(M[i,:], M[j,:])

        else:
            W[i,j] = 0
            print('no connection')

        print('-----------------')

    print('==========================================')

        
print('finish with weighted adjacency matrix')

# #%%

from scipy.sparse import csgraph

# testM = np.array([[1,2,3,4],
#              [5,6,7,8],
#              [9,10,11,12],
#              [13,14,15,16]])

# test2=np.ones((5,5))

B = csgraph.laplacian(W)
M = M_copy.T
import numpy as np
from scipy.linalg import svd
from scipy.linalg import cholesky
from scipy.linalg import inv
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd
from sksparse import cholmod

# H_star, W_star = regularized_svd(M, B, rank=1, alpha=0.1, as_sparse=False)

alpha = 0.1
rank = 1

I = np.eye(B.shape[0])
C = I + (alpha * B)
D = cholesky(C)
E, S, Fh = svd(M @ inv(D.T))
E_tilde = E[:, :rank]  # rank-r approximation; H_star = E_tilde (Eq 15)
H_star = E_tilde  # Eq 15
W_star = E_tilde.T @ M @ inv(C)  # Eq 15



plt.imshow(M_copy)
plt.show()


out=H_star@W_star
out = out.T
plt.imshow(out)
plt.show()





# cutoff = np.clip(out, 0, )

# plt.imshow(cutoff,vmin=0, vmax=20, aspect='auto')
# plt.show()
















#%%

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