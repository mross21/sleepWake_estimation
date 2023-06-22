#%%
#%%
import re
import glob
import pandas as pd
from scipy.sparse import csgraph
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns

# sort files numerically
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return(parts)

def medianAAIKD(dataframe):
    grpAA = dataframe.loc[((dataframe['keypress_type'] == 'alphanum') &
                                (dataframe['previousKeyType'] == 'alphanum'))]
    # get median IKD
    medAAIKD = np.nanmedian(grpAA['IKD']) if len(grpAA) >= 20 else float('NaN')
    return(medAAIKD)

def closest_hour(lst, K):
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

def day_weight(d1,d2):
    return (d1+d2) 

def hour_weight(h1,h2):
    return (h1+h2)/2 

def weighted_adjacency_matrix(mat): # for raw data into SVD
    # days = rows
    # hours = columns
    W = np.zeros((mat.size, mat.size))
    for i in range(mat.size):
        for j in range(mat.size):
            # iterate across hours of each day then across days
            # d1h1, d1h2, d1h3, d1h4...d2h1, d2h2, d3h3...
            i_Mi = i//mat.shape[1]
            i_Mj = i%mat.shape[1]
            j_Mi = j//mat.shape[1]
            j_Mj = j%mat.shape[1]
            # diagonals
            if i == j:
                W[i,j] = 0
            # if abs(subtraction of col indices) == 1 & subtraction of row indices == 0:
            elif (abs(j_Mj-i_Mj) == 1) & ((j_Mi-i_Mi) == 0):
                W[i,j] = hour_weight(mat[i_Mi,i_Mj],mat[j_Mi,j_Mj])
            # if abs(subtraction of row indices) == 1 & subtraction of col indices == 0:
            elif (abs(j_Mi-i_Mi) == 1) & ((j_Mj-i_Mj) == 0):
                W[i,j] = day_weight(mat[i_Mi,i_Mj],mat[j_Mi,j_Mj])
            # connect 23hr with 00hr
            elif (i_Mj == mat.shape[1]-1) & ((j_Mi-i_Mi) == 1) & (j_Mj == 0):
                W[i,j] = hour_weight(mat[i_Mi,i_Mj],mat[i_Mi+1,0])
            else:
                W[i,j] = 0
    return W

def day_weight2(d1,d2):
    return 1/(abs(d1-d2)+.0000001)

def hour_weight2(h1,h2):
    return 1/(abs(h1-h2)+.0000001)

def weighted_adjacency_matrix2(mat): # adj matrix for graph reg. svd
    # days = rows
    # hours = columns
    W = np.zeros((mat.size, mat.size))
    for i in range(mat.size):
        for j in range(mat.size):
            # iterate across hours of each day then across days
            # d1h1, d1h2, d1h3, d1h4...d2h1, d2h2, d3h3...
            i_Mi = i//mat.shape[1]
            i_Mj = i%mat.shape[1]
            j_Mi = j//mat.shape[1]
            j_Mj = j%mat.shape[1]
            # diagonals
            if i == j:
                W[i,j] = 0
            # if abs(subtraction of col indices) == 1 & subtraction of row indices == 0:
            elif (abs(j_Mj-i_Mj) == 1) & ((j_Mi-i_Mi) == 0):
                W[i,j] = hour_weight2(mat[i_Mi,i_Mj],mat[j_Mi,j_Mj])
            # if abs(subtraction of row indices) == 1 & subtraction of col indices == 0:
            elif (abs(j_Mi-i_Mi) == 1) & ((j_Mj-i_Mj) == 0):
                W[i,j] = day_weight2(mat[i_Mi,i_Mj],mat[j_Mi,j_Mj])
            # connect 23hr with 00hr
            elif (i_Mj == mat.shape[1]-1) & ((j_Mi-i_Mi) == 1) & (j_Mj == 0):
                W[i,j] = hour_weight2(mat[i_Mi,i_Mj],mat[i_Mi+1,0])
            else:
                W[i,j] = 0
    return W

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


def cosine_similarity(a,b):
    from numpy.linalg import norm
    cosine = np.dot(a,b)/(norm(a)*norm(b))
    return cosine

def get_regularity(mat, day_diff):
    sim_list = []
    for d in range(mat.shape[0]):
        if d < mat.shape[0]-day_diff:
            sim = cosine_similarity(mat[d], mat[d+day_diff])
            sim_list.append(sim)
    return sim_list


############################################################################################
pathIn = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/keypress/'
pathOut = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/graph_regularized_SVD/matrices/'

# list of user accel files
all_files = sorted(glob.glob(pathIn + "*.csv"), key = numericalSort)

for file in all_files:
    df = pd.read_csv(file, index_col=False)
    user = int(df['userID'].unique())
    print('user: {}'.format(user))

    df['hour'] = pd.to_datetime(df['keypressTimestampLocal']).dt.hour
    M1 = df.groupby(['dayNumber','hour'],as_index = False).size().pivot('dayNumber','hour').fillna(0)

    # insert hours with no activity across all days
    missingHours = [h for h in range(24) if h not in list(M1['size'].columns)]
    M1.columns = M1.columns.droplevel(0)
    for h in missingHours:
        # M.loc[h] = [0]*M.shape[1] # to add row
        M1.insert(h,h,[0]*M1.shape[0])
    M1 = M1.sort_index(ascending=True)

    # find avg number of hours of activity/day
    Mbinary = np.where(M1 > 0, 1, 0)
    avgActivityPerDay = Mbinary.mean(axis=1).mean()
    print('avg n hours per day with typing activity: {}'.format(avgActivityPerDay))

    # of the days with kp, find median amount
    Mkp = np.where(M1 > 0, M1, np.nan)
    avgAmountPerDay = np.nanmedian(np.nanmedian(Mkp, axis=1))
    print('median amount of kp overall: {}'.format(avgAmountPerDay))

    if (avgActivityPerDay < 0.2) | (avgAmountPerDay < 50):
        print('not enough data')
        print('-----------------------------------------------')
        continue

    # remove first and last days
    M1 = M1[1:-1]

    # if less than 7 days, continue
    if M1.shape[0] < 7:
        print('not enough days')
        continue

    # get dimensions
    n_days = M1.shape[0]
    n_hrs = M1.shape[1]

    # hard code adjacency matrix
    W = weighted_adjacency_matrix(np.array(M1))

    kp_values = np.array(M1).flatten()
    days_arr = np.repeat(range(n_days), n_hrs)
    hrs_arr = np.array(list(range(n_hrs)) * n_days)
    data = np.vstack((days_arr,hrs_arr, kp_values))
    B = csgraph.laplacian(W)
    H_star, W_star = regularized_svd(data, B, rank=1, alpha=0.1, as_sparse=False)

    # output of SVD matrix 
    out2 = W_star.reshape(M1.shape)
    out2 = out2 * -1
    clip_amount = out2.max()/10
    cutoff = np.clip(out2, 0, clip_amount)

    # K-means of the output from SVD 
    dfWstar = pd.DataFrame(cutoff.reshape(-1,1), columns = ['vals'])
    kmeans = KMeans(n_clusters=2, random_state=123).fit(dfWstar)
    dfKmeans = pd.DataFrame({
        # 'pca_x': X_pca[:, 0],
        # 'pca_y': X_pca[:, 1],
        'graph_reg_SVD_vals': dfWstar['vals'],
        'cluster': kmeans.labels_})


    # Visualize original data heatmap and graph reg. SVD
    f, ax = plt.subplots(nrows=1,ncols=2, sharex=False, sharey=True,
                        figsize=(10,5))
    # PLOT 1
    sns.heatmap(M1, cmap='viridis', ax=ax[0], vmin=0, vmax=500,
                cbar_kws={'label': '# keypresses', 'fraction': 0.043})
    # PLOT 2
    sns.heatmap(out2, cmap='viridis', ax=ax[1], vmin=0, vmax=200,
                cbar_kws={'label': '# keypresses', 'fraction': 0.043})

    ax[0].set(title='Original', xlabel='Hour', ylabel='Day')
    ax[1].set(title='Graph Reg. SVD', xlabel='Hour', ylabel='Day')
    f.tight_layout()
    plt.show(f)
    # f.savefig(pathOut+'HRxDAYsizeMat/SVD/user_{}_graphRegSVD.png'.format(user))
    plt.close(f)


    # calculate regularity
    diff1 = get_regularity(out2, 1)
    diff2 = get_regularity(out2, 2)
    diff3 = get_regularity(out2, 3)
    diff4 = get_regularity(out2, 4)
    diff5 = get_regularity(out2, 5)
    diff6 = get_regularity(out2, 6)
    diff7 = get_regularity(out2, 7)
    dfRegularity = pd.DataFrame([diff1,diff2,diff3,diff4,
                                diff5,diff6,diff7]).T
    dfRegularity.columns = [1,2,3,4,5,6,7]
    fig, axes = plt.subplots(figsize=(5,5))
    sns.set(style="whitegrid")
    sns.boxplot(data=dfRegularity, ax = axes, orient ='v').set(title = 'User {} Regularity'.format(user))
    plt.xlabel('Days Apart')
    plt.ylabel('Cosine Similarity')
    plt.ylim([0, 1])
    plt.show()
    # plt.savefig(pathOut + 'HRxDAYsizeMat/regularity/user_{}_regularity.png'.format(user))
    plt.clf()

  