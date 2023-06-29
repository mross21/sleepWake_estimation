#%%
import re
import glob
import pandas as pd
from scipy.sparse import csgraph
import numpy as np
from sklearn.cluster import KMeans
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt

# sort files numerically
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return(parts)

def closest_hour(lst, K):
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

def day_weight(d1,d2):
    return (d1+d2)

def hour_weight(h1,h2):
    return (h1+h2)/2

def weighted_adjacency_matrix(mat):
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

############################################################################################
pathIn = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/keypress/'
pathOut = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/graph_regularized_SVD/matrices/'

# list of user accel files
all_files = sorted(glob.glob(pathIn + "*.csv"), key = numericalSort)

dfCircReg = pd.DataFrame({'user': [],
                          'date': [],
                          'circvar': [],
                          'circmean': [],
                          'amount_noActivity': []})

for file in all_files:
    df = pd.read_csv(file, index_col=False)
    user = int(df['userID'].unique())
    print('user: {}'.format(user))

    # if user != 11:
    #     continue

    df['hour'] = pd.to_datetime(df['keypressTimestampLocal']).dt.hour
    M = df.groupby(['dayNumber','hour'],as_index = False).size().pivot('dayNumber','hour').fillna(0)

    # insert hours with no activity across all days
    missingHours = [h for h in range(24) if h not in list(M['size'].columns)]
    M.columns = M.columns.droplevel(0)
    for h in missingHours:
        M.insert(h,h,[0]*M.shape[0])
    M1 = M.sort_index(ascending=True)

    # find avg number of hours of activity/day
    Mbinary = np.where(M > 0, 1, 0)
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
    M = M[1:-1]

    # if less than 7 days, continue
    if M.shape[0] < 7:
        print('not enough days')
        continue

    # # insert days with no activity across all hours
    # missingDays = [d for d in range(M.index[0],M.index[-1]) if d not in list(M.index)]
    # for d in missingDays:
    #     M.loc[d] = [0]*M.shape[1] # to add row
    # M = M.sort_index(ascending=True)


    n_days = M.shape[0]
    n_hrs = M.shape[1]

    # hard code adjacency matrix
    W = weighted_adjacency_matrix(np.array(M))

    kp_values = np.array(M).flatten()
    days_arr = np.repeat(range(n_days), n_hrs)
    hrs_arr = np.array(list(range(n_hrs)) * n_days)
    data = np.vstack((days_arr,hrs_arr, kp_values))
    B = csgraph.laplacian(W)
    H_star, W_star = regularized_svd(data, B, rank=1, alpha=0.1, as_sparse=False)

    out2 = W_star.reshape(M.shape)
    out2 = out2 * -1
    clip_amount = out2.max()/4
    cutoff = np.clip(out2, 0, clip_amount)

    # k means
    dfWstar = pd.DataFrame(cutoff.reshape(-1,1), columns = ['vals'])
    kmeans = KMeans(n_clusters=2, random_state=123).fit(dfWstar)
    dfKmeans = pd.DataFrame({'day': (pd.Series(np.arange(n_days)).repeat(n_hrs).reset_index(drop=True)),
        'graph_reg_SVD_vals': dfWstar['vals'],
        'cluster': kmeans.labels_})

    ## Circular variance/mean
    circVarList = np.apply_along_axis(stats.circvar, 1, out2)
    circMeanList = np.apply_along_axis(stats.circmean, 1, out2)

    # find sleep and wake labels
    cluster_mat = dfKmeans['cluster'].to_numpy().reshape(M.shape)
    m = np.array(M)
    # get mean matrix value for each label
    i0,j0=np.where(cluster_mat==0)
    vals0 = m[i0,j0]
    median0 = np.mean(vals0)
    i1,j1=np.where(cluster_mat==1)
    vals1 = m[i1,j1]
    median1 = np.mean(vals1)
    # assign lower value to sleep
    sleep_label = np.where(median0 < median1, 0, 1)
    wake_label = np.where(median0 > median1, 0, 1)

    ## OLD
    # wake_label_idx = cutoff[0].argmax()
    # wake_label = cluster_mat[0,wake_label_idx]
    # sleep_label_idx = cutoff[0].argmin()
    # sleep_label = cluster_mat[0,sleep_label_idx]

    print('sleep label {}'.format(sleep_label))
    if wake_label == sleep_label:
        print('sleep and wake labels are the same')
        print('++++++++++++++++++++++++')
        break

    plt.imshow(cluster_mat)
    plt.show()

## find # of hours of no activity
    noActivityAmt = dfKmeans.groupby('day').apply(lambda x: x[x['cluster'] == sleep_label].shape[0])

# make dataframe
    # dates = pd.date_range(df['date'].unique()[1], periods=n_days).tolist()
    dates = df['date'].unique()[1:-1]
    dfCircReg = dfCircReg.append(pd.DataFrame({'user': user,
                              'date': dates,
                              'circvar': circVarList,
                              'circmean': circMeanList,
                              'amount_noActivity': noActivityAmt}))

    
    # if user > 5:
    #     break

dfCircReg.to_csv('/home/mindy/Desktop/BiAffect-iOS/UnMASCK/graph_regularized_SVD/circVars.csv', index=False)

print('finish')
# %%
