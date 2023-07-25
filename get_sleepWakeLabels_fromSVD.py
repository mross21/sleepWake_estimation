#%%
import re
import glob
import pandas as pd
from scipy.sparse import csgraph
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from skimage import segmentation
import matplotlib as mpl

# sort files numerically
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return(parts)

def closest_hour(lst, K):
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

def day_weight(d1,d2):
    return (d1+d2)/2

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

def sliding_window(elements, window_size, gap):
    if len(elements) <= window_size:
       return elements
    windows = []
    ls = np.arange(0, len(elements), gap)
    for i in ls:
        windows.append(elements[i:i+window_size])
    return windows

############################################################################################
pathIn = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/keypress/'
pathOut = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/graph_regularized_SVD/matrices/'

# list of user accel files
all_files = sorted(glob.glob(pathIn + "*.csv"), key = numericalSort)

for file in all_files:
    df = pd.read_csv(file, index_col=False)
    user = int(df['userID'].unique())
    print('user: {}'.format(user))

    # get matrix of typing activity by day and hour
    df['hour'] = pd.to_datetime(df['keypressTimestampLocal']).dt.hour
    M = df.groupby(['dayNumber','hour'],as_index = False).size().pivot('dayNumber','hour').fillna(0)

    # insert hours with no activity across all days
    missingHours = [h for h in range(24) if h not in list(M['size'].columns)]
    M.columns = M.columns.droplevel(0)
    for h in missingHours:
        M.insert(h,h,[0]*M.shape[0])
    M = M.sort_index(ascending=True)

    # find avg number of hours of activity/day
    Mbinary = np.where(M > 0, 1, 0)
    avgActivityPerDay = Mbinary.mean(axis=1).mean()

    # of the days with kp, find median amount
    Mkp = np.where(M > 0, M, np.nan)
    avgAmountPerDay = np.nanmedian(np.nanmedian(Mkp, axis=1))

    # if not enough typing activity for user, skip user
    if (avgActivityPerDay < 0.2) | (avgAmountPerDay < 50):
        print('not enough data')
        continue

    # remove first and last days
    M1 = M[1:-1]

    # if less than 7 days, continue
    if M1.shape[0] < 7:
        print('not enough days')
        continue

    # remove 7-day segments of data if not enough
    slidingWindowListForRemoval = sliding_window(M1, window_size=7, gap=1)
    daysToRemove = []
    c = 0
    for w in slidingWindowListForRemoval:
        if len(w) < 7:
            break
        Wbinary = np.where(w > 0, 1, 0)
        avgActivity = Wbinary.mean(axis=1).mean()
        Wkp = np.where(w > 0, w, np.nan)
        avgAmount = np.nanmedian(np.nanmedian(Wkp, axis=1))
        if (avgActivity < 0.2) | (avgAmount < 50):
            daysToRemove.extend(list(range(c, c + 7)))
        c += 1
    # remove rows corresponding to indices in daysToRemove
    M1 = M1[~M1.index.isin([*set(daysToRemove)])]

    # if less than 7 days, continue
    if M1.shape[0] < 7:
        print('not enough days')
        continue

    # number of days and hours
    n_days = M1.shape[0]
    n_hrs = M1.shape[1]

    # get adjacency matrix for SVD
    W = weighted_adjacency_matrix(np.array(M1))
    # normalize keypress values
    normKP = (np.array(M1).flatten())/M1.sum().sum()
    # get graph laplacian
    B = csgraph.laplacian(W)
    # get graph normalized SVD
    H_star, W_star = regularized_svd(normKP, B, rank=1, alpha=50, as_sparse=False)
    # get SVD matrix
    svdM = (W_star * -1).reshape(M1.shape)
    # binarize SVD
    binarizedSVD = np.where(svdM == 0, 0,1)
    # sleep/wake labels from binarized SVD matrix
    sleep_label = 0
    wake_label = 1

    # flood fill main sleep component
    # initiate matrix filled with value 2 (meaning no label yet)
    floodFillM = np.full(shape=svdM.shape, fill_value=2, dtype='int')
    for r in range(svdM.shape[0]):
        # get row
        row = binarizedSVD[r]
        # if entire row is sleep, continue
        if ((row == np.array([sleep_label]*len(row))).all()) == True:
            floodFillM[r] = [sleep_label]*len(row)
            continue
        # if wake labels only during hours 2-15, get min from 0-24 hr range
        if ((binarizedSVD[r, 2:15] == np.array([wake_label]*13)).all()) == True:
            idx_rowMin = np.argmin(svdM[r])
        else: # else limit min hour to between hr 2-15
            idx_rowMin = np.argmin(svdM[r, 2:15]) + 2
        # if min value not equal to sleep_label, then no sleep that day. continue
        if binarizedSVD[r,idx_rowMin] != sleep_label:
            floodFillM[r] = [wake_label]*len(row)
            continue
        # flood fill 
        sleep_flood = segmentation.flood(binarizedSVD, (r,idx_rowMin))#NEED DIAG CONNECTIVITY #connectivity=1)
        # replace output matrix row with flood fill values
        floodFillM[r] = np.invert(sleep_flood[r])

        # add sleep label before midnight if exists
        # if iteration at last row
        if r == svdM.shape[0]-1:
            # if last cell is sleep label
            if (row[-1] == sleep_label):
                i = 0
                # find earliest index of sleep label for that row prior to midnight
                while row[23-i] == sleep_label:
                    end_idx = i
                    i += 1
                # replace identified ending cells with sleep label
                floodFillM[r,(23-end_idx):] = sleep_label
        # if interation not at last row
        else:
            # if last cell is sleep label and first cell of next row is sleep label
            if (row[-1] == sleep_label) & (binarizedSVD[r+1,0] == sleep_label):
                i = 0
                # find earliest index of sleep label for that row prior to midnight
                while row[23-i] == sleep_label:
                    end_idx = i
                    i += 1
                # replace identified ending cells with sleep label
                floodFillM[r,(23-end_idx):] = sleep_label

    # fill in gaps in sleep component
    # get list of sliding windows to remove gaps in sleep
    window_size = 6
    hr_space = 1
    floodFillFlatten = floodFillM.flatten()
    slidingWindowList = sliding_window(floodFillFlatten, window_size, hr_space)

    # iterate through windows and fill gap between first and last sleep index in window
    windowCount = 0
    for window in slidingWindowList:
        try:
            # get index of first sleep label in window
            firstSleep_idx = next(i for i,v in enumerate(window) if v == sleep_label)
        # if no sleep label within window, continue
        except StopIteration:
            windowCount += 1
            continue
        # get index of last sleep label in window
        lastSleep_idx = len(window) - next(i for i, val in enumerate(reversed(window), 1) if val != wake_label)
        # get index of first label in window from array of all labels
        grpIdx0 = windowCount * hr_space
        # replace window values with sleep labels within identified indices
        floodFillFlatten[(grpIdx0+firstSleep_idx):(grpIdx0+lastSleep_idx)] = [sleep_label]*(lastSleep_idx-firstSleep_idx)
        windowCount += 1
    # reshape array to be matrix of sleep/wake activity
    sleepWakeMatrix = floodFillFlatten.reshape(binarizedSVD.shape)
      

    #############################################################
    # Visualize heatmap of steps
    f, ax = plt.subplots(nrows=2,ncols=2, sharex=False, sharey=True,
                        figsize=(10,10))
    # PLOT 1
    sns.heatmap(M1, cmap='viridis', ax=ax[0,0], vmin=0, vmax=500,
                cbar_kws={'label': '# keypresses', 'fraction': 0.043})
    # PLOT 2
    sns.heatmap(svdM, cmap='viridis', ax=ax[0,1], vmin=0, vmax=0.002,
                cbar_kws={'label': '# keypresses', 'fraction': 0.043})
    # PLOT 3
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom',
        colors=['#de8f05', '#0173b2'],
        N=2)
    sns.heatmap(binarizedSVD, ax=ax[1,0], cmap=cmap,
                cbar_kws={'fraction': 0.043})
    colorbar = ax[1,0].collections[0].colorbar
    colorbar.set_ticks([0.25, 0.75])
    colorbar.set_ticklabels(['0', '1'])
    colorbar.set_label('Cluster')
    # PLOT 4
    sns.heatmap(sleepWakeMatrix, ax=ax[1,1], cmap=cmap,
                cbar_kws={'fraction': 0.043})
    colorbar = ax[0,1].collections[0].colorbar
    colorbar.set_ticks([0.25, 0.75])
    colorbar.set_ticklabels(['0', '1'])
    colorbar.set_label('Cluster')
    ax[0,0].set(title='Original', xlabel='Hour', ylabel='Day')
    ax[0,1].set(title='Graph Reg. SVD', xlabel='Hour', ylabel='Day')
    ax[1,0].set(title='K-Means of Graph Reg. SVD', xlabel='Hour', ylabel='Day')
    ax[1,1].set(title='Flood Fill of K-Means', xlabel='Hour', ylabel='Day')
    f.tight_layout()
    plt.show(f)
    # f.savefig(pathOut+'/HRxDAYsizeMat/sleepWakeLabels/user_{}_SVD_kmeans-lowKPDaysSkipped-6hr1hrSlidingWindow.png'.format(user))
    plt.close(f)

print('finish')
