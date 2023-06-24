#%%
import re
import glob
import pandas as pd
from scipy.sparse import csgraph
import matplotlib.pyplot as plt
import numpy as np
# from scipy.stats import f_oneway
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib as mpl
# import copy
# from sklearn.preprocessing import normalize
# from sklearn.preprocessing import StandardScaler
# from chronobiology.chronobiology import CycleAnalyzer
# from itertools import groupby


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
    return (d1+d2) #1/(abs(d1-d2)+.0000001)

def hour_weight(h1,h2):
    return (h1+h2)/2 #1/(abs(h1-h2)+.0000001)

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

def day_weight_SVD(d1,d2):
    return 1/(abs(d1-d2)+.0000001)

def hour_weight_SVD(h1,h2):
    return 1/(abs(h1-h2)+.0000001)

def weighted_adjacency_SVD_matrix(mat):
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
                W[i,j] = hour_weight_SVD(mat[i_Mi,i_Mj],mat[j_Mi,j_Mj])
            # if abs(subtraction of row indices) == 1 & subtraction of col indices == 0:
            elif (abs(j_Mi-i_Mi) == 1) & ((j_Mj-i_Mj) == 0):
                W[i,j] = day_weight_SVD(mat[i_Mi,i_Mj],mat[j_Mi,j_Mj])
            # connect 23hr with 00hr
            elif (i_Mj == mat.shape[1]-1) & ((j_Mi-i_Mi) == 1) & (j_Mj == 0):
                W[i,j] = hour_weight_SVD(mat[i_Mi,i_Mj],mat[i_Mi+1,0])
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

def simple_threshold(im, threshold):
    return ((im > threshold) * 255).astype("uint8")

# def sliding_window(elements, window_size, hr_gap):
#     if len(elements) <= window_size:
#        return elements
#     windows = []
#     ls = np.arange(0, len(elements), hr_gap)
#     for i in ls:
#         windows.append(elements[i:i+window_size])
#     return windows

# # # A recursive function to replace
# # # previous color 'prevC' at '(x, y)'
# # # and all surrounding pixels of (x, y)
# # # with new color 'newC' and
# # def floodFillUtil(screen, x, y, prevC, newC, M, N):
     
# #     # Base cases
# #     if (x < 0 or x >= M or y < 0 or
# #         y >= N or screen[x][y] != prevC or
# #         screen[x][y] == newC):
# #         return
 
# #     # Replace the color at (x, y)
# #     screen[x][y] = newC
 
# #     # Recur for north, east, south and west
# #     floodFillUtil(screen, x + 1, y, prevC, newC)
# #     floodFillUtil(screen, x - 1, y, prevC, newC)
# #     floodFillUtil(screen, x, y + 1, prevC, newC)
# #     floodFillUtil(screen, x, y - 1, prevC, newC)
 
# # # It mainly finds the previous color on (x, y) and
# # # calls floodFillUtil()
# # def floodFill(screen, x, y, newC, M, N):
# #     prevC = screen[x][y]
# #     if(prevC==newC):
# #       return
# #     floodFillUtil(screen, x, y, prevC, newC, M, N)

# # Size of given matrix is M x N
# M = 6
# N = 6
 
# # A recursive function to replace previous
# # value 'prevV' at '(x, y)' and all surrounding
# # values of (x, y) with new value 'newV'.
# def floodFillUtil(mat, x, y, prevV, newV):
 
#     # Base Cases
#     if (x < 0 or x >= M or y < 0 or y >= N):
#         return
 
#     if (mat[x][y] != prevV):
#         return
 
#     # Replace the color at (x, y)
#     mat[x][y] = newV
 
#     # Recur for north, east, south and west
#     floodFillUtil(mat, x + 1, y, prevV, newV)
#     floodFillUtil(mat, x - 1, y, prevV, newV)
#     floodFillUtil(mat, x, y + 1, prevV, newV)
#     floodFillUtil(mat, x, y - 1, prevV, newV)
 
# # Returns size of maximum size subsquare
# #  matrix surrounded by 'X'
# def replaceSurrounded(mat, clusterM, M, N):
 
#     # Step 1: Replace all '1's with '-'
#     for i in range(M):
#         for j in range(N):
#             if (mat[i][j] == '1'):
#                 clusterM[i][j] = '-'
 
#     # Call floodFill for all '-'
#     # lying on edges
#     # Left Side
#     for i in range(M):
#         if (mat[i][0] == '-'):
#             floodFillUtil(clusterM, i, 0, '-', 'O')
     
#     # Right side
#     for i in range(M):
#         if (mat[i][N - 1] == '-'):
#             floodFillUtil(clusterM, i, N - 1, '-', 'O')
     
#     # Top side
#     for i in range(N):
#         if (mat[0][i] == '-'):
#             floodFillUtil(clusterM, 0, i, '-', 'O')
     
#     # Bottom side
#     for i in range(N):
#         if (mat[M - 1][i] == '-'):
#             floodFillUtil(clusterM, M - 1, i, '-', 'O')
 
#     # Step 3: Replace all '-' with 'X'
#     for i in range(M):
#         for j in range(N):
#             if (mat[i][j] == '-'):
#                 clusterM[i][j] = 'X'

############################################################################################
pathIn = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/keypress/'
pathOut = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/graph_regularized_SVD/matrices/'
fileDiag = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/diagnosis/unmasck_demographics.csv'
dfDiag = pd.read_csv(fileDiag, index_col=False)
dictDiag = dict(zip(dfDiag['healthCode'], dfDiag['diagnosis']))

# list of user accel files
all_files = sorted(glob.glob(pathIn + "*.csv"), key = numericalSort)

for file in all_files:
    df = pd.read_csv(file, index_col=False)
    user = int(df['userID'].unique())
    print('user: {}'.format(user))

    if user != 11:
        continue

    df['healthCode'] = df['healthCode'].str.lower()
    df['diagnosis'] = df['healthCode'].map(dictDiag)
    diag = df['diagnosis'].iloc[0]

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
    # M1.drop([1,M1.shape[0]], axis=0, inplace=True)
    M1 = M1[1:-1]

    # if less than 7 days, continue
    if M1.shape[0] < 7:
        print('not enough days')
        continue

    # # remove beginning and end days if all 0 kp activity
    # M1=M1.replace(0, np.nan)
    # firstIdx = M1.first_valid_index()
    # firstToDrop = list(range(0,firstIdx))
    # try:
    #     M1.drop(firstToDrop, axis=0, inplace=True)
    # except KeyError:
    #     pass
    # lastIdx = M1.last_valid_index()
    # lastToDrop = list(range(lastIdx+1,M1.index[-1]+1))
    # M1.drop(lastToDrop, axis=0, inplace=True)
    # M1=M1.replace(np.nan, 0)

# ###############################
#     # insert days with no activity across all hours
#     missingDays = [d for d in range(1,df['dayNumber'].max()) if d not in list(M1.index)]
#     # M1.columns = M1.columns.droplevel(0)
#     for d in missingDays:
#         M1.loc[d] = [0]*M1.shape[1] # to add row
#         # M1.insert(d,d,[0]*M1.shape[1])
#     M1 = M1.sort_index(ascending=True)
# ####################################

    # # incorporate typing speed
    # Mspeed=df.groupby(['dayNumber','hour'],as_index = False).apply(lambda x: medianAAIKD(x)).pivot('dayNumber','hour')
    # Mspeed.columns = Mspeed.columns.droplevel(0)
    # for h in missingHours:
    #     # M.loc[h] = [0]*M.shape[1] # to add row
    #     Mspeed.insert(h,h,[np.nan]*Mspeed.shape[0])
    # Mspeed = Mspeed.sort_index(ascending=True)


    # medMax = M.max().median()
    # if medMax < 300:
    #     print('not enough kp per day')
    #     continue






    # # LOG TRANSFORM KP
    # M2 = np.log(M1+1)
# then look at filtering based on activity per week

## not enough contrast between no kp activity and kp activity after normalization





    # f, ax = plt.subplots(nrows=1,ncols=2, sharex=False, sharey=True,
    #                     figsize=(12, 6))
    # # PLOT 1
    # sns.heatmap(M1, cmap='viridis', ax=ax[0], vmin=0, vmax=500,
    #             cbar_kws={'label': '# keypresses', 'fraction': 0.043})
    # sns.heatmap(Mspeed, cmap='viridis', ax=ax[1],
    #             cbar_kws={'label': '# keypresses', 'fraction': 0.043})
    # plt.show()
    # plt.clf()




    # # remove rows with less than 200 kp/day
    # M = M1[M1.sum(axis=1) > 200]

    # # create a scaler object
    # std_scaler = StandardScaler()
    # # fit and transform the data
    # M2=M1.T
    # M2_scaled = pd.DataFrame(std_scaler.fit_transform(M2), columns=M2.columns)
    # M = M2_scaled.T #M1.div(M1.sum(axis=1), axis=0) #normalize(M1, axis=1, norm='l1')

    # M = M1/M1.sum().sum()

# # remove weeks with not enough data
# # this doesn't make sense since large amounts of typing one hour out of
# #   the week would skew the entire filtering
#     M1['weekNumber']=np.arange(len(M1))//7
#     weekSums = M1.groupby('weekNumber').sum().sum(axis=1)
#     keepWeeks = list(weekSums.loc[weekSums>10000].index)
#     M = M1.loc[M1['weekNumber'].isin(keepWeeks)].drop(['weekNumber'], axis=1)

    # M = copy.deepcopy(M1)
    ###########################################################################
    # ADJACENCY MATRIX OF SIZE (DAYS X HRS) x (DAYS X HRS)
    # days = rows
    # hours = columns

    # M = np.array(M)
    n_days = M1.shape[0]
    n_hrs = M1.shape[1]


    # if M.mean() <= 100:
    #     print('mean KP too low')
    #     continue

    # hard code adjacency matrix
    W = weighted_adjacency_matrix(np.array(M1))

    kp_values = np.array(M1).flatten()
    days_arr = np.repeat(range(n_days), n_hrs)
    hrs_arr = np.array(list(range(n_hrs)) * n_days)
    data = np.vstack((days_arr,hrs_arr, kp_values))
    B = csgraph.laplacian(W)
    H_star, W_star = regularized_svd(data, B, rank=1, alpha=0.1, as_sparse=False)

    out2 = W_star.reshape(M1.shape)
    out2 = out2 * -1
    clip_amount = out2.max()/10
    cutoff = np.clip(out2, 0, clip_amount)

    # X = cutoff

    # # Reshape data into observations x features
    # # Columns (features): [day, hour, keypresses]
    # # Rows (observations) of size (days*hours)
    # dfM = pd.DataFrame(X.T)
    # dfM = dfM.melt(var_name='day', value_name='keypresses')
    # # dfM['day'] = (pd.Series(np.arange(n_days))
    # #             .repeat(n_hours).reset_index(drop=True))
    # dfM['hour'] = list(range(24))*n_days
    # dfM = dfM[['day', 'hour', 'keypresses']]  # rearrange columns

    # # Something simple for first approach: PCA
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(dfM.to_numpy())
    dfWstar = pd.DataFrame(cutoff.reshape(-1,1), columns = ['vals'])
    kmeans = KMeans(n_clusters=2, random_state=123).fit(dfWstar)
    dfKmeans = pd.DataFrame({
        # 'pca_x': X_pca[:, 0],
        # 'pca_y': X_pca[:, 1],
        'graph_reg_SVD_vals': dfWstar['vals'],
        'cluster': kmeans.labels_})

    dfPCA = dfKmeans

    # # Visualize k-means clusters in PCA embedding
    # f, ax = plt.subplots()
    # sns.scatterplot(data=dfKmeans, x='pca_x', y='pca_y', hue='cluster', ax=ax)
    # plt.show()
    # plt.clf()


    # n_bins = 4
    # b,bins,patches = plt.hist(x=dfPCA['pca_x'], bins=n_bins)
    # dfPCA['cluster_filt'] = np.where((dfPCA['pca_x'] <= bins[1]) | (dfPCA['pca_x'] >= bins[len(bins)-2]), dfPCA['cluster'],np.nan)
    # # dfPCA.loc[(dfPCA['pca_x'] <= bins[1]) | (dfPCA['pca_x'] >= bins[len(bins)-2])]

    # # just plot of one matrix M
    # f, ax = plt.subplots(nrows=1,ncols=1, sharex=False, sharey=True, figsize=(8,10))
    # sns.heatmap(M, cmap='viridis', vmin=0, vmax=500, cbar_kws{'label': '# keypresses', 'fraction': 0.043})


###########################################################
#   # replace islands with surrounding val


    # # make dataframe of day|hour|nKP|timestamp
    # dfActivity = pd.DataFrame(data.T, columns = ['day','hour','nKP'])
    # date2 = pd.to_datetime(df['date']).drop_duplicates().nsmallest(2).max()
    # start_date = np.datetime64(date2, 'h')
    # dfActivity['timestamp'] = start_date + days_arr.astype('timedelta64[D]') + hrs_arr.astype('timedelta64[h]')
    # dfActivity['cluster'] = dfPCA['cluster']
    # dateEnd = pd.to_datetime(df['date']).drop_duplicates().nlargest(2).iloc[-1]
    # end_date = np.datetime64(dateEnd, 'h') + np.timedelta64(23,'h')


    # cluster_mat = dfKmeans['cluster'].to_numpy().reshape(M1.shape)
    # plt.imshow(cluster_mat)
    # plt.show()



    # # # make new var to update rows
    # # cluster_mat2 = copy.deepcopy(cluster_mat)

    # # # make df of labels by day and hr
    # # dfLabels = pd.DataFrame(cluster_mat2.T).melt(var_name='day', value_name='cluster')
    # # dfLabels['day'] = (pd.Series(np.arange(n_days))
    # #             .repeat(n_hours).reset_index(drop=True))+1
    # # dfLabels['hour'] = list(range(24))*n_days
    # # dfLabels = dfLabels[['day', 'hour', 'cluster']]

    # wake_label_idx = cluster_mat[0].argmax()
    # wake_label = cluster_mat[0,wake_label_idx]
    # sleep_label_idx = cluster_mat[0].argmin()
    # sleep_label = cluster_mat[0,sleep_label_idx]

    # if wake_label == sleep_label:
    #     print('sleep and wake labels are the same')
    #     break

# #########################################
#     # get median wake time
#     # makes assumption that min hour is wake up (and not night schedule)
#     wake_time = dfActivity.loc[dfActivity['cluster'] == wake_label]
#     median_wake_hour = round(wake_time.groupby(['day'])['hour'].min().median(),0)

#     dfActivity['cluster_change_flag'] = abs(dfActivity['cluster'].diff()).replace(float('NaN'),0)
#     for obs in range(len(dfActivity)):
#         # print(obs)
#         if dfActivity['cluster_change_flag'].iloc[obs] == 1:
#             # get neighboring rows
#             neighbors = dfActivity.iloc[int(np.where((obs-1) < 0, 0, (obs-1))) :
#                                     int(np.where((obs+2) > len(dfActivity), len(dfActivity), (obs+2)))]
#             if sum(neighbors['cluster_change_flag']) > 1:
#                 # print(neighbors)
#                 # if one cluster label diff from all others
#                 surroundingLabel = dfActivity['cluster'].iloc[int(np.where((obs-2) < 0,0,(obs-2))):
#                                     int(np.where((obs+2) > len(dfActivity), len(dfActivity), 
#                                     (obs+3)))].value_counts().index[0]
#                 dfActivity['cluster'].iloc[obs] = surroundingLabel
#                 # if dfActivity['hour'].iloc[obs] >= median_wake_hour:
#                 #     dfActivity['cluster'].iloc[obs] = wake_label
#                 # else:
#                 #     dfActivity['cluster'].iloc[obs] = sleep_label
#                 # print('new label: {}'.format(dfActivity['cluster'].iloc[obs]))
#         # recalculate all change cluster labels
#         dfActivity['cluster_change_flag'] = abs(dfActivity['cluster'].diff()).replace(float('NaN'),0)
# # #########################################


# #### need to remove islands

# # Python3 program to implement
# # flood fill algorithm
#     dfActivity['cluster_change_flag'] = abs(dfActivity['cluster'].diff()).replace(float('NaN'),0)
#     for obs in range(len(dfActivity)):
#         # print(obs)
#         if dfActivity['cluster_change_flag'].iloc[obs] == 1:
#             # get neighboring rows
#             neighbors = dfActivity.iloc[int(np.where((obs-1) < 0, 0, (obs-1))) :
#                                     int(np.where((obs+2) > len(dfActivity), len(dfActivity), (obs+2)))]
#             if sum(neighbors['cluster_change_flag']) > 1:
#                 # print(neighbors)
#                 # if one cluster label diff from all others


#                 surroundingLabel = dfActivity['cluster'].iloc[int(np.where((obs-2) < 0,0,(obs-2))):
#                                     int(np.where((obs+2) > len(dfActivity), len(dfActivity), 
#                                     (obs+3)))].value_counts().index[0]
#                 dfActivity['cluster'].iloc[obs] = surroundingLabel
#                 # if dfActivity['hour'].iloc[obs] >= median_wake_hour:
#                 #     dfActivity['cluster'].iloc[obs] = wake_label
#                 # else:
#                 #     dfActivity['cluster'].iloc[obs] = sleep_label
#                 # print('new label: {}'.format(dfActivity['cluster'].iloc[obs]))
#         # recalculate all change cluster labels
#         dfActivity['cluster_change_flag'] = abs(dfActivity['cluster'].diff()).replace(float('NaN'),0)
# # #########################################



#     MchangeFlag = dfActivity['cluster_change_flag'].to_numpy().reshape(M1.shape)
#     for i in range(n_days):
#         for j in range(n_hrs):
#             if MchangeFlag[i][j] == 1:
#                 print(i,j)
#                 print(cluster_mat[i][j])



# # x = 4
# # y = 4
# # newC = 3
# # floodFill(screen, x, y, newC, n_days, n_hrs)

#     break









#  #  # loop through sliding window of approx 30 hrs to search for largest blocks of 0/1
#     window_size = 30 #int(len(dfActivity)/28)
#     hr_space = 8
#     # windowList = np.array_split(dfActivity['timestamp'], window_size)
#     slidingWindowList = sliding_window(dfActivity['timestamp'], window_size, hr_space)

#     dfConsecClusters = pd.DataFrame()
#     for window in slidingWindowList:
#         windowGrp = dfActivity.loc[dfActivity['timestamp'].isin(window.reset_index(drop=True))].reset_index(drop=True)
#         ranges=[list((v,list(g))) for v,g in groupby(range(len(windowGrp)),lambda idx:windowGrp['cluster'][idx])]
#         dfIdx = pd.DataFrame(ranges, columns = ['cluster','idx'])
#         try:
#             max0IdxList = max(dfIdx.loc[dfIdx['cluster'] == 0]['idx'], key=len)
#         except ValueError:
#             max0IdxList = []
#         try:
#             max1IdxList = max(dfIdx.loc[dfIdx['cluster'] == 1]['idx'], key=len)
#         except ValueError:
#             max1IdxList = []
#         # get the df info for indices
#         cluster0 = windowGrp.iloc[max0IdxList]
#         cluster1 = windowGrp.iloc[max1IdxList]
#         dfConsecClusters = dfConsecClusters.append((cluster0,cluster1))


#         # if len(dfConsecClusters) > 80:
#         #     break
#     # break

#     dfConsecClusters = dfConsecClusters.sort_values(by='timestamp').drop_duplicates(subset='timestamp')
#     dictClusterLabels = dict(zip(dfConsecClusters['timestamp'], dfConsecClusters['cluster']))
#     dfConsecClustersFilled = pd.DataFrame(dfActivity['timestamp'], columns = ['timestamp'])
#     dfConsecClustersFilled['cluster'] = dfConsecClustersFilled['timestamp'].map(dictClusterLabels)
#     dfConsecClustersFilled['day'] = dfActivity['day']
#     dfConsecClustersFilled['hour'] = dfActivity['hour']
#     # dfConsecClustersFilled['cluster2'] = dfConsecClustersFilled.groupby('day').apply(lambda x: x.loc['cluster'])
#     # dfConsecClustersFilled['cluster_bfill'] = dfConsecClustersFilled['cluster'].bfill().ffill()


#############################################################
    # # Visualize original data heatmap and heatmap with k-means cluster labels
    # f, ax = plt.subplots(nrows=2,ncols=2, sharex=False, sharey=True,
    #                     figsize=(10,10))
    # # PLOT 1
    # sns.heatmap(M1, cmap='viridis', ax=ax[0,0], vmin=0, vmax=500,
    #             cbar_kws={'label': '# keypresses', 'fraction': 0.043})
    # # PLOT 2
    # sns.heatmap(out2, cmap='viridis', ax=ax[0,1], vmin=0, vmax=200,
    #             cbar_kws={'label': '# keypresses', 'fraction': 0.043})
    # # # PLOT 3
    # # sns.heatmap(cutoff, cmap='viridis', ax=ax[1,0], #vmin=0, vmax=clip_amount,
    # #             cbar_kws={'label': '# keypresses', 'fraction': 0.043})
    
    # # # PLOT 2
    # # cluster_mat = dfPCA['cluster'].to_numpy().reshape(X.shape)
    # # cmap = mpl.colors.LinearSegmentedColormap.from_list(
    # #     'Custom',
    # #     colors=['#de8f05', '#0173b2'],
    # #     N=2)
    # # sns.heatmap(cluster_mat, ax=ax[0,1], cmap=cmap,
    # #             cbar_kws={'fraction': 0.043})
    # # colorbar = ax[0,1].collections[0].colorbar
    # # colorbar.set_ticks([0.25, 0.75])
    # # colorbar.set_ticklabels(['0', '1'])
    # # colorbar.set_label('Cluster')

    # # PLOT 3
    # # cluster_mat = dfActivity['cluster'].to_numpy().reshape(X.shape)
    # cluster_mat = dfKmeans['cluster'].to_numpy().reshape(M1.shape)
    # cmap = mpl.colors.LinearSegmentedColormap.from_list(
    #     'Custom',
    #     colors=['#de8f05', '#0173b2'],
    #     N=2)
    # sns.heatmap(cluster_mat, ax=ax[1,0], cmap=cmap,
    #             cbar_kws={'fraction': 0.043})
    # colorbar = ax[1,0].collections[0].colorbar
    # colorbar.set_ticks([0.25, 0.75])
    # colorbar.set_ticklabels(['0', '1'])
    # colorbar.set_label('Cluster')

    # # PLOT 4
    # # consecClusters=dfConsecClustersFilled['cluster'].to_numpy().reshape(M1.shape)
    # # cmap = mpl.colors.LinearSegmentedColormap.from_list(
    # #     'Custom', colors=['#de8f05', '#0173b2'], N=2)
    # # sns.heatmap(consecClusters, ax=ax[1,1], cmap=cmap,
    # #             cbar_kws={'fraction': 0.043})    
    # # colorbar = ax[1,1].collections[0].colorbar
    # # colorbar.set_ticks([0.25, 0.75])
    # # colorbar.set_ticklabels(['0', '1'])
    # # colorbar.set_label('Cluster')


    # ax[0,0].set(title='Original', xlabel='Hour', ylabel='Day')
    # ax[0,1].set(title='Graph Reg. SVD', xlabel='Hour', ylabel='Day')
    # # ax[1,0].set(title='Truncated Graph Reg. SVD', xlabel='Hour', ylabel='Day')
    # ax[1,0].set(title='K-Means Clustering from PCA', xlabel='Hour', ylabel='Day')
    # ax[1,1].set(title='Filtered K-Means Clustering from PCA', xlabel='Hour', ylabel='Day')
    # f.tight_layout()
    # plt.show(f)
    # # f.savefig(pathOut+'HRxDAYsizeMat/largestComponent/user_{}_svd_PCA-kmeans.png'.format(user))
    # plt.close(f)
##############################################################

    # # if user >= 22:
    # #     break

    # break

    # # Visualize original data heatmap and heatmap with k-means cluster labels
    # f, ax = plt.subplots(nrows=1,ncols=2, sharex=False, sharey=True,
    #                     figsize=(10,5))
    # # PLOT 1
    # sns.heatmap(M1, cmap='viridis', ax=ax[0], vmin=0, vmax=500,
    #             cbar_kws={'label': '# keypresses', 'fraction': 0.043})
    # # PLOT 2
    # sns.heatmap(out2, cmap='viridis', ax=ax[1], vmin=0, vmax=200,
    #             cbar_kws={'label': '# keypresses', 'fraction': 0.043})

    # ax[0].set(title='Original', xlabel='Hour', ylabel='Day')
    # ax[1].set(title='Graph Reg. SVD', xlabel='Hour', ylabel='Day')
    # f.tight_layout()
    # # plt.show(f)
    # # f.savefig(pathOut+'HRxDAYsizeMat/SVD/user_{}_graphRegSVD.png'.format(user))
    # plt.close(f)


    # # calculate regularity
    # diff1 = get_regularity(out2, 1)
    # diff2 = get_regularity(out2, 2)
    # diff3 = get_regularity(out2, 3)
    # diff4 = get_regularity(out2, 4)
    # diff5 = get_regularity(out2, 5)
    # diff6 = get_regularity(out2, 6)
    # diff7 = get_regularity(out2, 7)
    # dfRegularity = pd.DataFrame([diff1,diff2,diff3,diff4,
    #                             diff5,diff6,diff7]).T
    # dfRegularity.columns = [1,2,3,4,5,6,7]
    
    # fig, axes = plt.subplots(figsize=(5,5))
    # sns.set(style="whitegrid")
    # sns.boxplot(data=dfRegularity, ax = axes, orient ='v').set(title = 'User {} Regularity (Diagnosis: {})'.format(user, diag))
    # plt.xlabel('Days Apart')
    # plt.ylabel('Cosine Similarity')
    # plt.ylim([0, 1])
    # plt.show()
    # # print(diag)
    # # if diag == 'HC':
    # #     plt.savefig(pathOut + 'HRxDAYsizeMat/regularity/HC/user_{}_regularity.png'.format(user))
    # # elif diag == 'MD':
    # #     plt.savefig(pathOut + 'HRxDAYsizeMat/regularity/MD/user_{}_regularity.png'.format(user))
    # # else:
    # #     plt.savefig(pathOut + 'HRxDAYsizeMat/regularity/nan/user_{}_regularity.png'.format(user))
    # plt.clf()


    # polar_ls = []
    # for d in range(n_days):
    #     r = d
    #     theta = np.linspace(0, 2*np.pi, 24) 
    #     polar_ls.append(([r]*24,theta,out2[d]))
    #     area = 200 * r**2
    #     colors = theta

    # dfPolar = pd.DataFrame(polar_ls, columns=['r','theta','intensity']).explode(['r','theta','intensity'])
    # fig, ax = plt.subplots(figsize=(8,5))
    # ax = plt.subplot(111, projection='polar')
    # ax.set_xticklabels(['0', '', '6', '', '12', '', '18', ''])
    # ax.set_theta_offset(np.pi/2)
    # ax.set_theta_direction(-1)
    # c=ax.scatter(dfPolar['theta'], dfPolar['r'], c=dfPolar['intensity'], s=5, 
    #             cmap='viridis', alpha=1, vmin=0, vmax=200)
    # plt.colorbar(c, ax=ax)
    # plt.show()
    # # plt.savefig(pathOut + 'HRxDAYsizeMat/polarPlots/user_{}_regularity.png'.format(user))
    # plt.clf()

# image threshold for image segmentation
    # p = out2.max()/4
    # p2 = out2.max()/3
    # p3 = out2.max()/5
    # p4 = out2.max()/6
    # p5 = out2.max()/10

    # thresholds = [p5,p4,p3,p,p2]

    # fig, axs = plt.subplots(nrows=1, ncols=len(thresholds), figsize=(20,5))
    # gray_im = out2
                            
    # for t, ax in zip(thresholds, axs):
    #     ax.imshow(simple_threshold(gray_im, t), cmap='Greys')
    #     ax.set_title("Threshold: {}".format(t), fontsize=20)
    #     ax.set_axis_off()
    # plt.show()

    # if user >=10:
    #     break

    break


#%%
# normalized cuts

import networkx as nx

adjSVD = weighted_adjacency_SVD_matrix(out2)
adjSVD_upper = np.triu(adjSVD, k=0)
adjSVD_lower = np.tril(adjSVD, k=0)
G = nx.from_numpy_matrix(np.array(adjSVD_upper), parallel_edges=False, 
                         create_using=nx.DiGraph())

nx.draw_kamada_kawai(G)

# # Interactive networkx plot
# from pyvis.network import Network
# net = Network(
#     directed = True #,
#     # select_menu = True, # Show part 1 in the plot (optional)
#     # filter_menu = True, # Show part 2 in the plot (optional)
# )
# net.show_buttons() # Show part 3 in the plot (optional)
# net.from_nx(G) # Create directly from nx graph
# net.show('test.html')

# # edge detection
# from scipy import ndimage
# x = ndimage.sobel(out2, axis=0, mode='constant')
# y = ndimage.sobel(out2, axis=1, mode='constant')
# Sobel = np.hypot(x, y)
# plt.imshow(Sobel, cmap='viridis')
# plt.show()


# indices of max and min
maxSVD = np.argmax(out2)
minSVD = np.argmin(out2)
# sink vertex
T = maxSVD
# source vertex
S = minSVD

cut_value, partition = nx.minimum_cut(G, S, T, capacity='weight')
reachable, non_reachable = partition




#%%
from skimage import future, filters, io

img = out2

# labels1 = segmentation.slic(img, compactness=30, n_segments=400)
labels = np.array(dfKmeans['cluster']).reshape(M1.shape)
edge_map = filters.sobel(labels)
labs2 = np.where(edge_map > 0, 1, 0)
# rag = future.graph.rag_boundary(labels, edge_map, connectivity=0)
# print(future.graph.cut_normalized(labels, rag))



# # out3 = color.label2rgb(labels2, img, kind='avg', bg_label=0)

# fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))

# ax[0].imshow(out1)
# ax[1].imshow(out3)

# for a in ax:
#     a.axis('off')

# plt.tight_layout()
# plt.show()

#%%

from skimage import data, segmentation, future
img = data.astronaut()
labels = segmentation.slic(img)
rag = future.graph.rag_mean_color(img, labels, mode='similarity')
new_labels = future.graph.cut_normalized(labels, rag)

#%%

from skimage import data
from skimage import filters
# camera = data.camera()
val = filters.threshold_otsu(out2)
mask = out2 < val
plt.imshow(mask)

#%%

# # Second smallest eigenvector from graph laplacian

adj_M = weighted_adjacency_SVD_matrix(out2)
L = csgraph.laplacian(adj_M)
eigen_values, eigen_vectors = np.linalg.eig(L)

idx = eigen_values.argsort()[::-1]   
eigenValues = np.real(eigen_values[idx])
eigenVectors = np.real(eigen_vectors[:,idx])

binEig2 = np.where(eigenVectors[-2] > 0, 1, -1)

plt.imshow(binEig2.reshape(out2.shape))


#%%


######################################################
# ## double plot

#     start_date = np.datetime64('2020-01-01')
#     ts = start_date + days_arr.astype('timedelta64[D]') + hrs_arr.astype('timedelta64[h]')

#     ca = CycleAnalyzer(timestamps=ts, activity=kp_values, night=np.array([False]*len(ts)),
#                        min_data_points=-1,max_gap='1h')

#     ca.plot_actogram(log=True, activity_onset=False, height=20)



## double plot
    # dfActivity = pd.DataFrame(data.T, columns = ['day','hour','nKP'])
    # dfActivity['activity_binary'] = np.where(dfActivity['nKP'] > 0, 1, 0)

    # dfKPActivity = dfActivity[['day','hour','activity_binary']]
    # maxDay = dfKPActivity['day'].max()
    # byDay = dfKPActivity.groupby('day')
    # duplicatedDays = pd.DataFrame()
    # for day, grp in byDay:
    #     if day == 0:
    #         duplicatedDays = duplicatedDays.append(grp)
    #     elif day == maxDay:
    #         duplicatedDays = duplicatedDays.append(grp)
    #     else:
    #         duplicatedDays = duplicatedDays.append(grp)
    #         duplicatedDays = duplicatedDays.append(grp)

    # row_len = int(len(duplicatedDays)/48)

    # doublePlotData = duplicatedDays['activity_binary'].to_numpy().reshape(row_len, 48)


    # # just plot of one matrix M
    # f, ax = plt.subplots(figsize=(10,5))
    # sns.heatmap(doublePlotData, cmap='viridis', vmin=0, vmax=1,
    #                 cbar_kws={'label': '# keypresses', 'fraction': 0.043})
    # ax.set(title='Double Plot', xlabel='Hour', ylabel='Day')

    # plt.show()
    # # pathDubPlot = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/graph_regularized_SVD/matrices/HRxDAYsizeMat/doublePlots/'
    # # plt.savefig(pathDubPlot + 'user_{}_binaryKPActivity.png'.format(user))
    # plt.clf()

print('finish')





###############################################################################




#%%
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


    cluster_mat = dfPCA['cluster'].to_numpy().reshape(X.shape)

    # make new var to update rows
    cluster_mat2 = copy.deepcopy(cluster_mat)

    # # for days with low KP totals, replace cluster labels with surrounding days' data
    # lowKP_threshold = np.quantile(M.sum(axis=1), 0.25) # 25th percentile of nKP/day
    # lowKP_days = np.where(M.sum(axis=1) < lowKP_threshold)[0].tolist()
    # for d in lowKP_days:
    #     if d < 2:
    #         # set row to average of next 3 rows
    #         cluster_mat2[d] = np.around((cluster_mat2[d+1] + cluster_mat2[d+2] + cluster_mat2[d+3])/3,1)
    #     else:
    #         # set row to average of prev 3 rows
    #         cluster_mat2[d] = np.around((cluster_mat2[d-1] + cluster_mat2[d-2] + cluster_mat2[d-3])/3,1)

# ## or should it be if there is low spread of kp, then replace w/ surrounding days
# ## or both
######### would need to make it the spread between first and last typing of day
#     lowTyping_threshold = 5 #int(np.quantile(np.sum(M > 10, axis=1), 0.1)) # 10th percentile of #hrs typing/day
#     lowTyping_days = np.where(M.sum(axis=1) < lowTyping_threshold)[0].tolist()
#     for d in lowTyping_days:
#         if d < 2:
#             # set row to average of next 3 rows
#             cluster_mat2[d] = np.around((cluster_mat2[d+1] + cluster_mat2[d+2] + cluster_mat2[d+3])/3,0)
#         else:
#             # set row to average of prev 3 rows
#             cluster_mat2[d] = np.around((cluster_mat2[d-1] + cluster_mat2[d-2] + cluster_mat2[d-3])/3,0)


    # make df of labels by day and hr
    dfLabels = pd.DataFrame(cluster_mat2.T).melt(var_name='day', value_name='cluster')
    dfLabels['day'] = (pd.Series(np.arange(n_days))
                .repeat(n_hours).reset_index(drop=True))+1
    dfLabels['hour'] = list(range(24))*n_days
    dfLabels = dfLabels[['day', 'hour', 'cluster']]

    # # fix cluster labels to be continuous for sleep or wake and remove isolated cluster labels
    # byDay = df.groupby(['dayNumber','hour'])
    # for dayHour, grp in byDay:
    #     print('day: {}'.format(dayHour[0]))
    #     print('hour: {}'.format(dayHour[1]))
    #     print(grp['sleepWake_cluster'])

    # need to identify largest break in time
    # since time is continuous, might not be able to group by day
    # might need to stitch all days/hours together and find the gaps that way


    # something like if label diff from X surrounding labels, change
    # label to what the surrounding labels are? for continuous list and
    # not the groupby day
###########################


    # get cluster label for sleep/wake
    # wake_label = dfLabels.loc[(dfLabels['hour'] > 12) & (dfLabels['hour'] < 15)]['cluster'].value_counts().idxmax()
    # sleep_label = dfLabels.loc[(dfLabels['hour'] > 1) & (dfLabels['hour'] < 4)]['cluster'].value_counts().idxmax()
    wake_label_idx = M[0].argmax()
    wake_label = cluster_mat[0,wake_label_idx]
    sleep_label_idx = M[0].argmin()
    sleep_label = cluster_mat[0,sleep_label_idx]

    if wake_label == sleep_label:
        break

    # get median wake time
    # makes assumption that min hour is wake up (and not night schedule)
    wake_time = dfLabels.loc[dfLabels['cluster'] == wake_label]
    median_wake_hour = round(wake_time.groupby(['day'])['hour'].min().median(),0)



    dfLabels['cluster_change_flag'] = abs(dfLabels['cluster'].diff()).replace(float('NaN'),0)
    for obs in range(len(dfLabels)):
        # print(obs)
        if dfLabels['cluster_change_flag'].iloc[obs] == 1:
            # get neighboring rows
            neighbors = dfLabels.iloc[int(np.where((obs-1) < 0, 0, (obs-1))) :
                                    int(np.where((obs+2) > len(dfLabels), len(dfLabels), (obs+2)))]
            if sum(neighbors['cluster_change_flag']) > 1:
                print(neighbors)
                # if one cluster label diff from all others
                if dfLabels['hour'].iloc[obs] >= median_wake_hour:
                    dfLabels['cluster'].iloc[obs] = wake_label
                else:
                    dfLabels['cluster'].iloc[obs] = sleep_label
                print('new label: {}'.format(dfLabels['cluster'].iloc[obs]))
        # recalculate all change cluster labels
        dfLabels['cluster_change_flag'] = abs(dfLabels['cluster'].diff()).replace(float('NaN'),0)


    # # locate first wake_label after sleep_labels
    # byDay = dfLabels.groupby('day')
    # for day, grp in byDay:
    #     # print(day)
    #     wake_transition = grp.loc[(grp['cluster_change_flag'] == 1) & (grp['cluster'] == wake_label)]
    #     if len(wake_transition) > 1:
    #         wake_hr = closest_hour(list(wake_transition['hour']), median_wake_hour)
    #         wake_transition = wake_transition.loc[wake_transition['hour'] == wake_hr]
    #     # print(wake_transition)



        # break






    # Visualize original data heatmap and heatmap with k-means cluster labels
    f, ax = plt.subplots(nrows=2,ncols=2, sharex=False, sharey=True,
                        figsize=(10,10))
    sns.heatmap(M, cmap='viridis', ax=ax[0,0], vmin=0, vmax=500,
                cbar_kws={'label': '# keypresses', 'fraction': 0.043})
    # Reshape k-means cluster labels from 726d vector to 22x33
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom',
        colors=['#de8f05', '#0173b2'],
        N=2)
    sns.heatmap(cluster_mat, ax=ax[0,1], cmap=cmap,
                cbar_kws={'fraction': 0.043})
    colorbar = ax[0,1].collections[0].colorbar
    colorbar.set_ticks([0.25, 0.75])
    colorbar.set_ticklabels(['0', '1'])




    sns.heatmap(cluster_mat2, ax=ax[1,0], cmap=cmap,
                cbar_kws={'fraction': 0.043})
    colorbar = ax[1,0].collections[0].colorbar
    colorbar.set_ticks([0.25, 0.75])
    colorbar.set_ticklabels(['0', '1'])
    colorbar.set_label('Cluster')



    cluster_mat3 = dfLabels['cluster'].to_numpy().reshape(X.shape)
    sns.heatmap(cluster_mat3, ax=ax[1,1], cmap=cmap,
                cbar_kws={'fraction': 0.043})
    colorbar = ax[1,1].collections[0].colorbar
    colorbar.set_ticks([0.25, 0.75])
    colorbar.set_ticklabels(['0', '1'])
    colorbar.set_label('Cluster')
    ax[0,0].set(title='Original', xlabel='Hour', ylabel='Day')
    ax[0,1].set(title='K-Means Clustering from PCA', xlabel='Hour', ylabel='Day')

    ax[1,0].set(title='Replace Low KP Days', xlabel='Hour', ylabel='Day')

    ax[1,1].set(title='Filled In K-Means Clustering', xlabel='Hour', ylabel='Day')
    f.tight_layout()
    plt.show(f)
    # f.savefig(pathOut+'HRxDAYsizeMat/filledIn/user_{}_svd_PCA-kmeans.png'.format(user))
    plt.close(f)


    if any(dfLabels.groupby('day').apply(lambda x: sum(x['cluster_change_flag'])) > 3):
        print('gaps in sleep-wake for user {}'.format(user))

        # this doesn't catch all gaps either
        # fix 3+ hr gaps








    if user == 10:
        break


    ## still need to fix the 3+ hr gaps in typing during the day
    # but otherwise seems ok for filling in gaps and low typing days?
    # for first few users

    # potentially, if wake up way later than normal, fix to day before?


### instead what if label all vals as wake_label after first wake_label detected
    # for the rest of the day...
    # instead of trying to fill in gaps for quick changes



## the next part for sleep wake labels
    # # map labels to the df at the end, since the df has many obs per hour
    # dictLabels = dfLabels.groupby('day')[['hour','cluster']].apply(lambda x: x.set_index('hour')
    #             .to_dict(orient='index')).to_dict()

    # df['sleepWake_label'] = df.apply(lambda x: dictLabels[x['dayNumber']][x['hour']]['cluster'], axis=1)





    print('==========================================')