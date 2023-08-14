#%%

from function_sleepWakeLabels import *

# FUNCTIONS TO GET SLEEP/WAKE LABELS BY HOUR FROM BIAFFECT KEYPRESS FILE

# sort files numerically
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return(parts)

# calculate typing speed (median IKD)
def medianAAIKD(dataframe):
    import numpy as np
    grpAA = dataframe.loc[((dataframe['keypress_type'] == 'alphanum') &
                                (dataframe['previousKeyType'] == 'alphanum'))]
    # get median IKD
    medAAIKD = np.nanmedian(grpAA['IKD']) if len(grpAA) >= 20 else float('NaN')
    return(medAAIKD)

# def get_typingMatrices(df):
#     """
#     Set up Biaffect typing activity and typing speed matrices for graph
#     regularized SVD.

#     Parameters
#     ----------
#     df : pandas dataframe
#          preprocessed BiAffect keypress file.

#     Returns
#     -------
#     activityM : pandas dataframe
#                 BiAffect typing activity by hour of shape (days x hours).
#     speedM : pandas dataframe
#              BiAffect typing speed by hour of shape (days x hours).
#     """
#     import numpy as np
#     import pandas as pd

#     # get matrix of typing activity by day and hour
#     df['hour'] = pd.to_datetime(df['keypressTimestampLocal']).dt.hour
#     M = df.groupby(['dayNumber','hour'],as_index = False).size().pivot('dayNumber','hour').fillna(0)

#     # insert hours with no activity across all days
#     missingHours = [h for h in range(24) if h not in list(M['size'].columns)]
#     M.columns = M.columns.droplevel(0)
#     for h in missingHours:
#         M.insert(h,h,[0]*M.shape[0])
#     M = M.sort_index(ascending=True)

#     # Filter users with not enough data
#     # find avg number of hours of activity/day
#     Mbinary = np.where(M > 0, 1, 0)
#     avgActivityPerDay = Mbinary.mean(axis=1).mean()
#     # of the days with kp, find median amount
#     Mkp = np.where(M > 0, M, np.nan)
#     avgAmountPerDay = np.nanmedian(np.nanmedian(Mkp, axis=1))
#     # if not enough typing activity for user, skip user
#     if (avgActivityPerDay < 0.2) | (avgAmountPerDay < 50):
#         return None, None

#     # remove first and last days
#     activityM = M[1:-1]

#     # if less than 7 days, skip subject
#     if activityM.shape[0] < 7:
#         return None, None

#     # remove 7-day segments of data if not enough
#     slidingWindowListForRemoval = sliding_window(activityM, window_size=7, gap=1)
#     daysToRemove = []
#     c = 0
#     for w in slidingWindowListForRemoval:
#         if len(w) < 7:
#             break
#         Wbinary = np.where(w > 0, 1, 0)
#         avgActivity = Wbinary.mean(axis=1).mean()
#         Wkp = np.where(w > 0, w, np.nan)
#         avgAmount = np.nanmedian(np.nanmedian(Wkp, axis=1))
#         if (avgActivity < 0.2) | (avgAmount < 50):
#             daysToRemove.extend(list(range(c, c + 7)))
#         c += 1
#     # remove rows corresponding to indices in daysToRemove
#     activityM = activityM[~activityM.index.isin([*set(daysToRemove)])]

#     # if less than 7 days, skip subject
#     if activityM.shape[0] < 7:
#         return None, None

#     # get matrix of typing speed by hour
#     speedM=df.groupby(['dayNumber','hour'],as_index = False).apply(lambda x: medianAAIKD(x)).pivot('dayNumber','hour')
#     speedM.columns = speedM.columns.droplevel(0)
#     for h in missingHours:
#         speedM.insert(h,h,[np.nan]*speedM.shape[0])
#     speedM = speedM.sort_index(ascending=True)
#     # remove first and last days
#     speedM = speedM[1:-1]
#     # remove rows corresponding to indices in daysToRemove
#     speedM = speedM[~speedM.index.isin([*set(daysToRemove)])]
#     speedM = speedM.replace(np.nan, 0)

# #     return activityM, speedM

# # adjacency matrix weight between consecutive days
# def day_weight(d1,d2):
#     return (d1+d2)

# # adjacency matrix weight between consecutive hours
# def hour_weight(h1,h2):
#     return (h1+h2)/2

# # calculate weighted adjacency matrix for graph regulated SVD
# def weighted_adjacency_matrix(mat):
#     import numpy as np
#     # days = rows
#     # hours = columns
#     W = np.zeros((mat.size, mat.size))
#     for i in range(mat.size):
#         for j in range(mat.size):
#             # iterate across hours of each day then across days
#             # d1h1, d1h2, d1h3, d1h4...d2h1, d2h2, d3h3...
#             i_Mi = i//mat.shape[1]
#             i_Mj = i%mat.shape[1]
#             j_Mi = j//mat.shape[1]
#             j_Mj = j%mat.shape[1]
#             # diagonals
#             if i == j:
#                 W[i,j] = 0
#             # if abs(subtraction of col indices) == 1 & subtraction of row indices == 0:
#             elif (abs(j_Mj-i_Mj) == 1) & ((j_Mi-i_Mi) == 0):
#                 W[i,j] = hour_weight(mat[i_Mi,i_Mj],mat[j_Mi,j_Mj])
#             # if abs(subtraction of row indices) == 1 & subtraction of col indices == 0:
#             elif (abs(j_Mi-i_Mi) == 1) & ((j_Mj-i_Mj) == 0):
#                 W[i,j] = day_weight(mat[i_Mi,i_Mj],mat[j_Mi,j_Mj])
#             # connect 23hr with 00hr
#             elif (i_Mj == mat.shape[1]-1) & ((j_Mi-i_Mi) == 1) & (j_Mj == 0):
#                 W[i,j] = hour_weight(mat[i_Mi,i_Mj],mat[i_Mi+1,0])
#             else:
#                 W[i,j] = 0
#     return W

# def regularized_svd(X, B, rank, alpha, as_sparse=False):
#     """
#     Perform graph regularized SVD as defined in
#     Vidar & Alvindia (2013).

#     Parameters
#     ----------
#     X : numpy array
#         m x n data matrix.

#     B : numpy array
#         n x n graph Laplacian of nearest neighborhood graph of data.

#     W : numpy array
#         n x n weighted adjacency matrix.

#     rank : int
#         Rank of matrix to approximate.

#     alpha : float
#         Scaling factor.

#     as_sparse : bool
#         If True, use sparse matrix operations. Default is False.

#     Returns
#     -------
#     H_star : numpy array
#         m x r matrix (Eq 15).

#     W_star : numpy array
#         r x n matrix (Eq 15).
#     """
#     import numpy as np
#     from scipy.linalg import svd
#     # from scipy.linalg import cholesky
#     from numpy.linalg import cholesky
#     from scipy.linalg import solve_triangular
#     # from scipy.linalg import inv
#     import scipy.sparse as sp
#     from sklearn.utils.extmath import randomized_svd
#     from sksparse import cholmod

#     if as_sparse:
#         # Use sparse matrix operations to reduce memory
#         I = sp.lil_matrix(B.shape)
#         I.setdiag(1)
#         C = I + (alpha * B)
#         print('Computing Cholesky decomposition')
#         factor = cholmod.cholesky(C)
#         D = factor.L()
#         print('Computing inverse of D.T')
#         invDt = sp.linalg.inv(D.T)
#         # Eq 11
#         print('Computing randomized SVD')
#         E, S, Fh = randomized_svd(X @ invDt,
#                                   n_components=rank,
#                                   random_state=123)
#         E_tilde = E[:, :rank]  # rank-r approximation; H_star = E_tilde (Eq 15)
#         H_star = E_tilde  # Eq 15
#         W_star = E_tilde.T @ X @ sp.linalg.inv(C)  # Eq 15
#     # else:
#     #     # Eq 11
#     #     I = np.eye(B.shape[0])
#     #     C = I + (alpha * B)
#     #     D = cholesky(C)
#     #     E, S, Fh = svd(X @ inv(D.T))
#     #     E_tilde = E[:, :rank]  # rank-r approximation; H_star = E_tilde (Eq 15)
#     #     H_star = E_tilde  # Eq 15
#     #     W_star = E_tilde.T @ X @ inv(C)  # Eq 15
#     # return H_star, W_star
#     else:
#     # Eq 11
#         I = np.eye(B.shape[0])
#         C = I + (alpha * B)
#         D = cholesky(C)
#         Y = solve_triangular(D, X.T, lower=True).T
#         E, S, Fh = svd(Y)
#         E_tilde = E[:, :rank]  # rank-r approximation; H_star = E_tilde (Eq 15)
#         H_star = E_tilde  # Eq 15
#         # print("C: {}, D: {}, X: {}, Y: {}, E: {}, E_tilde: {}".format(C.shape, D.shape, X.shape, Y.shape, E.shape, E_tilde.shape))
#         W_star = solve_triangular(D.T, Y.T @ E_tilde)  # Eq 15
#     return H_star, W_star

# def get_SVD(activityM, speedM):
#     """
#     Apply graph regularized SVD as defined in
#     Vidar & Alvindia (2013) to typing data.

#     Parameters
#     ----------
#     activityM : pandas dataframe
#                 BiAffect typing activity by hour of shape (days x hours).
#     speedM : pandas dataframe
#              BiAffect typing speed by hour of shape (days x hours).

#     Returns
#     -------
#     svdM : numpy array
#            graph regularized SVD of typing features matrix of shape (days x hours).
#     """
#     from scipy.sparse import csgraph
#     import numpy as np

#     if activityM is None:
#         return None
#     # normalize nKP matrix
#     activityM = np.log(activityM+1)
#     # SVD
#     # get adjacency matrix for SVD
#     W = weighted_adjacency_matrix(np.array(activityM))
#     # normalize keypress values
#     normKP = np.array(activityM).flatten()
#     ikd_vals = np.array(speedM).flatten()
#     data = np.vstack((normKP,ikd_vals))
#     # get graph laplacian
#     B = csgraph.laplacian(W)
#     # get graph normalized SVD
#     H_star, W_star = regularized_svd(data, B, rank=1, alpha=1, as_sparse=False)
#     # get SVD matrix
#     svdM = W_star.reshape(activityM.shape)
#     if svdM.max() <= 0:
#         svdM = svdM * -1
#     return svdM

# # create list of indices for each sliding window
# def sliding_window(elements, window_size, gap):
#     import numpy as np
#     if len(elements) <= window_size:
#        return elements
#     windows = []
#     ls = np.arange(0, len(elements), gap)
#     for i in ls:
#         windows.append(elements[i:i+window_size])
#     return windows

# def get_sleepWakeLabels(svd_mat):
#     """
#     Get sleep/wake labels per hour of BiAffect typing data.

#     Parameters
#     ----------
#     df : numpy array
#          graph regularized SVD of typing features matrix of shape (days x hours).

#     Returns
#     -------
#     sleepWakeMatrix : numpy array
#                       sleep/wake labels per hour of BiAffect typing data 
#                       of shape (days x hours).
#     """
#     import numpy as np
#     from skimage import segmentation

#     if svd_mat is None:
#         return None

#     # Binarize SVD
#     binarizedSVD = np.where(svd_mat == 0, 0,1)
#     # sleep/wake labels from binarized SVD matrix
#     sleep_label = 0
#     wake_label = 1
#     # flood fill main sleep component
#     # initiate matrix filled with value 2 (meaning no label yet)
#     floodFillM = np.full(shape=svd_mat.shape, fill_value=2, dtype='int')
#     for r in range(svd_mat.shape[0]):
#         # get row
#         row = binarizedSVD[r]
#         # if entire row is sleep, continue
#         if ((row == np.array([sleep_label]*len(row))).all()) == True:
#             floodFillM[r] = [sleep_label]*len(row)
#             continue
#         # if wake labels only during hours 2-15, get min from 0-24 hr range
#         if ((binarizedSVD[r, 2:15] == np.array([wake_label]*13)).all()) == True:
#             idx_rowMin = np.argmin(svd_mat[r])
#         else: # else limit min hour to between hr 2-15
#             idx_rowMin = np.argmin(svd_mat[r, 2:15]) + 2
#         # if min value not equal to sleep_label, then no sleep that day. continue
#         if binarizedSVD[r,idx_rowMin] != sleep_label:
#             floodFillM[r] = [wake_label]*len(row)
#             continue
#         # flood fill 
#         sleep_flood = segmentation.flood(binarizedSVD, (r,idx_rowMin))#NEED DIAG CONNECTIVITY #connectivity=1)
#         # replace output matrix row with flood fill values
#         floodFillM[r] = np.invert(sleep_flood[r])

#         # add sleep label before midnight if exists
#         # if iteration at last row
#         if r == svd_mat.shape[0]-1:
#             # if last cell is sleep label
#             if (row[-1] == sleep_label):
#                 i = 0
#                 # find earliest index of sleep label for that row prior to midnight
#                 while row[23-i] == sleep_label:
#                     end_idx = i
#                     i += 1
#                 # replace identified ending cells with sleep label
#                 floodFillM[r,(23-end_idx):] = sleep_label
#         # if interation not at last row
#         else:
#             # if last cell is sleep label and first cell of next row is sleep label
#             if (row[-1] == sleep_label) & (binarizedSVD[r+1,0] == sleep_label):
#                 i = 0
#                 # find earliest index of sleep label for that row prior to midnight
#                 while row[23-i] == sleep_label:
#                     end_idx = i
#                     i += 1
#                 # replace identified ending cells with sleep label
#                 floodFillM[r,(23-end_idx):] = sleep_label

#     # fill in gaps in sleep component
#     # get list of sliding windows to remove gaps in sleep
#     window_size = 6
#     hr_space = 1
#     floodFillFlatten = floodFillM.flatten()
#     slidingWindowList = sliding_window(floodFillFlatten, window_size, hr_space)

#     # iterate through windows and fill gap between first and last sleep index in window
#     windowCount = 0
#     for window in slidingWindowList:
#         try:
#             # get index of first sleep label in window
#             firstSleep_idx = next(i for i,v in enumerate(window) if v == sleep_label)
#         # if no sleep label within window, continue
#         except StopIteration:
#             windowCount += 1
#             continue
#         # get index of last sleep label in window
#         lastSleep_idx = len(window) - next(i for i, val in enumerate(reversed(window), 1) if val != wake_label)
#         # get index of first label in window from array of all labels
#         grpIdx0 = windowCount * hr_space
#         # replace window values with sleep labels within identified indices
#         floodFillFlatten[(grpIdx0+firstSleep_idx):(grpIdx0+lastSleep_idx)] = [sleep_label]*(lastSleep_idx-firstSleep_idx)
#         windowCount += 1
#     # reshape array to be matrix of sleep/wake activity
#     sleepWakeMatrix = floodFillFlatten.reshape(binarizedSVD.shape)
#     return sleepWakeMatrix

def svd_output(dfKP):
    Mactivity, Mspeed = get_typingMatrices(dfKP)
    if Mactivity is None:
        return None
    svd = get_SVD(Mactivity, Mspeed)
    return svd

def labels_output(dfKP):
    Mactivity, Mspeed = get_typingMatrices(dfKP)
    if Mactivity is None:
        return None
    svd = get_SVD(Mactivity, Mspeed)
    labelsMatrix = get_sleepWakeLabels(svd)
    return labelsMatrix

## Circular variance/mean
def var_circular_variance(df):
    import numpy as np
    from scipy import stats
    M = svd_output(df)
    if M is None:
        return np.nan
    circVarList = np.apply_along_axis(stats.circvar, 1, M)
    return np.nanvar(circVarList)

## Circular variance/mean using normalized KP
def normalized_var_circular_variance(df):
    import numpy as np
    from scipy import stats
    M = svd_output(df)
    if M is None:
        return np.nan
    sumKP = M.sum().sum()
    normM = M/sumKP
    circVarList = np.apply_along_axis(stats.circvar, 1, normM)
    return np.nanvar(circVarList)

def var_circular_mean(df):
    import numpy as np
    from scipy import stats
    M = svd_output(df)
    if M is None:
        return np.nan
    circMeanList = np.apply_along_axis(stats.circmean, 1, M)
    return np.nanvar(circMeanList)

def normalized_var_circular_mean(df):
    import numpy as np
    from scipy import stats
    M = svd_output(df)
    if M is None:
        return np.nan
    sumKP = M.sum().sum()
    normM = M/sumKP
    circMeanList = np.apply_along_axis(stats.circmean, 1, normM)
    return np.nanvar(circMeanList)

# find # of hours of no activity
def medianAmount_noActivity(dfKP, sleep_label=0):
    import numpy as np
    import pandas as pd
    labs = labels_output(dfKP)
    if labs is None:
        return np.nan
    days = (pd.Series(np.arange(labs.shape[0])).repeat(labs.shape[1]).reset_index(drop=True))+1
    dfLabels = pd.DataFrame(np.vstack((days,labs.flatten())).T, columns = ['day','label'])
    noActivityAmt = dfLabels.groupby('day').apply(lambda x: x[x['label'] == sleep_label].shape[0])
    return np.nanmedian(noActivityAmt)
    
def varAmount_noActivity(dfKP, sleep_label=0):
    import numpy as np
    import pandas as pd
    labs = labels_output(dfKP)
    if labs is None:
        return np.nan
    days = (pd.Series(np.arange(labs.shape[0])).repeat(labs.shape[1]).reset_index(drop=True))+1
    dfLabels = pd.DataFrame(np.vstack((days,labs.flatten())).T, columns = ['day','label'])
    noActivityAmt = dfLabels.groupby('day').apply(lambda x: x[x['label'] == sleep_label].shape[0])
    return np.nanvar(noActivityAmt)

def cosine_similarity(a,b):
    import numpy as np
    from numpy.linalg import norm
    cosine = np.dot(a,b)/(norm(a)*norm(b))
    return cosine

def adj_cosine_similarity(a,b):
    import numpy as np
    from numpy.linalg import norm
    a_normalized = a - np.mean(a)
    b_normalized = b - np.mean(b)
    cosine = np.dot(a_normalized,b_normalized)/(norm(a_normalized)*norm(b_normalized))
    return cosine

def norm_cosine_similarity(a,b):
    import numpy as np
    from numpy.linalg import norm
    a_normalized = a/sum(a) # /norm(a)
    b_normalized = b/sum(a) # /norm(b)
    cosine = np.dot(a_normalized,b_normalized)/(norm(a_normalized)*norm(b_normalized))
    return cosine

def get_medianCosSim(df, day_diff):
    import numpy as np
    # SVD matrix
    svdMat = svd_output(df)
    if svdMat is None:
        return np.nan
    sim_list = []
    for d in range(svdMat.shape[0]):
        if d < svdMat.shape[0]-day_diff:
            sim = cosine_similarity(svdMat[d], svdMat[d+day_diff])
            sim_list.append(sim)
    return np.nanmedian(sim_list)

def get_medianAdjCosSim(df, day_diff):
    import numpy as np
    # SVD matrix
    svdMat = svd_output(df)
    if svdMat is None:
        return np.nan
    sim_list = []
    for d in range(svdMat.shape[0]):
        if d < svdMat.shape[0]-day_diff:
            sim = adj_cosine_similarity(svdMat[d], svdMat[d+day_diff])
            sim_list.append(sim)
    return np.nanmedian(sim_list)

def get_medianNormCosSim(df, day_diff):
    import numpy as np
    # SVD matrix
    svdMat = svd_output(df)
    if svdMat is None:
        return np.nan
    sim_list = []
    for d in range(svdMat.shape[0]):
        if d < svdMat.shape[0]-day_diff:
            sim = norm_cosine_similarity(svdMat[d], svdMat[d+day_diff])
            sim_list.append(sim)
    return np.nanmedian(sim_list)

def decay_function(t, a, b):
    import numpy as np
    return a * np.exp(-b*t)

def decay_coeff(M, day_range=7):
    from scipy.optimize import curve_fit
    l = []
    for d in range(day_range):
        cossim=get_medianCosSim(M, d)
        l.append(cossim)
    x=list(range(7))
    y=l
    popt,_ = curve_fit(decay_function, x, y)
    return popt[1]

######################################################################
## REST ACTIVITY RHYTHM FEATURES
#  acrophase
def acrophase(df):
    import numpy as np
    # SVD matrix
    svdMat = svd_output(df)
    if svdMat is None:
        return np.nan
    # # get hour with most keypresses over whole matrix
    # max_kp_idx = np.unravel_index(np.argmax(svdMat), svdMat.shape)
    # get average hour with most keypresses
    avgMaxKP_idx = np.argmax(svdMat, axis=1).mean()
    return avgMaxKP_idx #max_kp_idx[1] # return just the hour

# RAR-alpha
# average ratio of sleep to wake
def RAR_alpha(dfKP, sleep_label=0, wake_label=1):
    import numpy as np
    import pandas as pd
    labs = labels_output(dfKP)
    if labs is None:
        return np.nan
    days = (pd.Series(np.arange(labs.shape[0])).repeat(labs.shape[1]).reset_index(drop=True))+1
    dfLabels = pd.DataFrame(np.vstack((days,labs.flatten())).T, columns = ['day','label'])
    noActivityAmt = dfLabels.groupby('day').apply(lambda x: x[x['label'] == sleep_label].shape[0])
    activityAmt = dfLabels.groupby('day').apply(lambda x: x[x['label'] == wake_label].shape[0])
    return np.mean(noActivityAmt / activityAmt)

# RAR-beta
# avg nKP in "wake" hours
def RAR_beta(dfKP):
    import numpy as np
    Mkp, _ = get_typingMatrices(dfKP)
    if Mkp is None:
        return np.nan
    avgnKP = Mkp.sum(axis=1).mean()
    return avgnKP
    # # if calculating from the SVD instead of raw nKP
    # import numpy.ma as ma
    # # SVD matrix
    # svdMat = svd_output(dfKP)
    # if svdMat is None:
    #     return np.nan
    # labelsMatrix = labels_output(dfKP)

    # wake_mask = np.isin(labelsMatrix,[sleep_label]) # mask all sleep labels
    # wakeOnlyMat = ma.masked_array(svdMat, wake_mask)
    # avgWakeKP = wakeOnlyMat.sum(axis=1).mean()

# below taken from pyActigraphy
# https://github.com/ghammad/pyActigraphy/blob/master/pyActigraphy/metrics/metrics.py
def intradaily_variability(dfKP):
    import numpy as np
    import pandas as pd
    Mkp, _ = get_typingMatrices(dfKP)
    if Mkp is None:
        return np.nan
    day_mssd = pd.Series(np.array(Mkp).flatten()).diff(1).pow(2).mean()
    overall_var = np.var(np.array(Mkp))
    mean_mssd_ratio = day_mssd / overall_var
    return mean_mssd_ratio

def intradaily_stability(dfKP):
    import numpy as np
    Mkp, _ = get_typingMatrices(dfKP)
    if Mkp is None:
        return np.nan
    meanKPperHr = np.mean(Mkp, axis=0)
    var_meanKPperHour = np.var(meanKPperHr)
    overall_var = np.var(np.array(Mkp))
    ratio = var_meanKPperHour / overall_var
    return ratio

def M10(arr):
    import pandas as pd
    window_activity = pd.Series(arr).rolling(10).sum().shift(-9)
    m10 = window_activity.max()
    return m10

def L5(arr):
    import pandas as pd
    window_activity = pd.Series(arr).rolling(5).sum().shift(-4)
    l5 = window_activity.min()
    return l5

def relative_amplitude(dfKP):
    import numpy as np
    Mkp, _ = get_typingMatrices(dfKP)
    if Mkp is None:
        return np.nan
    rowSums=np.array(Mkp.sum(axis=1))
    normalizedM = (np.array(Mkp).T/rowSums).T
    mean_M10 = np.apply_along_axis(M10, 1, normalizedM).mean()
    mean_L5 = np.apply_along_axis(L5, 1, normalizedM).mean()
    return mean_M10 - mean_L5


