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
import copy
# from sklearn.preprocessing import normalize
# from sklearn.preprocessing import StandardScaler

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

############################################################################################
pathIn = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/keypress/'
pathOut = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/graph_regularized_SVD/matrices/'

# list of user accel files
all_files = sorted(glob.glob(pathIn + "*.csv"), key = numericalSort)

for file in all_files:
    df = pd.read_csv(file, index_col=False)
    user = int(df['userID'].unique())
    print('user: {}'.format(user))

    # if user != 11:
    #     continue

    df['hour'] = pd.to_datetime(df['keypressTimestampLocal']).dt.hour
    M1 = df.groupby(['dayNumber','hour'],as_index = False).size().pivot('dayNumber','hour').fillna(0)

    if M1.shape[0] < 7:
        print('not enough days')
        continue
    # medMax = M.max().median()
    # if medMax < 300:
    #     print('not enough kp per day')
    #     continue

    missingHours = [h for h in range(24) if h not in list(M1['size'].columns)] #M.index
    M1.columns = M1.columns.droplevel(0)
    for h in missingHours:
        # M.loc[h] = [0]*M.shape[1] # to add row
        M1.insert(h,h,[0]*M1.shape[0])
    M1 = M1.sort_index(ascending=True)    

    Mspeed=df.groupby(['dayNumber','hour'],as_index = False).apply(lambda x: medianAAIKD(x)).pivot('dayNumber','hour')
    Mspeed.columns = Mspeed.columns.droplevel(0)
    for h in missingHours:
        # M.loc[h] = [0]*M.shape[1] # to add row
        Mspeed.insert(h,h,[np.nan]*Mspeed.shape[0])
    Mspeed = Mspeed.sort_index(ascending=True)

    
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
    hrs_arr = list(range(n_hrs)) * n_days
    data = np.vstack((days_arr,hrs_arr, kp_values))
    B = csgraph.laplacian(W)
    H_star, W_star = regularized_svd(data, B, rank=1, alpha=0.1, as_sparse=False)

    out2 = W_star.reshape(M1.shape)
    out2 = out2 * -1
    clip_amount = out2.max()/4
    cutoff = np.clip(out2, 0, clip_amount)

    X = out2 #cutoff
    n_hours = X.shape[1]
    n_days = X.shape[0]

    # Reshape data into observations x features
    # Columns (features): [day, hour, keypresses]
    # Rows (observations): 726
    dfM = pd.DataFrame(X.T)
    dfM = dfM.melt(var_name='day', value_name='keypresses')
    # dfM['day'] = (pd.Series(np.arange(n_days))
    #             .repeat(n_hours).reset_index(drop=True))
    dfM['hour'] = list(range(24))*n_days
    dfM = dfM[['day', 'hour', 'keypresses']]  # rearrange columns
 
    # Something simple for first approach: PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(dfM.to_numpy())
    kmeans = KMeans(n_clusters=2, random_state=123).fit(X_pca)
    dfPCA = pd.DataFrame({
        'pca_x': X_pca[:, 0],
        'pca_y': X_pca[:, 1],
        'cluster': kmeans.labels_})
    
    # # Visualize k-means clusters in PCA embedding
    # f, ax = plt.subplots()
    # sns.scatterplot(data=dfPCA, x='pca_x', y='pca_y', hue='cluster', ax=ax)
    # plt.show()
    # plt.clf()



    # n_bins = 4
    # b,bins,patches = plt.hist(x=dfPCA['pca_x'], bins=n_bins)
    # dfPCA['cluster_filt'] = np.where((dfPCA['pca_x'] <= bins[1]) | (dfPCA['pca_x'] >= bins[len(bins)-2]), dfPCA['cluster'],np.nan)
    # # dfPCA.loc[(dfPCA['pca_x'] <= bins[1]) | (dfPCA['pca_x'] >= bins[len(bins)-2])]
    
    # # just plot of one matrix M
    # # f, ax = plt.subplots(nrows=1,ncols=1, sharex=False, sharey=True, figsize=(8,10))
    # # sns.heatmap(M, cmap='viridis', vmin=0, vmax=500, cbar_kws{'label': '# keypresses', 'fraction': 0.043})





    # Visualize original data heatmap and heatmap with k-means cluster labels
    f, ax = plt.subplots(nrows=2,ncols=2, sharex=False, sharey=True,
                        figsize=(10,10))
    # PLOT 1
    sns.heatmap(M1, cmap='viridis', ax=ax[0,0], #vmin=0, vmax=500,
                cbar_kws={'label': '# keypresses', 'fraction': 0.043})
    # PLOT 2
    sns.heatmap(out2, cmap='viridis', ax=ax[0,1], #vmin=0, vmax=200,
                cbar_kws={'label': '# keypresses', 'fraction': 0.043})
    # PLOT 3
    sns.heatmap(cutoff, cmap='viridis', ax=ax[1,0], #vmin=0, vmax=clip_amount,
                cbar_kws={'label': '# keypresses', 'fraction': 0.043})

    # PLOT 3    
    cluster_mat = dfPCA['cluster'].to_numpy().reshape(X.shape)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom',
        colors=['#de8f05', '#0173b2'],
        N=2)
    sns.heatmap(cluster_mat, ax=ax[1,1], cmap=cmap,
                cbar_kws={'fraction': 0.043})
    colorbar = ax[1,1].collections[0].colorbar
    colorbar.set_ticks([0.25, 0.75])
    colorbar.set_ticklabels(['0', '1'])
    colorbar.set_label('Cluster')

    # # PLOT 4
    # cluster_mat = dfPCA['cluster_filt'].to_numpy().reshape(X.shape)
    # cmap = mpl.colors.LinearSegmentedColormap.from_list(
    #     'Custom',
    #     colors=['#de8f05', '#0173b2'],
    #     N=2)
    # sns.heatmap(cluster_mat, ax=ax[1,1], cmap=cmap,
    #             cbar_kws={'fraction': 0.043})
    # colorbar = ax[1,1].collections[0].colorbar
    # colorbar.set_ticks([0.25, 0.75])
    # colorbar.set_ticklabels(['0', '1'])
    # colorbar.set_label('Cluster')


    ax[0,0].set(title='Original', xlabel='Hour', ylabel='Day')
    ax[0,1].set(title='Graph Reg. SVD', xlabel='Hour', ylabel='Day')
    ax[1,0].set(title='Truncated Graph Reg. SVD', xlabel='Hour', ylabel='Day')
    ax[1,1].set(title='K-Means Clustering from PCA', xlabel='Hour', ylabel='Day')
    # ax[1,1].set(title='Filtered K-Means Clustering from PCA', xlabel='Hour', ylabel='Day')
    f.tight_layout()
    plt.show(f)
    # f.savefig(pathOut+'HRxDAYsizeMat/user_{}_svd_PCA-kmeans.png'.format(user))
    plt.close(f)
    
    
    break

# print('finish')





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


# %%



# plot pc1 vs pc2

# then get i,j of each