"""
@author: Mindy Ross
python version 3.7.4
pandas version: 1.3.5
numpy version: 1.19.2
"""
# Get features about diurnal patterns and estimated sleep from graph regularized SVD labels

from estimate_sleepWake_cycle import *

# calculate typing speed (median IKD)
def medianAAIKD(dataframe):
    import numpy as np
    grpAA = dataframe.loc[((dataframe['keypress_type'] == 'alphanum') &
                                (dataframe['previousKeyType'] == 'alphanum'))]
    # get median IKD
    medAAIKD = np.nanmedian(grpAA['IKD']) if len(grpAA) >= 20 else float('NaN')
    return(medAAIKD)

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

def var_circular_mean(df):
    import numpy as np
    from scipy import stats
    M = svd_output(df)
    if M is None:
        return np.nan
    circMeanList = np.apply_along_axis(stats.circmean, 1, M)
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