#%%
# Loran's code to link estimated sleep from typing activity and CLEAR3 sleep self-reports
# reformatted to run on python 3.7

import function_sleepWakeLabels as sleep
from save_gsvds import save_gsvds
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import glob
import re
from os.path import join
from pathlib import Path
from datetime import date
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pickle
from sklearn.metrics import mean_absolute_error as mae


dat_dir = '/home/mindy/Desktop/BiAffect-iOS/CLEAR/Loran_sleep/data/'
all_files = sorted(glob.glob(dat_dir+"sub-*/preproc/*dat-kp.csv", recursive=True))
pat = re.compile(r"sub-(\d+)")
subs = [re.search(pat, f).group(1) for f in all_files]
gsvd_file = 'gsvd_results.pkl'


# day_weight functions
def day_weight1(d1,d2,arr1,arr2,arrAround,dAround):
    return (d1+d2)/2

def cosine_similarity(a,b):
    import numpy as np
    from numpy.linalg import norm
    cosine = np.dot(a,b)/(norm(a)*norm(b))
    return cosine

def day_weight2(d1,d2,arr1,arr2,arrAround,dAround):
    import numpy as np
    # get cosine similarity between week around the day
    cosSimList = np.apply_along_axis(cosine_similarity, 1, np.array(arrAround), b=np.array(arr1))    
    # remove day's arr compared to itself
    cosSimList = cosSimList[cosSimList <1]
    # get median cosine similarity 
    medCosSim = np.nanmedian(cosSimList)
    return medCosSim

def day_weight3(d1,d2,arr1,arr2,arrAround,dAround):
    import numpy as np
    cosSim = cosine_similarity(arr1,arr2)
    return cosSim

def day_weight4(d1,d2,arr1,arr2,arrAround,dAround):
    return np.nanmedian(dAround)

def day_weight5(d1,d2,arr1,arr2,arrAround,dAround):
    import numpy as np
    # get cosine similarity between week around the day
    cosSimList = np.apply_along_axis(cosine_similarity, 1, np.array(arrAround), b=np.array(arr1))    
    # remove day's arr compared to itself
    cosSimList = cosSimList[cosSimList <1]
    # get median cosine similarity 
    medCosSim = np.nanmedian(cosSimList)
    return medCosSim * (d1+d2)/2

def day_weight6(d1,d2,arr1,arr2,arrAround,dAround):
    import numpy as np
    return np.nanmedian(dAround) * (d1+d2)/2
    
# def day_weight(arr1,arr2,dAround):
    # import numpy as np
    # # # get cosine similarity between week around the day
    # # cosSimList = np.apply_along_axis(cosine_similarity, 1, np.array(arrAround), b=np.array(arr1))    
    # # # remove day's arr compared to itself
    # # cosSimList = cosSimList[cosSimList <1]
    # # # get median cosine similarity 
    # # medCosSim = np.nanmedian(cosSimList)

    # # cosSim = cosine_similarity(arr1,arr2)


    # # compare same hour across different days
    # med = np.nanmedian(dAround)
    # # print(dAround)
    # # print(med)
    # return med * (arr1+arr2)/2


function_ls = [day_weight1,day_weight2,day_weight3,day_weight4,day_weight5,day_weight6]
error_ls = []
for func in function_ls:
    print('function: {}'.format(func))

    # Calculate graph SVDs, save them to file, and read them back in.
    save_gsvds(dat_dir, all_files, subs, out_file=gsvd_file, fun = func)
    with open(join(dat_dir, gsvd_file), 'rb') as handle:
        gsvd_results = pickle.load(handle)

    # Extract (complete rows of) CLEAR-3 sleep data.
    self_reports_raw = pd.read_csv(join(dat_dir, "clear3daily_20221205_sleep.csv"), index_col=False)
    self_reports = self_reports_raw[['id', 'daterated', 'sleepdur_yest', 'SleepLNQuality']]
    self_reports = self_reports.dropna()
    self_reports['daterated'] = self_reports['daterated'].map(lambda d: date.fromisoformat(d))


    cors = {}
    sleep_scores = {}

    for sub, res in gsvd_results.items():
        print(sub)

        # sub = subs[0]
        # res = gsvd_results.get(sub)
        
        Mactivity = res['Mactivity']
        Mspeed = res['Mspeed']
        svd = res['svd']
        sleepMatrix = res['sleepMatrix']

        # Fuse self-report with day numbers
        sr_sub = self_reports.loc[self_reports['id'] == int(sub)]
        merged = res['dates'].merge(sr_sub, how='outer', left_on='date', right_on='daterated')
        sub_sleep_scores = merged[['dayNumber', 'sleepdur_yest', 'SleepLNQuality']]
        
        # Fuse self-report with predicted sleep scores
        days = Mactivity.index
        sleep_pred = np.sum(1 - sleepMatrix, axis=1)
        sleep_pred = pd.DataFrame({'dayNumber': days.values, 'sleep_pred': sleep_pred})
        sub_sleep_scores = sub_sleep_scores.merge(sleep_pred, how='outer', on='dayNumber')
        sleep_scores[sub] = sub_sleep_scores
        
        ss_complete = sub_sleep_scores.dropna()     
        if len(ss_complete) <2:
            continue
        plt.scatter(ss_complete['sleep_pred'],ss_complete['sleepdur_yest'])
        plt.show()

        error = mae(ss_complete['sleepdur_yest'], ss_complete['sleep_pred'])
        print('error: {}'.format(error))
        error_ls.append((func,sub,error))
        
    #     break
    # break

print('finish')

print(error_ls)
# %%

dfError = pd.DataFrame(error_ls,columns=['funct','userID','error'])
dfError['funct'] = dfError['funct'].astype(str)
aggError = dfError.groupby('funct').agg({'error':'mean'})








# %%
