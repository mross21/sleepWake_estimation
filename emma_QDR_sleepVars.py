#%%
import pandas as pd
from pyarrow import parquet
import numpy as np
import os
from typing_regularity_features import *

def find_keypress_file(user, kp_file_list):
    import re
    for u in kp_file_list:
        file_user = int(re.findall('\d+', u)[0])
        if file_user == user:
            return u

def cosine_similarity(a,b):
    import numpy as np
    from numpy.linalg import norm
    cosine = np.dot(a,b)/(norm(a)*norm(b))
    return cosine

############################################################################################
pathQDR = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/QDR_emma/'
QDRfile = 'Back_QDR_KP_discard_multiple_with_demo_6-23-23.csv'
pathKP = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/keypress/'
all_files = os.listdir(pathKP)

dfQDR = pd.read_csv(pathQDR + QDRfile, index_col=False)
dfQDR['time_start_back_date'] = pd.to_datetime(dfQDR['time_start_back']).dt.date
dfQDR['time_end_back_date'] = pd.to_datetime(dfQDR['time_end_back']).dt.date
dfQDR['userID'] = dfQDR['unmID'].str.extract('(\d+)').astype(int)
userList = sorted(dfQDR['userID'].unique())

# initiate empty outer dictionary with all user keys
dictOuter = dict(zip(userList, [None]*len(userList)))
for user in userList:
    print('user {}'.format(user))
    # get user's QDR
    grpQDR = dfQDR.loc[dfQDR['userID'] == user]
    # get user's keypress file
    kpFile = find_keypress_file(user, all_files)
    dfKP = pd.read_parquet(pathKP + kpFile, engine='pyarrow')
    dfKP['date'] = pd.to_datetime(dfKP['date']).dt.date
    # match day number and date
    dictDatesAll = dict(zip(dfKP['dayNumber'], dfKP['date']))
    # initiate empty inner dictionary with all date keys
    dictInner = dict(zip(grpQDR['time_end_back_date'].unique().astype(str), 
                         [np.nan]*len(grpQDR['time_end_back_date'].unique())))
    
    # get matrix of typing activity
    Mactivity, Mspeed = get_typingMatrices(dfKP)
    if Mactivity is None:
        dictOuter[user] = dictInner
        continue
    # reduce dictionary to only dates in typing matrices
    dictDates = {k: dictDatesAll[k] for k in Mactivity.index}
    dictIdx = dict(zip(dictDates.values(), range(Mactivity.shape[0])))
    # get svd 
    svd = get_SVD(Mactivity, Mspeed)
    # get sleep labels matrix
    sleepMatrix = get_sleepWakeLabels(svd)
    # get date ranges for QDRs
    startDates = grpQDR['time_start_back_date'].unique()
    endDates = grpQDR['time_end_back_date'].unique()
    # loop through dates to calculate sleep features
    for start, end in zip(startDates, endDates):
        print('start day: {}'.format(start))
        print('end day: {}'.format(end))

        # GET HOURS OF SLEEP THE NIGHT BEFORE QDR
        hoursSleep=np.nan
        dictInner[str(end)] = hoursSleep
        # get index corresponding to end date
        try:
            dayNumEnd = dictIdx[end] 
        except KeyError:
            print('end date {} not in matrix'.format(end))
            continue
        # get array corresponding to end date
        sleepArrEnd = sleepMatrix[dayNumEnd]
        # get sleep for night before window end date
        # get first occurrance of sleep
        if ((sleepArrEnd == [1]*len(sleepArrEnd)).all()) == True:
            hoursSleep = 0
        else:
            idxFirstSleep = np.where(sleepArrEnd==0)[0][0]
            if idxFirstSleep > 15:
                hoursSleep = 0
                continue
            # get number of hours of sleep following first occurrance
            i = 0
            while sleepArrEnd[idxFirstSleep+i] == 0:
                hoursSleep = 1+i
                i += 1
                if i == (24-idxFirstSleep):
                    break
            # if first cell is sleep and array isn't at index 0, get any sleep from before midnight
            if (idxFirstSleep == 0) & (dayNumEnd != 0):
                # get index corresponding to start date
                try:
                    dayNumStart = dictIdx[start]
                except KeyError:
                    print('start date {} not in matrix'.format(start))
                    continue
                # get array corresponding to start date
                sleepArrStart = sleepMatrix[dayNumStart]
                # if last cell is sleep, find earliest index of sleep in row
                if sleepArrStart[-1] == 0:
                    j = 0
                    # find earliest index of sleep label for that row prior to midnight
                    while (sleepArrStart[23-j] == 0):
                        hoursNightBefore = 1+j
                        j += 1
                        if j == 24:
                            break
                    hoursSleep += hoursNightBefore

        # # GET COSINE REGULARITY TO PREVIOUS DAY
        # # get index corresponding to start date
        # try:
        #     dayNumStart = dictIdx[start]
        # except KeyError:
        #     print('start date {} not in matrix'.format(start))
        #     continue
        # # get array corresponding to start date
        # sleepArrStart = sleepMatrix[dayNumStart]
        # cosSim = cosine_similarity(sleepArrStart, sleepArrEnd)

        # add date and variables to dictionary
        dictInner[str(end)] = hoursSleep #,'cosineSimilarity': cosSim}
        print('-------------------------')
    # add hours sleep dictionary to userID
    dictOuter[user] = dictInner
    print('----------------------------------------')
print('finish')

dfQDR['time_end_back_date'] = dfQDR['time_end_back_date'].astype(str)
dfQDR['hoursSleep'] = dfQDR.apply(lambda x: dictOuter[x['userID']][x['time_end_back_date']], axis=1)
dfQDR.to_csv(pathQDR+'QDR_biaffect_withSleep.csv', index=False)

# %%
import matplotlib.pyplot as plt

plt.hist(dfQDR['hoursSleep'])
plt.show()

# %%
