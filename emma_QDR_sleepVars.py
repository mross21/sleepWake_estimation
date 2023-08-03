#%%
import pandas as pd
from pyarrow import parquet
import os
from typing_regularity_features_v2 import *

def find_keypress_file(user, kp_file_list):
    import re
    for u in kp_file_list:
        file_user = int(re.findall('\d+', u)[0])
        if file_user == user:
            return u



pathQDR = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/QDR_emma/'
QDRfile = 'Back_QDR_KP_discard_multiple_with_demo_6-23-23.csv'
pathKP = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/keypress/'
all_files = os.listdir(pathKP)

dfQDR = pd.read_csv(pathQDR + QDRfile, index_col=False)
dfQDR['userID'] = dfQDR['unmID'].str.extract('(\d+)').astype(int)
# userList = sorted(dfQDR['userID'].unique())

for i in range(len(dfQDR)):
    row = dfQDR.iloc[i]
    user = row['userID']
    print('user {}'.format(user))

    kpFile = find_keypress_file(user, all_files)
    dfKP = pd.read_parquet(pathKP + kpFile, engine='pyarrow')

    minDate = row['time_start_back']
    maxDate = row['time_end_back']

    grpKP = dfKP.loc[(dfKP['sessionTimestampLocal'] >= minDate) & (dfKP['sessionTimestampLocal'] <= maxDate)]



    break















# %%
