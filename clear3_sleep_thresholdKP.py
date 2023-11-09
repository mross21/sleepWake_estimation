#%%
# Loran's code to link estimated sleep from typing activity and CLEAR3 sleep self-reports
# reformatted to run on python 3.7

import function_sleepWakeLabels as sleep
from save_gsvds_thresholdKP import save_gsvds_thresholdKP
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

# sns.set_theme(style="darkgrid")

def reshape_sleep_labels(sleepM):
    # reshape sleep matrix prior to calculating amount of sleep
    import numpy as np
    sleepM = sleepM.astype(float)
    # check which columns have all same value
    sameVals = np.all(sleepM == sleepM[0,:], axis = 0)
    sameCols = np.where(sameVals == True)[0]
    for i in sameCols[::-1]: # flip order so that choose last col w/ all wake labels
        if np.unique(sleepM.T[i])[0] == 1:
            idx = i
            break
    try: # check if idx variable exists
        idx
    except NameError: # if not, get first col with most wake labels
        maxVals = np.sum(sleepM, axis=0) # get sum of all wake labels for each col
        maxCols = np.where(maxVals == max(maxVals)) # find max sum (most wake labels)
        idx = maxCols[0][-1]
    flatM = sleepM.flatten()
    addNaN1 = np.insert(flatM, 0, [np.nan]*(23-idx))
    addNaN2 = np.append(addNaN1,[np.nan]*(idx+1))
    newSleepMat = addNaN2.reshape(sleepM.shape[0]+1,sleepM.shape[1])
    outMat = newSleepMat[:-1,]
    return outMat


dat_dir = '/home/mindy/Desktop/BiAffect-iOS/CLEAR/Loran_sleep/data/'
all_files = sorted(glob.glob(dat_dir+"sub-*/preproc/*dat-kp.csv", recursive=True))
pat = re.compile(r"sub-(\d+)")
subs = [re.search(pat, f).group(1) for f in all_files]
gsvd_file = 'gsvd_results_thresholdKP.pkl'

# Calculate graph SVDs, save them to file, and read them back in.
save_gsvds_thresholdKP(dat_dir, all_files, subs, out_file=gsvd_file)
with open(join(dat_dir, gsvd_file), 'rb') as handle:
    gsvd_results = pickle.load(handle)


# Extract (complete rows of) CLEAR-3 sleep data.
self_reports_raw = pd.read_csv(join(dat_dir, "clear3daily_20221205_sleep.csv"), index_col=False)
self_reports = self_reports_raw[['id', 'daterated', 'sleepdur_yest', 'SleepLNQuality']]
self_reports = self_reports.dropna()
self_reports['daterated'] = self_reports['daterated'].map(lambda d: date.fromisoformat(d))

# Generate plots of predicted and true sleep for every subject.
# I don't want all these plots embedded in this notebook
backend = mpl.rcParams['backend']
mpl.use("agg")
Path(dat_dir+'images').mkdir(exist_ok=True)
cors = {}
sleep_scores = {}
for sub, res in gsvd_results.items():
    print(sub)

    Mactivity = res['Mactivity']
    Mspeed = res['Mspeed']
    svd = res['svd']
    sleepMatrix = res['sleepMatrix']
    reshapedSleepMatrix = reshape_sleep_labels(sleepMatrix)

    # find avg number of hours of activity/day
    Mbinary = np.where(Mactivity > 0, 1, 0)
    avgActivityPerDay = Mbinary.mean(axis=1).mean()
    # of the days with kp, find median amount
    Mkp = np.where(Mactivity > 0, Mactivity, np.nan)
    avgAmountPerDay = np.nanmedian(np.nanmedian(Mkp, axis=1))

    # if sub == '3009':
    #     break

    # Plot steps
    plot = sleep.plot_heatmaps(Mactivity, Mspeed, svd, sleepMatrix)
    plot.savefig(dat_dir+"images/matrices_sub-{}_thresholdKP.png".format(sub), dpi=300)
    plt.close()
    sr_sub = self_reports.loc[self_reports['id'] == int(sub)]

    # Fuse self-report with day numbers
    merged = res['dates'].merge(sr_sub, how='outer', left_on='date', right_on='daterated')
    sub_sleep_scores = merged[['dayNumber', 'sleepdur_yest', 'SleepLNQuality']]
    
    # Fuse self-report with predicted sleep scores
    days = Mactivity.index
    sleep_pred = np.sum(1 - reshapedSleepMatrix, axis=1)
    sleep_pred = pd.DataFrame({'dayNumber': days.values, 'sleep_pred': sleep_pred})
    sub_sleep_scores = sub_sleep_scores.merge(sleep_pred, how='outer', on='dayNumber')
    sub_sleep_scores['avgActivityPerDay'] = avgActivityPerDay
    sub_sleep_scores['avgAmountPerDay'] = avgAmountPerDay

    sleep_scores[sub] = sub_sleep_scores

    ### Scatter plot
    ss_complete = sub_sleep_scores.dropna()
    if len(ss_complete.index) < 2:
        cor = 0.0
        p_cor = 1.0
        cors[sub] = {
            'cor': cor,
            'p_cor': p_cor}
        continue
    else:
        cor_res = pearsonr(ss_complete['sleepdur_yest'], ss_complete['sleep_pred'])
        cor = cor_res[0] #.statistic
        if np.isnan(cor) == True:
            print('nan')
            break
        p_cor = cor_res[1] #.pvalue
        cors[sub] = {
            'cor': cor,
            'p_cor': p_cor}
    sns.stripplot(data=sub_sleep_scores, x="sleepdur_yest", y="sleep_pred", alpha=0.4) \
        .set(
            title=fr"Sub {sub}, $\rho$ = {cor:.2f} (p = {p_cor:.2g})",
            xlabel="Reported sleep duration (hours)",
            ylabel="Predicted sleep duration (hours)"
        )
    plt.savefig(dat_dir+"images/scatter_sub-{}_thresholdKP.png".format(sub), dpi=300)
    plt.close()

    ### Plot sleep duration
    fig = plt.figure() #constrained_layout=False)
    fig.suptitle("Sleep duration sub {}".format(sub))
    gs = GridSpec(1, 5, fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2:], sharey=ax1)

    ax1.plot(sub_sleep_scores['sleepdur_yest'], sub_sleep_scores['dayNumber'])
    ax1.invert_yaxis()
    ax1.set_title('Reported')

    ax2.plot(sub_sleep_scores['sleep_pred'], sub_sleep_scores['dayNumber'])
    ax2.set_title('Predicted')
    plt.setp(ax2.get_yticklabels(), visible=False)

    sns.heatmap(svd, cmap='viridis', ax=ax3,vmin=0,vmax=0.25,
                cbar_kws={'label': 'Typing intensity', 'fraction': 0.043})
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.savefig(dat_dir+"images/duration_sub-{}_thresholdKP.png".format(sub), dpi=300)
    plt.close()

    ### Plot sleep quality
    fig = plt.figure() # constrained_layout=True)
    fig.suptitle("Sleep quality sub {}".format(sub))
    gs = GridSpec(1, 5, fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2:], sharey=ax1)

    ax1.plot(sub_sleep_scores['SleepLNQuality'], sub_sleep_scores['dayNumber'])
    ax1.invert_yaxis()
    ax1.set_title('Reported')

    ax2.plot(sub_sleep_scores['sleep_pred'], sub_sleep_scores['dayNumber'])
    ax2.set_title('Predicted')
    plt.setp(ax2.get_yticklabels(), visible=False)

    sns.heatmap(svd, cmap='viridis', ax=ax3,
                cbar_kws={'label': 'Typing intensity', 'fraction': 0.043})
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.savefig(dat_dir+"images/quality_sub-{}_thresholdKP.png".format(sub), dpi=300)
    plt.close()

mpl.use(backend)


# Save sleep scores (id, day number, sleep duration, sleep quality, and predicted sleep duration) to file. To be used in R.
sleep_scores_df = pd.concat(sleep_scores).reset_index(0).rename(columns={'level_0':'id'}) #, names="id")
sleep_scores_df.to_csv(join(dat_dir, "sleep_scores_thresholdKP.csv"), index=False)


# Analyse the relationships between high correlation values and subject data characteristics
zipped = list(zip(*cors.items()))
cor_subs = [int(id) for id in zipped[0]]
(cor_cors, cor_pvals) = list(zip(*[d.values() for d in zipped[1]]))

cor_df = pd.DataFrame({'id': cor_subs, 'correlation': cor_cors, 'pvalue': cor_pvals})

cor_reports = self_reports.merge(cor_df, how='outer', on='id')
cor_reports = cor_reports.loc[~np.isnan(cor_reports['correlation'])]
cor_reports = cor_reports.groupby('id').agg(
    n=pd.NamedAgg('sleepdur_yest', 'size'),
    correlation=pd.NamedAgg('correlation', 'first')
)

# %matplotlib inline
sns.relplot(data=cor_reports, x='n', y='correlation') \
    .set(
        xlabel="Number of days with self-report data",
        ylabel="Correlation betw. reported and predicted amount of sleep"
    )
plt.savefig(dat_dir+"images/cor_days_self-report_thresholdKP.png", dpi=300)


total_activities = [np.sum(np.array(res['Mactivity'])) for res in gsvd_results.values()]
cor_reports['activity'] = total_activities
sns.relplot(data=cor_reports, x='activity', y='correlation')\
    .set(
        xlabel="Total amount of typing activity",
        ylabel="Correlation betw. reported and predicted amount of sleep"
    )
plt.savefig(dat_dir+"images/cor_activity_thresholdKP.png", dpi=300)

cor_reports['n_matrix'] = [len(res['Mactivity'].index) for res in gsvd_results.values()]
sns.relplot(data=cor_reports, x='n_matrix', y='correlation')\
    .set(
        xlabel="Number of days in activity matrix",
        ylabel="Correlation betw. reported and predicted amount of sleep"
    )
plt.savefig(dat_dir+"images/cor_days_mat_thresholdKP.png", dpi=300)


mats = [np.array(res['Mactivity']) for res in gsvd_results.values()]
cor_reports['activity_per_day'] = [np.sum(mat) / mat.shape[0] for mat in mats]
sns.relplot(data=cor_reports, x='activity_per_day', y='correlation')\
    .set(
        xlabel="Average typing activity per day",
        ylabel="Correlation betw. reported and predicted amount of sleep"
    )
plt.savefig(dat_dir+"images/cor_activity_per_day_thresholdKP.png", dpi=300)

# %%
