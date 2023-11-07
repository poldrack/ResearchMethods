# %% markdown
# ### NARPS behavioral analysis

# %% codecell

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from prospect_theory import (
    calc_subjective_utility, calc_utility_diff, calc_prob_accept,
    fit_pt_model
)
from firthlogist import FirthLogisticRegression
from keras.models import Sequential
from keras.layers import Dense
from keras_visualizer import visualizer

plt.rcParams.update({'font.size': 16})
plt.rcParams['figure.autolayout'] = True

# %% codecell

# load data
if os.path.exists('/home/poldrack/Dropbox/code/ResearchMethods/Data/NARPS/narps_behav_data.csv'):
    alldata = pd.read_csv('/home/poldrack/Dropbox/code/ResearchMethods/Data/NARPS/narps_behav_data.csv')
else:
    alldata = pd.read_csv('https://raw.githubusercontent.com/poldrack/ResearchMethods/main/Data/NARPS/narps_behav_data.csv')
alldata = alldata.query('condition == "equalIndifference"')
alldata['cert'] = 0
alldata = alldata.rename({'RT': 'rt', 'accept': 'response'}, axis=1)
subjects = alldata['sub'].unique()

print(f'number of subjects = {len(subjects)}')




# %%
# plot acceptance as a function of gain/loss across all subjects

# acceptance by gain

sns.lineplot(data=alldata, x='gain', y='response', legend=False)
plt.savefig('narps_gain.png')

# %%


# acceptance by gain

sns.lineplot(data=alldata, x='loss', y='response', legend=False)
plt.savefig('narps_loss.png')

# %%
# heatmap by gain/loss
hmap_data = alldata.groupby(['gain', 'loss'])['response'].mean().unstack()
sns.heatmap(hmap_data, cmap='coolwarm')

plt.savefig('narps_heatmap.png')


# %%
# use regularlized logistic regression to estimate gain/loss parameters
# since some subjects have perfect separation


lr_params = []
fl = {}
subdata = {}

for sub in subjects:
    subdata[sub] = alldata.query(f'sub == {sub}')
    try:
        fl[sub] = FirthLogisticRegression(max_iter=1000, max_halfstep=1000, 
                                     skip_pvals=True, skip_ci=True)
        fl[sub].fit(subdata[sub][['gain', 'loss']], subdata[sub]['response'])
        accuracy = np.mean(
            fl[sub].predict(subdata[sub][['gain', 'loss']]) == subdata[sub]['response'])
        lr_params.append([
            sub, accuracy, 
            fl[sub].coef_[0], fl[sub].coef_[1], fl[sub].intercept_])
    except np.linalg.LinAlgError :
        print(f'could not fit subject {sub}')
        continue
lr_params_df = pd.DataFrame(lr_params, columns=['sub', 'accuracy', 'gain', 'loss', 'intercept'])

print('median accuracy = ', np.median(lr_params_df.accuracy))

# %%

plt.figure(figsize=(10,5))
plt.subplot(1,3,1)
sns.swarmplot(lr_params_df.gain)
plt.subplot(1,3,2)
sns.swarmplot(lr_params_df.loss)
plt.subplot(1,3,3)
sns.swarmplot(lr_params_df.accuracy)
plt.tight_layout()

plt.savefig('narps_logistic_params.png')

# %%
# plot logistic response for a given subject

sub = 9
plot_data = subdata[sub].copy()
plot_data['response_fix'] = None
offset = .01
coord_ctr = {}
for i in plot_data.index:
    coords = (plot_data.loc[i, 'gain'], plot_data.loc[i, 'response'])
    if coords in coord_ctr:
        coord_ctr[coords] += 1
    else:
        coord_ctr[coords] = 1
    if plot_data.loc[i, 'response'] == 1:
        plot_data.loc[i, 'response_fix'] = 1 - offset * coord_ctr[coords]
    else:
        plot_data.loc[i, 'response_fix'] =  offset * coord_ctr[coords]

plot_data['pred_lr'] = fl[sub].predict_proba(plot_data[['gain', 'loss']])[:,1]
sns.scatterplot(data=plot_data, x='gain', y='response_fix', 
                legend=False)
sns.lineplot(data=plot_data, 
             x='gain', 
             y='pred_lr',
             legend=False, ci=None, color='black')
plt.tight_layout()

plt.savefig(f'narps_logistic_sub{sub}.png')

# %%

sns.heatmap(plot_data.groupby(['gain', 'loss'])['pred_lr'].mean().unstack(),
            cmap='coolwarm')
plt.tight_layout()
plt.savefig(f'narps_logistic_heatmap_sub{sub}.png')


# %%

# fit random guessing with bias model

def test_guessing_model(df, n_iter=1000):
    bias = np.mean(df['response'])
    acc = []
    for i in range(n_iter):
        guesses = np.random.binomial(1, bias, size=df.shape[0])
        acc.append(np.mean(np.array(guesses) == df['response']))
    return np.mean(acc)

sub_guessing_pars = {}
for sub in subjects:
    bias = np.mean(subdata[sub]['response'])
    sub_guessing_pars[sub] = [bias, test_guessing_model(subdata[sub])]
    
sub_guessing_pars_df = pd.DataFrame(sub_guessing_pars).T
sub_guessing_pars_df.columns = ['bias', 'accuracy']
print(f'median accuracy (guessing model with bias) = {np.median(sub_guessing_pars_df.accuracy):.02}')


# %%
# plot accuracy vs bias
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.swarmplot(sub_guessing_pars_df['accuracy'])
plt.title(f'accuracy, median = {np.median(sub_guessing_pars_df.accuracy):.03}')

plt.subplot(1,2,2)
sns.scatterplot(data=sub_guessing_pars_df, x='bias', y='accuracy')
plt.tight_layout()
plt.savefig('guessing_model_accuracy.png')


# %%

# plot sigmoid

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5, 5, .1)
y = sigmoid(x)
plt.plot(x, y)
plt.axhline(0.5, color='black', linestyle='--', alpha=0.5)
plt.axvline(0, color='black')
plt.ylabel('probability of acceptance')
plt.xlabel('v (gain - loss)')
plt.savefig('eu_sigmoid.png')

# %%

# plot utility function for EU model

values = np.arange(0, 40, .1)
y = values ** 0.8

plt.plot(values, y)
plt.plot([0,40],[0,40], color='black', linestyle='--', alpha=0.5)
plt.xlabel('face value')
plt.ylabel('subjective value')
plt.savefig('narps_EU_value_function.png')


# %%
# estimate EU model parameters
# equivalent to prospect theory with lambda = 1

sub_pars_EU = {}
for sub in subjects:
    try:
      pars_est, ll = fit_pt_model(subdata[sub],
                              bounds=((1, 1), (0.1, 2), (1, 1)))
    except RuntimeError:
      print(f'could not fit subject {sub}')
      continue
    sub_pars_EU[sub] = np.hstack((pars_est, ll))
    print(f'subject {sub} lambda = {pars_est[0]}, rho = {pars_est[1]}, mu = {pars_est[2]}')

sub_pars_EU_df = pd.DataFrame(sub_pars_EU).T
sub_pars_EU_df.columns = ['lambda', 'rho', 'mu', 'll']

plt.figure(figsize=(3,5))
sns.swarmplot(sub_pars_EU_df.rho)
plt.title(f'rho, median = {np.median(sub_pars_EU_df["rho"]):.02}')
plt.tight_layout()

plt.savefig('narps_EU_params.png')

# %%

pred_output_EU_df = get_predicted_output(sub_pars_EU, subdata)
sns.scatterplot(data=pred_output_EU_df, x='pred_accept', y='accept', 
                legend=False)
plt.plot([0,256],[0,256], color='black')
plt.savefig('narps_PT_pred_vs_actual.png')

print('median accuracy = ', np.median(pred_output_EU_df.predacc))

# %%
# estimate prospect theory parameters

sub_pars_PT = {}
for sub in subjects:
    try:
      pars_est, ll = fit_pt_model(subdata[sub],
                              bounds=((0, None), (0.1, 2), (1, 1)))
    except RuntimeError:
      print(f'could not fit subject {sub}')
      continue
    sub_pars_PT[sub] = np.hstack((pars_est, ll))
    print(f'subject {sub} lambda = {pars_est[0]}, rho = {pars_est[1]}, mu = {pars_est[2]}')

sub_pars_PT_df = pd.DataFrame(sub_pars_PT).T
sub_pars_PT_df.columns = ['lambda', 'rho', 'mu', ll]

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.swarmplot(sub_pars_df['lambda'])
plt.title(f'loss aversion (lambda), median = {np.median(sub_pars_PT_df["lambda"]):.03}')
plt.subplot(1,2,2)
sns.swarmplot(sub_pars_df.rho)
plt.title(f'curvature (rho), median = {np.median(sub_pars_PT_df["rho"]):.02}')

plt.tight_layout()

plt.savefig('narps_PT_params.png')

# %%
# plot prospect theory value function for median parameters

values = np.arange(-40, 40, .1)
su = [calc_subjective_utility(i, np.median(sub_pars_PT_df['lambda']), 
                                np.median(sub_pars_PT_df['rho'])) for i in values]
plt.plot(values, su)
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.ylabel('subjective value')
plt.xlabel('face value')
plt.savefig('narps_PT_value_function.png')

# %%

# plot predicted # of accepted gambles vs actual for each subject



def get_predicted_output(sub_pars, subdata):
    pred_output = []
    for sub, pars in sub_pars.items():
        cert_su = [calc_subjective_utility(i, pars[0], pars[1]) for i in subdata[sub].cert]
        loss_su = [calc_subjective_utility(-1 * i, pars[0], pars[1]) for i in subdata[sub].loss]
        gain_su = [calc_subjective_utility(i, pars[0], pars[1]) for i in subdata[sub].gain]
        gamble_cert_diff = calc_utility_diff(gain_su, loss_su, cert_su)
        prob_accept = calc_prob_accept(gamble_cert_diff, mu = pars[2])
        n_pred_accepted = np.sum(prob_accept > 0.5)
        n_accepted = np.sum(subdata[sub].response)
        pred_acc = np.mean((prob_accept > 0.5) == subdata[sub].response)
        pred_output.append([n_pred_accepted, n_accepted, pred_acc, sub, pars[0], pars[1], pars[2]])

    return pd.DataFrame(pred_output, columns=['pred_accept', 'accept', 'predacc', 'sub', 'lambda', 'rho', 'mu']) 

pred_output_PT_df = get_predicted_output(sub_pars_PT, subdata)
sns.scatterplot(data=pred_output_PT_df, x='pred_accept', y='accept', legend=False)
plt.plot([0,256],[0,256], color='black')
plt.savefig('narps_PT_pred_vs_actual.png')

print('median accuracy = ', np.median(pred_output_PT_df.predacc))

# %% markdown

## Neural network model

# %%
# fit a neural network
def fit_nn_model(df, verbose=0):

    # define model
    model = Sequential()
    model.add(Dense(3, activation='relu', input_dim=2))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])

    # fit model
    model.fit(df[['gain', 'loss']], df['response'], 
              epochs=100, batch_size=10, verbose=verbose)


    loss, accuracy = model.evaluate(df[['gain', 'loss']], df['response'])
    print('Train model loss:', loss)
    print('Train model accuracy:', accuracy)

    return model, accuracy

nn_acc = {}
pred_output_NN = []
for sub in subjects:
    model, nn_acc[sub] = fit_nn_model(subdata[sub])
    pred = model.predict(subdata[sub][['gain', 'loss']])
    pred_output_NN.append([nn_acc[sub], np.sum(pred > 0.5)])

# %%

visualizer(model, file_format='png', view=True)
plt.savefig('narps_nn.png')

# %%
