# utility funcs
# after https://www.thegreatstatsby.com/posts/2021-03-08-ml-prospect/

import numpy as np
from scipy.optimize import minimize
import pandas as pd


# Step 2: Calculate subjective utility given lambda/rho (Equation (1) above)
# takes a vector of values and parameters and returns subjective utilities
def calc_subjective_utility(vals, lam, rho):
    if vals >= 0:
        retval = vals ** rho
    else:
        retval = (-1 * lam) * (((-1 * vals) ** rho))
    return retval


# Step 3: Calculate utility difference from vectors of gains,
# losses, and certainty (Equation (2) above)
def calc_utility_diff(su_gain, su_loss, su_cert):
    su_gain = np.array(su_gain)
    su_loss = np.array(su_loss)
    su_cert = np.array(su_cert)
    # gamble subjective utility (su) = .5 times gain subjective utility plus 
    # .5 times loss subjective utility. Then take the difference with certain
    return (.5*su_gain + .5*su_loss) - su_cert


# Step 4: Calculate the probability of accepting a gamble, 
# given a difference in subjective utility and mu (Equation (3) above)
def calc_prob_accept(gamble_cert_diff, mu):
    return (1 + np.exp(-mu * (gamble_cert_diff))) ** -1


def LL_prospect(par):
    lambda_par, rho_par, mu_par = par
    assert not np.isnan(lambda_par)
    cert_su = [calc_subjective_utility(i, lambda_par, rho_par) for i in data.cert]
    loss_su = [calc_subjective_utility(-1 * i, lambda_par, rho_par) for i in data.loss]
    gain_su = [calc_subjective_utility(i, lambda_par, rho_par) for i in data.gain]
    gamble_cert_diff = calc_utility_diff(gain_su, loss_su, cert_su)
    prob_accept = calc_prob_accept(gamble_cert_diff, mu = mu_par)
    prob_accept = np.clip(prob_accept, 1e-8, 1-1e-8)
    # calculate log likelihood on this slightly altered amount
    log_likelihood_trial = data.response.values * np.log(prob_accept) + \
        (1-data.response.values) * np.log(1-prob_accept)
    LL = -1 * np.sum(log_likelihood_trial)
    if np.isnan(LL):
        raise RuntimeError('LL is nan')
    return LL 



def fit_pt_model(df, pars0=None, bounds=None, method='L-BFGS-B'):
    if bounds is None:
        bounds = ((0, None), (0, None))
    # need to make data global so that it can be accessed by LL_prospect
    global data
    data = df
    print(data['response'].mean())
    if 'cert' not in data.columns:
        data['cert'] = 0
    if pars0 is None:
        # defaults based loosely on prior papers
        pars0 = [1.5, 0.9, 1]
    output = minimize(LL_prospect, pars0, method=method, tol=1e-8,
                bounds=bounds, options={'maxiter': 1000})
    if output.success:
        return output.x, output.fun
    else:
        print(output)
        raise RuntimeError(output.message)


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


if __name__ == '__main__':
    import pandas as pd

    alldata = pd.read_csv('https://raw.githubusercontent.com/poldrack/ResearchMethods/main/Data/NARPS/narps_behav_data.csv')
    alldata = alldata.query('condition == "equalIndifference"')
    alldata['cert'] = 0
    alldata = alldata.rename({'RT': 'rt', 'accept': 'response'}, axis=1)
    subjects = alldata['sub'].unique()
    subdata = alldata.query('sub == @subjects[0]')

    pars_est, ll = fit_pt_model(subdata,
                            bounds=((0, None), (0.1, 2), (1, 1)))
