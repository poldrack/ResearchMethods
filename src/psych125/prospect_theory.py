# utility funcs
# after https://www.thegreatstatsby.com/posts/2021-03-08-ml-prospect/

import numpy as np
from scipy.optimize import minimize


# Step 2: Calculate subjective utility given lambda/rho (Equation (1) above)
# takes a vector of values and parameters and returns subjective utilities
def calc_subjective_utility(vals, lam, rho):
    if vals >= 0:
        return vals ** rho
    else:
        return (-1 * lam) * ((-vals ** rho))


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
  
    cert_su = [calc_subjective_utility(i, lambda_par, rho_par) for i in data.cert]
    loss_su = [calc_subjective_utility(-1 * i, lambda_par, rho_par) for i in data.loss]
    gain_su = [calc_subjective_utility(i, lambda_par, rho_par) for i in data.gain]
    gamble_cert_diff = calc_utility_diff(gain_su, loss_su, cert_su)
    prob_accept = calc_prob_accept(gamble_cert_diff, mu = mu_par)
    prob_accept = np.clip(prob_accept, 1e-8, 1-1e-8)
    # calculate log likelihood on this slightly altered amount
    log_likelihood_trial = data.response.values * np.log(prob_accept) + \
        (1-data.response.values) * np.log(1-prob_accept)

    return -1 * np.sum(log_likelihood_trial)



def fit_pt_model(df, pars0=None, bounds=None, method='L-BFGS-B'):
    if bounds is None:
        bounds = ((0, None), (0, None), (0, None))
    # need to make data global so that it can be accessed by LL_prospect
    global data
    data = df
    if 'cert' not in data.columns:
        data['cert'] = 0
    if pars0 is None:
        # defaults based on Sokol-Hessner paper
        pars0 = [1, 1, 1]
    output = minimize(LL_prospect, pars0, method=method, tol=1e-8,
                bounds=bounds)
    if output.success:
        return output.x, output.fun
    else:
        raise RuntimeError(output.message)
