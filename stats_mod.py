import numpy as np
#from pba import Interval, Pbox
import matplotlib.pyplot as plt

#import pymc3 as pm
from scipy import stats
from scipy.optimize import curve_fit


def list_parameters(distribution):
    """List parameters for stats.distribution.
    # Arguments
        distribution: a string or stats distribution object.
    # Returns
        A list of distribution parameter strings.
    """
    if isinstance(distribution, str):
        distribution = getattr(pm, distribution)
    if distribution:
        parameters = distribution.dist()._distr_parameters_for_repr()
        #[name.strip() for name in distribution.shapes.split(',')]
    # else:
    #     parameters = []
    # if distribution.name in stats._discrete_distns._distn_names:
    #     parameters += ['loc']
    # elif distribution.name in stats._continuous_distns._distn_names:
    #     parameters += ['loc', 'scale']
    else:
        sys.exit("Distribution name not found in discrete or continuous lists.")
    return parameters


def MLE(data, distribution='Normal', **kwargs):
    """ Maximum Likelihood Estimator"""
    mom = getattr(stats.distributions, distribution).fit(data, **kwargs)
    return mom


def MOM(data, distribution='Normal'):
    """ Method of Moments Estimator"""
    return MLE(data, distribution=distribution, method='MM')


def LSQ(X, Y, p0, distribution='Normal', return_cov=False, maxfev=600):
    """ Least Squares regression estimator """
    # assert len(p0) == len(list_parameters(distribution)
    #                       ), 'Warning: p0 must be same length as distribution parameters'

    def CDF(x, *mom):
        return cumulative_density_function(x, mom, distribution=distribution, Fast=True)

    fit_params, fit_cov = curve_fit(
        CDF, xdata=X, ydata=Y, p0=p0, maxfev=maxfev)
    if return_cov:
        return fit_params, fit_cov
    return fit_params


def density_function(x, mom, distribution='Normal'):
    """ Evaluate PDF at X given the parameters and distribution name """
    return np.exp(log_density_function(x, mom, distribution='Normal'))


def cumulative_density_function(x, mom, distribution='Normal', Fast=False):
    """ Evaluate CDF at X given the parameters and distribution name """
    if Fast:
        assert hasattr(
            stats, distribution), 'WARNING: Choosen distribution not found, check you are using scipy.stats name, not pymc3 name'
        return getattr(stats, distribution).cdf(x, *mom)
    else:
        return np.exp(getattr(pm, distribution).dist(*mom).logcdf(x).eval())


def log_density_function(x, mom, distribution='Normal'):
    """ Evaluate logLikelihood at X given the parameters and distribution name """
    return getattr(pm, distribution).dist(*mom).logp(x).eval()


def sampler_function(N, mom, distribution='Normal'):
    """ Generate N samples of the distribution given the parameters and distribution name """
    return getattr(pm, distribution).dist(*mom).random(size=N)


def ecdf(x):
    xs = np.sort(x)
    #xs = np.append(xs,xs[-1])
    n = xs.size
    y = np.linspace(0, 1, n)
    #np.arange(1, n+1) / n
    #xs = np.append(xs[0],xs)
    #ps =
    return [y, xs]

def plot_ecdf(x, ax = None, return_ax = True, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1,figsize=(10,10))    
    P, X = ecdf(x)
    ax.step(X, P, **kwargs)    
    if return_ax:
        return ax

def sort_interval_list(int_list: list, by_lower_bound=True, return_index=False):
    """
    Sort a list of pba.Interval
    """
    if by_lower_bound:
        LB = [e.left for e in int_list]
        id_lb = np.argsort(LB).tolist()
    else:
        RB = [e.right for e in int_list]
        id_lb = np.argsort(RB).tolist()
    int_list = [int_list[i] for i in id_lb]
    if return_index:
        return int_list, id_lb
    return int_list



def s_ln(x, data):
    n = np.size(data)
    l = np.sum(data <= x)
    return l/n



def smirnov_critical_value(alpha, n):
    # a = np.array([0.20,0.15,0.10,0.05,0.025,0.01,0.005,0.001])
    # c_a = np.array([1.073,1.138,1.224,1.358,1.48,1.628,1.731,1.949])
    #
    # if any(np.isclose(0.0049,a,2e-2)):
    # c_alpha = c_a[np.where(np.isclose(0.0049,a,2e-2))[0]]
    # else:
    c_alpha = np.sqrt(-np.log(alpha/2)*(1/2))
    return (1/np.sqrt(n))*c_alpha


def confidence_limits_distribution(x, alpha, interval=False, n_interp=100, plot=False, x_lim=[-10, 10], label='', savefig=[]):
    """
    The confidence limits of F(x) is an inversion of the well known KS-test.
    KS test is usually used to test whether a given F(x) is the underlying probability distribution of Fn(x).

    See      : Experimental uncertainty estimation and statistics for data having interval uncertainty. Ferson et al.
               for this implimentation. Here interval valued array is valid.
    """

    if not interval:
        data = np.zeros([2, np.size(x)])
        data[0] = x
        data[1] = x
    else:
        data = x

    x_i = np.linspace(np.min(data[0])-abs(x_lim[0]),
                      np.max(data[1])+x_lim[1], n_interp)

    N = np.size(data[0])

    if N < 50:
        print('Dont trust me! I really struggle with small sample sizes\n')
        print('TO DO: Impliment the statistical conversion table for Z score with lower sample size')

    def b_l(x): return min(
        1, s_ln(x, data[0])+smirnov_critical_value(round((1-alpha)/2, 3), N))
    def b_r(x): return max(
        0, s_ln(x, data[1])-smirnov_critical_value(round((1-alpha)/2, 3), N))

    L = []
    R = []
    for i, xi in enumerate(x_i):
        L.append(b_l(xi))
        R.append(b_r(xi))

    if plot:
        fig = plt.figure(figsize=(6, 6))
        pl, xl = ecdf(data[0])
        pr, xr = ecdf(data[1])
        plt.step(xl, pl, color='blue', label='data', alpha=0.3)
        plt.step(xr, pr, color='blue', alpha=0.7)
        plt.step(x_i, L, color='red', label='data', alpha=0.7)
        plt.step(x_i, R, color='red', alpha=0.7,
                 label='KS confidence limits {}%'.format(alpha))
        plt.xlabel(label)
        plt.xlim(x_lim)
        plt.ylabel('P(x)')
        if savefig:
            fig.savefig(savefig)
    return L, R, x_i


def interpCDF_2(xd, yd, pvalue):
    """
    %INTERPCDF Summary of this function goes here
    %   Detailed explanation goes here
    %
    % .
    % . by The Liverpool Git Pushers
    """
    # [yd,xd]=ecdf(data)
    beforr = np.zeros(len(yd))
    beforr = np.diff(pvalue <= yd) == 1
    beforrr = np.append(0, beforr[:])
    if pvalue == 0:
        xvalue = xd[1]
    else:
        xvalue = xd[beforrr == 1]

    outputArg1 = xvalue

    return outputArg1


def area_metric_robust(D1, D2):
    """
    #   Returns the stochastic distance between two data
    #   sets, using the area metric (horizontal integral between their ecdfs)
    #
    #   As described in: "Validation of imprecise probability models" by S.
    #   Ferson et al. Computes the area between two ECDFs
    #
    #                  By Marco De Angelis, (adapted for python Dominic Calleja)
    #                     University of Liverpool by The Liverpool Git Pushers
    """

    if np.size(D1) > np.size(D2):
        d1 = D2
        d2 = D1
    else:
        d1 = D1
        d2 = D2      # D1 will always be the larger data set

    Pxs, xs = ecdf(d1)            # Compute the ecdf of the data sets
    Pys, ys = ecdf(d2)

    Pys_eqx = Pxs
    Pys_pure = Pys[0:-1]  # this does not work with a single datum
    Pall = np.sort(np.append(Pys_eqx, Pys_pure))

    ys_eq_all = np.zeros(len(Pall))
    ys_eq_all[0] = ys[0]
    ys_eq_all[-1] = ys[-1]
    for k in range(1, len(Pall)-1):
        ys_eq_all[k] = interpCDF_2(ys, Pys, Pall[k])

    xs_eq_all = np.zeros(len(Pall))
    xs_eq_all[0] = xs[0]
    xs_eq_all[-1] = xs[-1]
    for k in range(1, len(Pall)-1):
        xs_eq_all[k] = interpCDF_2(xs, Pxs, Pall[k])

    diff_all_s = abs(ys_eq_all-xs_eq_all)
    diff_all_s = diff_all_s[range(1, len(diff_all_s))]
    diff_all_p = np.diff(Pall)
    area = np.matrix(diff_all_p) * np.matrix(diff_all_s).T

    return np.array(area)[0]

def normalize(D):
    return np.abs((D - np.min(D)) / (np.min(D) - np.max(D)))


def standardize(D):
    return (D - np.mean(D)) / np.std(D)
