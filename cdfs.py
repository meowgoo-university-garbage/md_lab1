import cdfs

import numpy as np
from scipy.stats import norm, erlang, poisson

def cdf_exp(l): return lambda x: cdfi_exp(x, l)
def cdfi_exp(x, l):
    if x < 0:
        return 0
    return 1 - np.exp(-l * x)

def cdf_uniform(): return lambda x: cdfi_uniform(x)
def cdfi_uniform(x):
    0 if x < 0 else 1 if x > 1 else x

def cdf_normal(mu, sigma): return lambda x: cdfi_normal(x, mu, sigma)
def cdfi_normal(x, mu, sigma):
    return norm.cdf(x, loc = mu, scale = sigma)

def cdf_erlang(k, l): return lambda x: cdfi_erlang(x, k, l)
def cdfi_erlang(x, k, l):
    return erlang.cdf(x, a = k, scale = 1 / l)

def cdf_poisson(l): return lambda x: cdfi_poisson(x, l)
def cdfi_poisson(x, l):
    return poisson.cdf(x, mu = l)
