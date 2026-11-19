import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.stats import norm

from dists import *
from cdfs import *


def calc_mu(xs):
    return sum(xs) / len(xs)

def calc_disp(xs):
    mu = calc_mu(xs)

    disp = 0
    for x in xs:
        disp += (x - mu) ** 2
    disp /= (len(xs) - 1)
    return disp

def calc_sigma(xs):
    return math.sqrt(calc_disp(xs))



def main():
    xs = distribution_1(1000)

    dists = [ distribution_1, distribution_2, distribution_3, distribution_4, distribution_5 ]
    fig, ax = plt.subplots(1, len(dists))
    for i in range(len(dists)):
        ax[i].hist(dists[i](1000), bins = 20)

    plt.show()

main()
