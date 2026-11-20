import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time

from scipy.stats import norm, chi2, poisson
from scipy.optimize import brentq

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


def chi_sqr(xs, cdf, bins, pc):
    xs = np.asarray(xs, dtype = float)
    n = xs.size

    probabilities = np.arange(1, bins, dtype = float) / bins
    lhs, rhs = -1.0, +1.0

    for i in range(1024):
        if cdf(lhs) <= probabilities[0] and probabilities[-1] <= cdf(rhs):
            break

        if cdf(lhs) > probabilities[0]:
            lhs *= 2

        if probabilities[-1] > cdf(rhs):
            rhs *= 2

    edges_inner = np.array([ brentq(lambda x: cdf(x) - p, lhs, rhs) for p in probabilities ])
    edges = np.concatenate(([ -np.inf ], edges_inner, [ +np.inf ]))

    edges_cdf = np.array([ 0.0 ] + [ cdf(x) for x in edges_inner ] + [ 1.0 ])

    observed, _ = np.histogram(xs, bins = edges)
    expected = n * np.diff(edges_cdf)

    merged_observed = []
    merged_expected = []
    i = 0

    while i < len(expected):
        cum_observed = observed[i]
        cum_expected = expected[i]
        i += 1

        while cum_expected < 5 and i < len(expected):
            cum_observed += observed[i]
            cum_expected += expected[i]
            i += 1

        merged_observed.append(cum_observed)
        merged_expected.append(cum_expected)

    if len(merged_expected) > 1 and merged_expected[-1] < 5:
        merged_observed[-2] += merged_observed.pop()
        merged_expected[-2] += merged_expected.pop()

    observed = np.asarray(merged_observed)
    expected = np.asarray(merged_expected)

    dof = len(expected) - 1 - pc
    result = np.sum((observed - expected) ** 2 / expected)
    return float(chi2.sf(result, dof))


def chi_sqr_poisson(xs, mu, pc):
    xs = np.asarray(xs, dtype = float)
    n = xs.size

    max_k = int(poisson.ppf(0.999, mu = mu)) + 1
    lower_bounds = np.arange(max_k + 1, dtype = float)

    category_indexes = np.searchsorted(lower_bounds, xs, side = "right") - 1
    observed = np.bincount(category_indexes, minlength = len(lower_bounds))

    cdf_before = poisson.cdf(lower_bounds - 1, mu = mu)
    sf_before  = poisson.sf (lower_bounds - 1, mu = mu)
    probabilities = np.empty(len(lower_bounds), dtype = float)

    for i in range(len(lower_bounds) - 1):
        if cdf_before[i] < 0.5:
            probabilities[i] = cdf_before[i + 1] - cdf_before[i]
        else:
            probabilities[i] = sf_before[i] - sf_before[i + 1]

    probabilities[-1] = sf_before[-1]
    expected = n * probabilities

    merged_observed = []
    merged_expected = []
    i = 0

    while i < len(expected):
        cum_observed = observed[i]
        cum_expected = expected[i]
        i += 1

        while cum_expected < 5 and i < len(expected):
            cum_observed += observed[i]
            cum_expected += expected[i]
            i += 1

        merged_observed.append(cum_observed)
        merged_expected.append(cum_expected)

    if len(merged_expected) > 1 and merged_expected[-1] < 5:
        merged_observed[-2] += merged_observed.pop()
        merged_expected[-2] += merged_expected.pop()

    observed = np.asarray(merged_observed)
    expected = np.asarray(merged_expected)

    dof = len(expected) - 1 - pc
    result = np.sum((observed - expected) ** 2 / expected)
    return float(chi2.sf(result, dof))





def main():

    dists = [ distribution_1, distribution_2, distribution_3, distribution_4, distribution_5 ]

    ANALYZE_EXP = False
    ANALYZE_UNI = False
    ANALYZE_NOR = False
    ANALYZE_ERL = False
    ANALYZE_POI = False
    BENCH_LIN   = False
    BENCH_LOG   = False

    seqs = []

    if True:
        fig = plt.figure()
        fig.suptitle("Generating distributions")

        for i in range(len(dists)):
            xs = dists[i](1000)
            seqs.append(xs)
            ax = fig.add_subplot(1, 5, 1 + i)
            ax.hist(xs, bins = 20)

            print(f"Distribution #{i}. MU = {calc_mu(xs)}, DISPERSION = {calc_disp(xs)}, SIGMA = {calc_sigma(xs)}")

        plt.show()

    print(len(seqs))

    if ANALYZE_EXP:
        fig = plt.figure()
        fig.suptitle("Analyzing for exponential distributions")

        ls = np.arange(0.05, 0.9, 0.01)

        for di in range(len(seqs)):
            seq = seqs[di]

            X2 = np.zeros(len(ls))
            for i in range(len(ls)):
                X2[i] = chi_sqr(seq, cdf_exp(ls[i]), 20, 0)

            ax = fig.add_subplot(2, 5, 6 + di)
            ax.plot(ls, X2)
            ax.axhline(y = 0.05, color = "red")

            cx = fig.add_subplot(2, 5, 1 + di)
            cx.hist(seq, bins = 20)

        plt.show()

    if ANALYZE_UNI:
        fig = plt.figure()
        fig.suptitle("Analyzing for uniform distributions")

        aS = np.arange(-1, 0.5, 0.01)
        bs = np.arange(0.6, 2, 0.01)
        A, B = np.meshgrid(aS, bs)

        for di in range(len(seqs)):
            seq = seqs[di]

            X2 = np.zeros(A.shape)

            for i in range(len(bs)):
                for j in range(len(aS)):
                    X2[i, j] = chi_sqr(seq, cdf_uniform(A[i, j], B[i, j]), 20, 0)

            ax = fig.add_subplot(2, 5, 6 + di, projection = "3d")
            ax.plot_surface(A, B, X2, cmap = "viridis")

            cx = fig.add_subplot(2, 5, 1 + di)
            cx.hist(seq, bins = 20)

        plt.show()

    if ANALYZE_NOR:
        fig = plt.figure()
        fig.suptitle("Analyzing for normal distributions")

        ms = np.arange(0.35, 0.65, 0.01)
        ss = np.arange(0.05, 0.15, 0.01)
        M, S = np.meshgrid(ms, ss)

        for di in range(len(seqs)):
            seq = seqs[di]

            X2 = np.zeros(M.shape)

            for i in range(len(ss)):
                for j in range(len(ms)):
                    X2[i, j] = chi_sqr(seq, cdf_normal(M[i, j], S[i, j]), 20, 0)

            ax = fig.add_subplot(2, 5, 6 + di, projection = "3d")
            ax.plot_surface(M, S, X2, cmap = "viridis")

            cx = fig.add_subplot(2, 5, 1 + di)
            cx.hist(seq, bins = 20)

        plt.show()

    if ANALYZE_ERL:
        fig = plt.figure()
        fig.suptitle("Analyzing for erlang distributions")

        ks = np.array([ 1, 2, 3, 4, 5, 6 ], dtype = "float")
        ls = np.arange(0.2, 0.5, 0.01)
        K, L = np.meshgrid(ks, ls)

        for di in range(len(seqs)):
            seq = seqs[di]

            X2 = np.zeros(K.shape)

            for i in range(len(ls)):
                for j in range(len(ks)):
                    X2[i, j] = chi_sqr(seq, cdf_erlang(K[i, j], L[i, j]), 20, 0)

            ax = fig.add_subplot(2, 5, 6 + di, projection = "3d")
            ax.plot_surface(K, L, X2, cmap = "viridis")

            cx = fig.add_subplot(2, 5, 1 + di)
            cx.hist(seq, bins = 20)

        plt.show()

    if ANALYZE_POI:
        fig = plt.figure()
        fig.suptitle("Analyzing for poisson distributions")

        ls = np.arange(1.5, 4.5, 0.01)

        for di in range(len(seqs)):
            seq = seqs[di]

            X2 = np.zeros(len(ls))
            for i in range(len(ls)):
                X2[i] = chi_sqr_poisson(seq, ls[i], 0)

            ax = fig.add_subplot(2, 5, 6 + di)
            ax.plot(ls, X2)
            ax.axhline(y = 0.05, color = "red")

            cx = fig.add_subplot(2, 5, 1 + di)
            cx.hist(seq, bins = 20)

        plt.show()



    if BENCH_LIN:
        bench_counts = []

        for i in range(len(dists)):
            dist = dists[i]

            lo = 0
            hi = 20000000
            c = 0
            t = 0

            print(f"Benchmarking dist {i}")

            while True:
                c = int((hi + lo) / 2)
                time_start = time.process_time()
                _ = dist(c)
                time_end = time.process_time()
                t = time_end - time_start

                print(f"\tN = {c}, time = {t}")

                if math.fabs(1 - t) < 0.01:
                    break
                
                if t > 1:
                    hi = c
                else:
                    lo = c

            print(f"Dist {i}, N = {c}, time = {t}")
            bench_counts.append(c)

        fig = plt.figure()
        fig.suptitle("Benchmarking")
        for i in range(len(dists)):
            ns = [ 1, 2, 3, 4, 5, 6, 7 ]
            times = []

            for n in ns:
                time_start = time.process_time()
                _ = dists[i](n * bench_counts[i])
                time_end = time.process_time()
                t = time_end - time_start
                times.append(t)

            ax = fig.add_subplot(1, 5, 1 + i)
            ax.plot(ns, times)

        plt.show()


    if BENCH_LOG:
        fig = plt.figure()
        fig.suptitle("Benchmarking (log)")

        for i in range(len(dists)):
            dist = dists[i]

            times = []
            ns = []

            n = 10
            for j in range(6):
                time_start = time.process_time()
                _ = dists[i](n)
                time_end = time.process_time()
                t = time_end - time_start

                times.append(t)
                ns.append(j)

                n *= 10

            ax = fig.add_subplot(1, 5, 1 + i)
            ax.plot(ns, times)

        plt.show()







main()
