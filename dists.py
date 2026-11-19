import dists

import random
import math

def distribution_1(n):
    xs = []
    for i in range(0, n):
        u = random.random()
        x = -2 * math.log(u)
        xs.append(x)
    return xs

def distribution_2(n):
    a = 25214903917
    c = 11
    m = 2 ** 48
    seed = (m - 1) / 2
    xs = []

    x = seed
    def mk(x):
        return float(x) / (m - 1)

    xs.append(mk(x))
    for i in range(1, n):
        x = (a * x + c) % m
        xs.append(mk(x))

    return xs

def distribution_3(n):
    sigma = 0.1
    a = 0.5
    k = 12
    m = 6

    xs = []

    for i in range(0, n):
        theta = 0
        for j in range(0, k):
            u = random.random()
            theta += u
        theta -= m
        x = sigma * theta + a
        xs.append(x)

    return xs

def distribution_4(n):
    xs = []

    for i in range(0, n):
        x = 0
        for j in range(0, 4):
            u = random.random()
            x += math.log(u)
        x *= -3
        xs.append(x)

    return xs

def distribution_5(n):
    xs = []

    L = 1 / (math.e ** 3)

    for i in range(0, n):
        k = 0
        p = 1

        while True:
            u = random.random()

            k += 1
            p *= u

            if not p > L:
                break

        xs.append(k - 1)

    return xs
