from abc import ABC, abstractmethod
import math
import numpy as np
import pandas as pd
import xxhash

class FrequencyOracle(ABC):

    def __init__(self, array, epsilon, domain_size=None):
        self.array = array
        self.epsilon = epsilon
        if domain_size == None:
            domain_size = max(array) + 1 #include 0
        self.domain_size = domain_size
        self.n = len(array)

    @abstractmethod
    def perturb(self):
        pass

    @abstractmethod
    def aggregate(self, perturbed_data):
        pass

    def count(self):
        perturbed = self.perturb()
        frequency = self.aggregate(perturbed)
        return sum(frequency)

    def count_in_range(self, start, end):
        perturbed = self.perturb()
        frequency = self.aggregate(perturbed)
        count = 0
        for i in range(start, end+1):
            count += frequency[i]
        # print(pd.Series(array).value_counts())
        # print(frequency)
        return count

class OLH(FrequencyOracle):
    def __init__(self, array, epsilon, domain_size=None):
        super().__init__(array, epsilon, domain_size)
        # self.g = int(round(math.exp(epsilon))) + 1
        self.g = self.domain_size
        self.p = math.exp(epsilon) / (math.exp(epsilon) + self.g - 1)
        self.q = 1.0 / (math.exp(epsilon) + self.g - 1)
        print("OLH: g =", self.g)
        print("OLH: p =", self.p)
        print("OLH: q =", self.q)

    def perturb(self):
        # TODO: perturbation can be done in parallel to simulate each user sending their message
        #  (but this step is fast)
        Y = np.zeros(self.n, dtype=int)
        counter = 0
        for i in range(self.n):
            v = self.array[i]
            x = (xxhash.xxh32(str(v), seed=i).intdigest() % self.g)
            y = x

            p_sample = np.random.random_sample()
            # the following two are equivalent
            # if p_sample > p:
            #     while not y == x:
            #         y = np.random.randint(0, g)
            if p_sample > self.p - self.q:
                # perturb
                y = np.random.randint(0, self.g)
                counter += 1

            Y[i] = y

        print("OLH: randomized {} times".format(counter))

        return Y

    def aggregate(self, perturbed_data):
        ESTIMATE_DIST = np.zeros(self.domain_size)
        # TODO: too slow
        for i in range(self.n):
            for v in range(self.domain_size):
                if perturbed_data[i] == (xxhash.xxh32(str(v), seed=i).intdigest() % self.g):
                    ESTIMATE_DIST[v] += 1
        a = 1.0 * self.g / (self.p * self.g - 1)
        b = 1.0 * self.n / (self.p * self.g - 1)
        ESTIMATE_DIST = a * ESTIMATE_DIST - b
        return ESTIMATE_DIST


class RR(FrequencyOracle):
    def __init__(self, array, epsilon, domain_size=None):
        super().__init__(array, epsilon, domain_size)
        self.p = math.exp(self.epsilon) / (math.exp(self.epsilon) + self.domain_size - 1)
        self.q = 1 / (math.exp(self.epsilon) + self.domain_size - 1)
        print("RR: p = ", self.p)
        print("RR: q = ", self.q)
        # self.var = self.q * (1 - self.q) / (self.p - self.q) ** 2

    def perturb(self):
        # TODO: perturbation can be done in parallel to simulate each user sending their message
        #  (but this step is fast)
        perturbed_data = np.zeros(self.n, dtype=int)
        counter = 0
        for i in range(self.n):
            y = self.array[i]
            p_sample = np.random.random_sample()

            if p_sample > self.p:
                # perturb
                y = np.random.randint(0, self.domain_size)
                counter += 1

            perturbed_data[i] = y

        print("RR: randomized {} times".format(counter))

        # print(self.array[:100])
        # print(perturbed_data[:100])

        return perturbed_data

    def aggregate(self, perturbed_data):
        ESTIMATE_DIST = np.zeros(self.domain_size)
        unique, counts = np.unique(perturbed_data, return_counts=True)

        for i in range(len(unique)):
            ESTIMATE_DIST[unique[i]] = counts[i]

        return ESTIMATE_DIST

