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

class OLH(FrequencyOracle):
    def __init__(self, array, epsilon, domain_size=None):
        super().__init__(array, epsilon, domain_size)
        self.g = int(round(math.exp(epsilon))) + 1
        self.p = math.exp(epsilon) / (math.exp(epsilon) + self.g - 1)
        self.q = 1.0 / (math.exp(epsilon) + self.g - 1)

    def perturb(self):
        Y = np.zeros(self.n)
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
            Y[i] = y
        return Y

    def aggregate(self, perturbed_data):
        ESTIMATE_DIST = np.zeros(self.domain_size)
        for i in range(self.n):
            for v in range(self.domain_size):
                if perturbed_data[i] == (xxhash.xxh32(str(v), seed=i).intdigest() % self.g):
                    ESTIMATE_DIST[v] += 1
        a = 1.0 * self.g / (self.p * self.g - 1)
        b = 1.0 * self.n / (self.p * self.g - 1)
        ESTIMATE_DIST = a * ESTIMATE_DIST - b
        return ESTIMATE_DIST

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
