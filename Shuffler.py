import math
from abc import ABC, abstractmethod

import numpy as np


class Shuffler(ABC):

    def __init__(self, array, epsilon_c, delta, domain_size=None):
        self.array = array
        self.epsilon_c = epsilon_c
        self.delta = delta
        if domain_size == None:
            domain_size = max(array) + 1 #include 0
        self.domain_size = domain_size
        self.n = len(array)

    @abstractmethod
    def local_randomizer(self):
        pass

    @abstractmethod
    def shuffler(self, randomized_data):
        pass

    @abstractmethod
    def analyzer(self, shuffled_data):
        pass

    def count(self):
        randomized_data = self.local_randomizer()
        shuffled_data = self.shuffler(randomized_data)
        frequency = self.analyzer(shuffled_data)
        return sum(frequency)

    def count_in_range(self, start, end):
        randomized_data = self.local_randomizer()
        shuffled_data = self.shuffler(randomized_data)
        frequency = self.analyzer(shuffled_data)
        count = 0
        for i in range(start, end+1):
            count += frequency[i]
        # print(pd.Series(array).value_counts())
        # print(frequency)
        return count

class PrivacyBlanket(Shuffler):
    def __init__(self, array, epsilon_c, delta, domain_size=None):
        super().__init__(array, epsilon_c, delta, domain_size)

        # condition: sqrt(14 ln(2/delta) d / (n-1) <= epsilon_c <= 1
        if epsilon_c < math.sqrt(14 * math.log(2 / delta) * self.domain_size / (self.n - 1)) or epsilon_c > 1:
            print("epsilon_c does not match condition")
            exit(1)

            # epsilon_l = ln((epsilon_c^2 (n-1) / (14 ln(2/delta))) + 1 - d)
        exp_epsilon_l = self.epsilon_c**2 * (self.n - 1) / (14 * math.log(2/self.delta)) + 1 - self.domain_size
        # gamma = d / (exp(epsilon_l) + d - 1)
        self.gamma = self.domain_size / (exp_epsilon_l + self.domain_size - 1)
        print("PrivacyBlanket: gamma =", str(self.gamma))

    def local_randomizer(self):
        randomized_data = np.zeros(self.n, dtype=int)
        counter = 0
        for i in range(self.n):
            y = self.array[i]
            p_sample = np.random.random_sample()

            if p_sample <= self.gamma:
                y = np.random.randint(0, self.domain_size)
                counter += 1

            randomized_data[i] = y

        print("PrivacyBlanket: randomized {} times".format(counter))

        return randomized_data

    def shuffler(self, randomized_data):
        # print(randomized_data[:10])
        np.random.shuffle(randomized_data)
        # print(randomized_data[:10])
        return randomized_data

    def analyzer(self, shuffled_data):
        ESTIMATE_DIST = np.zeros(self.domain_size)
        unique, counts = np.unique(shuffled_data, return_counts=True)

        for i in range(len(unique)):
            ESTIMATE_DIST[unique[i]] = counts[i]

        return ESTIMATE_DIST
