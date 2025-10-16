# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 07:23:51 2025

@author: jesus
"""

import numpy as np # aplicacion de matematicas
import matplotlib.pyplot as plt # mostrar datos
import pandas as pd # manipular datos

import math
import random

dataset = pd.read_csv('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/Ads_CTR_Optimisation.csv')

N = 10000
d = 10
ads_selected = []
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward


plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()