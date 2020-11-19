import os
import matplotlib.pyplot as plt
import numpy as np
import virl
from math import log
import pandas as pd

env = virl.Epidemic(stochastic = False, noisy = False)

print("Observations/States: " + str(env.observation_space))
print("Actions: " + str(env.action_space))
print("Rewards: " + str(env.reward_range))

states = []
rewards = []

Q = np.zeros([env.observation_space.shape[0],env.action_space.n])
alpha = 0.5
gamma = 0.2
episodes = 1
bins = [0,1,2,3]
for i in range(episodes):
    state = env.reset()
    done = False
    reward_ep = 0
    while not done:
        if(state[2] == 0):
            state[2] = 1
        state_log = np.log(state)
        # print(state)
        state_digitized = pd.cut(x = state_log,
                            bins = [-1,5,10,15,21], 
                            labels = bins)
        # print(state_digitized)
    #     bins_indeces = np.digitize(state, bins)
    #     print(bins_indeces)
        print(state_digitized)
        action = np.argmax(Q[state_digitized,:])
        print(Q[state_digitized,:])
        print(action)
        new_state, reward, done, _ = env.step(action)
        new_state_log = np.log(new_state)
        new_state_digitized = pd.cut(x = new_state_log,
                            bins = [-1,5,10,15,21], 
                            labels = bins)
    #     _, new_state = pd.qcut(np.log(new_state)/np.log(1000), q=4,retbins = True)
        Q[state_digitized,action] = Q[state_digitized,action] + alpha*(reward + gamma*np.max(Q[new_state_digitized,:]) - Q[state_digitized,action])
        reward_ep += reward
        state = new_state
    rewards.append(reward_ep)

print(rewards)