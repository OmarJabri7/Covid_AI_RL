import os
import matplotlib.pyplot as plt
import numpy as np
import virl
from math import log, isnan
import pandas as pd
from itertools import permutations

def listToString(s):  
    
    # initialize an empty string 
    str1 = ""  
    
    # traverse in the string   
    for ele in s:  
        str1 += str(ele)   
    
    # return string   
    return str1  


env = virl.Epidemic(stochastic = False, noisy = False)

print("Observations/States: " + str(env.observation_space))
print("Actions: " + str(env.action_space))
print("Rewards: " + str(env.reward_range))

states = []
rewards = []
rewards_plotting = []
Q = np.array([])
alpha = 0.1
gamma = 0.9
episodes = 10
bins = np.array([0,1,2,3])
res = permutations(bins)
res = ((list(res)))
import itertools
result= []
for (i,j,k,p) in itertools.product(range(4),range(4),range(4),range(4)):
    result.append([i,j,k,p])
print("Number of combinations:",len(result))
Q = np.array(result)
Q_table = {}
for i in range(len(Q)):
    Q_table[listToString(Q[i])] = [0,0,0,0]
for i in range(episodes):
    state = env.reset()
    done = False
    reward_ep = 0
    while not done:
        if(state[2] == 0):
            state[2] = 1
        state_log = np.log(state)
        state_digitized = pd.cut(x = state_log,
                            bins = [-5,5,10,15,21],
                            labels = bins)
        state_digitized = [0 if isnan(x) else x for x in state_digitized] #! why NaN?
        keys_states = listToString(state_digitized)
        action = np.argmax(Q_table[keys_states])
        new_state, reward, done, _ = env.step(action)
        new_state_log = np.log(new_state)
        new_state_digitized = pd.cut(x = new_state_log,
                            bins = [-1,5,10,15,21],
                            labels = bins)
        states.append(new_state)
        new_state_digitized = [0 if isnan(x) else x for x in new_state_digitized] #! why NaN?
        new_keys_states = listToString(new_state_digitized)
        Q_table[new_keys_states][action] = Q_table[keys_states][action] + alpha*(reward + gamma*np.max(Q_table[new_keys_states][:]) - Q_table[keys_states][action])
        reward_ep += reward
        state = new_state
        rewards_plotting.append(reward) # For episode = 1, if > 1, delete and use rewards []
    rewards.append(reward_ep)

print("\nFinal state: " + "\nSusceptible: " + str(state[0]) + "\nInfectious: " + str(state[1])
      + "\nQuarantined: " + str(state[2]) + "\nRecovered: " + str(state[3]))

fig,axes = plt.subplots(1,2,figsize=(20,8))
labels = ['s[0]: susceptible','s[1]: infectious','s[2]: quarantined','s[3]: recovereds']
states = np.array(states)
for i in range(4):
    axes[0].plot(states[:,i],label = labels[i])

axes[0].set_xlabel('weeks since start of epidemic')
axes[0].set_ylabel('State s(t)')
axes[0].legend()
axes[1].plot(rewards_plotting)
axes[1].set_title('Reward')
axes[1].set_xlabel('weeks since start of epidemic')
axes[1].set_ylabel('reward r(t)')
print("Rewards per episode: " + str(rewards))
print(max(rewards))