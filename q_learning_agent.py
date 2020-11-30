import os
import matplotlib.pyplot as plt
import numpy as np
import virl
from math import log, isnan
import pandas as pd
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython import get_ipython
from collections import defaultdict
from bayes_opt import BayesianOptimization
from numpy import inf
# get_ipython().run_line_magic('matplotlib', 'inline')

env = virl.Epidemic(stochastic = False, noisy = True,problem_id = 0)

print("Observations/States: " + str(env.observation_space))
print("Actions: " + str(env.action_space))
print("Rewards: " + str(env.reward_range))


dim_size = 100
Q = np.ones([dim_size + 1,dim_size + 1,dim_size + 1,dim_size + 1,env.action_space.n])
gamma = 0.65
epsilon = 0.4
epsilon_decay = 0.99993
epsilon_min = 0.001
episodes = 1000
factor = 1e8**(1/dim_size)
Nsa = defaultdict(float)
alpha = lambda n: 60./(59+n)
alpha_lr = lambda x: x/episodes
alpha_TD = lambda x: 1/x
pbounds = {'n': (0, episodes)}
TD_values = []

# optimizer = BayesianOptimization(
#     f=alpha,
#     pbounds=pbounds,
#     random_state=1,
# )
bins = [factor**x for x in range(1,dim_size + 1)]
states = []
rewards = []
Q_error = []
random_count = 0
epsilons = []
alpha_cte = 0.01
alphas = []
print("\nTraining Q agent...")
for i in tqdm(range(episodes)):
    # for problem in range(0,10):
        # env = virl.Epidemic(stochastic = False, noisy = True,problem_id = problem)
        state = env.reset()
        done = False
        reward_ep = 0
        rewards_plotting = []
        alpha_ep = 0
        # optimizer.maximize(init_points=2,n_iter=1,)
        while not done:
            state_digitized = np.digitize(state,bins)
            if(np.random.rand() < epsilon):
                random_count+=1
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state_digitized[0],state_digitized[1],state_digitized[2],state_digitized[3],:])
            new_state, reward, done, _ = env.step(action)
            new_state_digitized = np.digitize(new_state,bins)
            TD = ((reward + gamma*np.max(Q[new_state_digitized[0],new_state_digitized[1],new_state_digitized[2],new_state_digitized[3],:]) - Q[state_digitized[0],state_digitized[1],state_digitized[2],state_digitized[3],action]))
            # TD_values.append(TD)
            Nsa[new_state_digitized[0],new_state_digitized[1],new_state_digitized[2],new_state_digitized[3],action] +=1
            Q[new_state_digitized[0],new_state_digitized[1],new_state_digitized[2],new_state_digitized[3],action] = Q[state_digitized[0],state_digitized[1],state_digitized[2],state_digitized[3],action] + alpha(Nsa[new_state_digitized[0],new_state_digitized[1],new_state_digitized[2],new_state_digitized[3],action])*(reward + gamma*np.max(Q[new_state_digitized[0],new_state_digitized[1],new_state_digitized[2],new_state_digitized[3],:]) - Q[state_digitized[0],state_digitized[1],state_digitized[2],state_digitized[3],action])
            # Q_error.append(alpha(i)*(reward + gamma*np.max(Q[new_state_digitized[0],new_state_digitized[1],new_state_digitized[2],new_state_digitized[3],:]) - Q[state_digitized[0],state_digitized[1],state_digitized[2],state_digitized[3],action]))
            # Q[new_state_digitized[0],new_state_digitized[1],new_state_digitized[2],new_state_digitized[3],action] = Q[state_digitized[0],state_digitized[1],state_digitized[2],state_digitized[3],action] + alpha(Nsa[new_state_digitized[0],new_state_digitized[1],new_state_digitized[2],new_state_digitized[3],action])*(reward + gamma*np.max(Q[new_state_digitized[0],new_state_digitized[1],new_state_digitized[2],new_state_digitized[3],:]) - Q[state_digitized[0],state_digitized[1],state_digitized[2],state_digitized[3],action])
            # Q[new_state_digitized[0],new_state_digitized[1],new_state_digitized[2],new_state_digitized[3],action] = Q[state_digitized[0],state_digitized[1],state_digitized[2],state_digitized[3],action] + alpha_cte*(reward + gamma*np.max(Q[new_state_digitized[0],new_state_digitized[1],new_state_digitized[2],new_state_digitized[3],:]) - Q[state_digitized[0],state_digitized[1],state_digitized[2],state_digitized[3],action])
            alpha_ep+=alpha(Nsa[new_state_digitized[0],new_state_digitized[1],new_state_digitized[2],new_state_digitized[3],action])
            reward_ep += reward
            state = new_state
            rewards_plotting.append(reward) # For episode = 1, if > 1, delete and use rewards []
            if epsilon >= epsilon_min:
                epsilon *= epsilon_decay
        epsilons.append(epsilon)
        rewards.append(reward_ep)
        alphas.append(alpha_ep)

print("Random actions chosen: " + str(((random_count/(52*episodes)))*100) + "%")

print("\nFinal state: " + "\nSusceptible: " + str(state[0]) + "\nInfectious: " + str(state[1])
      + "\nQuarantined: " + str(state[2]) + "\nRecovered: " + str(state[3]))
f1 = plt.figure() 
f2 = plt.figure()
f3 = plt.figure()
ax3 = f3.add_subplot(111)
ax3.plot(alphas)
ax2 = f2.add_subplot(111)
ax2.plot(epsilons)
ax1 = f1.add_subplot(111)
ax1.plot(rewards)
final_reward = rewards[len(rewards) - 1]
# print("First TD: " + str(TD_values[0]))
# print("Last TD: " + str(TD_values[len(TD_values) - 1]))
print(final_reward)
print("FINAL EPSILON: " + str(epsilon))
env = virl.Epidemic(stochastic = False, noisy = True,problem_id=0)
print("\nTesting Q agent...")
episodes = 1000
rewards = []
Q[Q == 1] = -inf
for i in tqdm(range(episodes)):
    state = env.reset()
    done = False
    reward_ep = 0
    rewards_plotting = []
    while not done:
            state_digitized = np.digitize(state,bins)
            action = np.argmax(Q[state_digitized[0],state_digitized[1],state_digitized[2],state_digitized[3],:])
            new_state, reward, done, _ = env.step(action)
            reward_ep += reward
            state = new_state
            rewards_plotting.append(reward)
    rewards.append(reward_ep)

print("\nFinal state: " + "\nSusceptible: " + str(state[0]) + "\nInfectious: " + str(state[1])
      + "\nQuarantined: " + str(state[2]) + "\nRecovered: " + str(state[3]))
f3 = plt.figure()
ax3 = f3.add_subplot(111)
ax3.plot(rewards)
plt.show()
final_reward = rewards[len(rewards) - 1]
print(final_reward)