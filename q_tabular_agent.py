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
from numpy import inf
import os
import logging
if not os.path.exists('Q_tabular'):
    os.mkdir("Q_tabular")

class QAgent():
    def __init__(self,dim_size,gamma,epsilon,epsilon_decay,epsilon_min,episodes,lr,bins,Nsa,Q):
        super().__init__()
        self.dim_size  = dim_size
        self.gamma= gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.episodes = episodes
        self.lr = lr
        self.bins = bins
        self.Nsa = Nsa
        self.Q = Q
        
    def train(self,env,bins):
        Q = self.Q
        dim_size = self.dim_size
        epsilon = self.epsilon
        alpha = self.lr
        Nsa = self.Nsa
        gamma = self.gamma
        Q = np.ones([dim_size + 1,dim_size + 1,dim_size + 1,dim_size + 1,env.action_space.n])
        random_count = 0
        rewards = []
        epsilons = []
        for epoch in tqdm(range(self.episodes)):
            state = env.reset()
            done = False
            reward_ep = 0
            while not done:
                state_dig = np.digitize(state,bins)
                if(np.random.rand() < epsilon):
                    random_count+=1
                    action = env.action_space.sample()
                else:
                    action = np.argmax(Q[(*state_dig,None)])
                next_state, reward, done, _ = env.step(action)
                next_state_dig = np.digitize(next_state,bins)
                Nsa[(*next_state_dig,action)] +=1
                Q[(*next_state_dig,action)] = Q[(*state_dig,action)] + alpha(Nsa[(*next_state_dig,action)])*(reward + gamma*np.max(Q[(*next_state_dig,None)]) - Q[(*state_dig,action)])
                reward_ep += reward
                state = next_state
                if epsilon >= self.epsilon_min:
                    epsilon *= self.epsilon_decay
            epsilons.append(epsilon)
            rewards.append(reward_ep)
        self.Q = Q
        return rewards
    
def plot_learning_curve(scores, x, figure_file):
    running_avg = np.zeros(len(scores))
    fig = plt.figure()
    axes = fig.add_subplot(111)
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-50):(i+1)])
    axes.plot(x, running_avg)
    axes.title.set_text('Running average of previous 100 scores')
    fig.savefig(figure_file)
    
def evaluate_q_tabular(episodes,stochastic_train = False, noisy_train = False,stochastic_test = False, noisy_test = False):
    problems = [0,1,2,3,4,5,6,7,8,9]
    rewards_per_problem = []
    for problem in problems:
        env = virl.Epidemic(stochastic = stochastic_train, noisy = noisy_train,problem_id = problem)
        dim_size = 9
        factor = 6e8**(1/dim_size)
        bins= [factor**x for x in range(1,dim_size+1)]
        Q = np.ones([dim_size + 1,dim_size + 1,dim_size + 1,dim_size + 1,env.action_space.n])
        q_agent_1 = QAgent(Q = Q,dim_size = dim_size,gamma = 0.65,epsilon = 0.4,epsilon_decay = 0.99993,epsilon_min  = 0.001,episodes = episodes,lr = lambda n: 60./(59+n),
                        bins = bins,Nsa = defaultdict(float))
        q_agent_2 = QAgent(Q = Q,dim_size = dim_size,gamma = 0.65,epsilon = 0.4,epsilon_decay = 0.99993,epsilon_min  = 0.001,episodes = episodes,lr = lambda x: 60./(59+x),
                        bins = bins,Nsa = defaultdict(float))
        q_agent_3 = QAgent(Q = Q,dim_size = dim_size,gamma = 0.65,epsilon = 0.4,epsilon_decay = 0.99993,epsilon_min  = 0.001,episodes = episodes,lr = lambda x: 60./(59+x),
                        bins = bins,Nsa = defaultdict(float))
        print("Training agent 1 on problem " + str(problem) + "...")
        rewards_1 = q_agent_1.train(env,bins)
        print("Training agent 2 on problem " + str(problem) + "...")
        rewards_2 = q_agent_2.train(env,bins)
        print("Training agent 3 on problem " + str(problem) + "...")
        rewards_3 = q_agent_3.train(env,bins)
        rewards_final = (np.array(rewards_1) + np.array(rewards_2) + np.array(rewards_3))/3
        x = [x for x in range(q_agent_1.episodes)]
        if(noisy_train == True):
            plot_learning_curve(rewards_final,x,"Q_tabular/q_tabular_train_" + str(problem) + "_noisy.png")
        else:
            plot_learning_curve(rewards_final,x,"Q_tabular/q_tabular_train_" + str(problem) + ".png")
        best_agent = np.argmax([np.sum(rewards_1),np.sum(rewards_2),np.sum(rewards_3)])
        best_agent_train = best_agent + 1
        best_agent_test = None
        if(best_agent_train == 1):
            best_agent_test = q_agent_1
        elif(best_agent_train == 2):
            best_agent_test = q_agent_2
        else:
            best_agent_test = q_agent_3
        env_test = virl.Epidemic(stochastic = stochastic_test,noisy=noisy_test,problem_id=problem)
        test_epochs = 1
        best_agent_test.Q[best_agent_test.Q == 1] = -1e2
        rewards_test = []
        actions_taken = []
        print("Testing agent on problem " + str(problem) + "...")
        unknown_states = 0
        states = []
        for i in tqdm(range(test_epochs)):
            state = env_test.reset()
            done = False
            reward_ep = 0
            rewards_plotting = []
            states.append(state)
            while not done:
                    state_dig = np.digitize(state,bins)
                    if(np.sum(best_agent_test.Q[(*state_dig,None)]) == -4e2):
                        unknown_states+=1
                    action = np.argmax(best_agent_test.Q[(*state_dig,None)])
                    actions_taken.append(action)
                    next_state, reward, done, _ = env_test.step(action)
                    states.append(next_state)
                    reward_ep += reward
                    state = next_state
                    rewards_plotting.append(reward)
            rewards_test.append(len(rewards_plotting))
        if(noisy_test):
            figure_file = "Q_tabular/q_tabular_noisy_" + str(problem) + '_test.png'
        elif(stochastic_test):
            figure_file = "Q_tabular/q_tabular_stochastic_" + str(problem) + '_test.png'
        else:
            figure_file = "Q_tabular/q_tabular_" + str(problem) + '_test.png'
        fig,axes = plt.subplots(1,2,figsize=(20,8))
        labels = ['s[0]: susceptible','s[1]: infectious','s[2]: quarantined','s[3]: recovereds']
        states = np.array(states)
        for i in range(4):
            axes[0].plot(states[:,i],label = labels[i])
        
        axes[0].set_xlabel('weeks since start of epidemic')
        axes[0].set_ylabel('State s(t)')
        axes[0].legend()
        axes[1].plot(rewards_plotting);
        axes[1].set_title('Reward')
        axes[1].set_xlabel('weeks since start of epidemic')
        axes[1].set_ylabel('reward r(t)')
        fig.savefig(figure_file)
        print('total reward', np.sum(rewards_plotting))
        rewards_per_problem.append(np.sum(rewards_plotting))
        
    return rewards_per_problem