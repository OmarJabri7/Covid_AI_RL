import torch as T
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import virl
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
from collections import defaultdict
from torch.optim.lr_scheduler import StepLR
from bayes_opt import BayesianOptimization

class DeepQNetwork(nn.Module):
    def _huber_loss(self,y_true, y_pred, clip_delta=0.01):
        """
        Huber loss (for use in Keras), see https://en.wikipedia.org/wiki/Huber_loss
        The huber loss tends to provide more robust learning in RL settings where there are 
        often "outliers" before the functions has converged.
        """
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta
        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
        return K.mean(tf.where(cond, squared_loss, quadratic_loss))
        # sqrt(1+error^2)-1
        #error = prediction - target
        #return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, 
            n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        # self.dropout = nn.Dropout(0.5)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=2, gamma=0.1)
        self.loss = nn.SmoothL1Loss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state.float()))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions

class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
            max_mem_size=1000000, eps_end=0.01, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100

        self.Q_eval = DeepQNetwork(lr, n_actions=n_actions, input_dims=input_dims,
                                    fc1_dims=64, fc2_dims=64)
        self.Q_next = DeepQNetwork(lr, n_actions=n_actions, input_dims=input_dims,
                                    fc1_dims=64, fc2_dims=64)

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()
        
        
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma*T.max(q_next,dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.Q_eval.scheduler.step()
        self.iter_cntr += 1
        self.epsilon = self.epsilon * self.eps_dec if self.epsilon > self.eps_min \
                       else self.eps_min

        #if self.iter_cntr % self.replace_target == 0:
        #   self.Q_next.load_state_dict(self.Q_eval.state_dict())
def plotLearning(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)
    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Episode", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    #ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)
        
def evaluate_model():
    for i in range(1,episodes + 1):
        score = 0
        done = False
        state = env.reset()
        reward = 0
        while not done:
            action = agent.choose_action(state/6e8)
            old_reward = reward
            new_state, reward, done, info = env.step(action)
            new_state/=6e8
            diff_reward = abs(reward) - abs(old_reward)
            score+=reward
            agent.store_transition(state,action,reward,new_state,done)
            agent.learn()
            state = new_state
        alphas.append(agent.lr)
        scores.append(score)
        eps_hist.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.4f'% score,
              'average score %.4f' % avg_score, 
              'epsilon %.4f' %agent.epsilon,'alpha %.4f' %agent.lr)
        x = [i+1 for i in range(episodes)]
        filename = "epidemic_reward.png"
    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    ax1.plot(scores)
    f2 = plt.figure()
    ax2 = f2.add_subplot(111)
    ax2.plot(eps_hist)
    plt.show()
    # plt.figure()
    # plt.plot(alphas,'r:')
    
env = virl.Epidemic(stochastic = False, noisy = False)
agent = Agent(gamma =0.99,epsilon = 1.0,eps_end = 0.01,eps_dec = 0.9993, batch_size = 64,n_actions = 4,input_dims = [4],
              lr = 0.001)
scores, eps_hist = [], []       
episodes = 500
optimize_lr = lambda x,y: x+y
alphas = []
sigma = 0.0001
evaluate_model()