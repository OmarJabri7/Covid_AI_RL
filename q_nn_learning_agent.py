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
from collections import namedtuple
from tqdm import tqdm
import os
if not os.path.exists('Q_nn_plots'):
    os.mkdir("Q_nn_plots")
if not os.path.exists('Q_nn_plots_noisy'):
    os.mkdir('Q_nn_plots_noisy')

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()

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
        # self.dropout = nn.Dropout(0.3)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=2, gamma=0.9)
        self.loss = nn.SmoothL1Loss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state.float()))
        # x = F.dropout(x,training = True)
        x = F.relu(self.fc2(x))
        # x = F.dropout(x,training = True)
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
                                    fc1_dims=96, fc2_dims=96)
        self.Q_next = DeepQNetwork(lr, n_actions=n_actions, input_dims=input_dims,
                                    fc1_dims=96, fc2_dims=96)

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
    def choose_action_test(self, observation):
        state = T.tensor([observation]).to(self.Q_eval.device)
        actions = self.Q_eval.forward(state)
        action = T.argmax(actions).item()
        return action

    def learn(self,epoch):
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

        if self.iter_cntr % self.replace_target == 0:
          self.Q_next.load_state_dict(self.Q_eval.state_dict())
          
        torch.save({
            'epoch': epoch + 1,
            'valid_loss_min': loss,
            'state_dict': self.Q_eval.state_dict(),
            'optimizer': self.Q_eval.optimizer.state_dict(),
            }, "covid_nn.h5")
        
    
    def test(self,epoch):
        self.Q_eval = torch.load("covid_nn.h5")
        
        
        
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
        
def evaluate_model(env,episodes,agent):
    actions_taken = []
    scores, eps_hist = [], []
    factor = 6e8
    for i in tqdm(range(1,episodes+1)):
        score = 0
        done = False
        state = env.reset()
        reward = 0
        t = 0
        while not done:
            action = agent.choose_action(state/factor)
            actions_taken.append(action)
            old_reward = reward
            new_state, reward, done, info = env.step(action)
            new_state/=factor
            diff_reward = abs(reward) - abs(old_reward)
            score+=reward
            agent.store_transition(state,action,reward,new_state,done)
            agent.learn(i)
            state = new_state
            t+=1
        scores.append(score)
        eps_hist.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        # print('episode ', i, 'score %.4f'% score,
        #       'average score %.4f' % avg_score, 
        #       'epsilon %.4f' %agent.epsilon,'alpha %.4f' %agent.lr)
        x = [i+1 for i in range(episodes)]
        filename = "epidemic_2_reward.png"
    print("Action 0 taken:" + str(actions_taken.count(0)) + " times")
    print("Action 1 taken:" + str(actions_taken.count(1)) + " times")
    print("Action 2 taken:" + str(actions_taken.count(2)) + " times")
    print("Action 3 taken:" + str(actions_taken.count(3)) + " times")
    return scores,eps_hist

import pandas as pd
def plot_episode_stats(stats, episodes,smoothing_window):
    # Plot the episode length over time
    #fig1 = plt.figure(figsize=(10,5))
    plt.plot(episodes)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    plt.grid(True)
    plt.show()
    
    # Plot the episode reward over time
    
    #fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    plt.grid(True)
    plt.show()

def plot_learning_curve(scores, x, figure_file,title):
    running_avg = np.zeros(len(scores))
    fig = plt.figure()
    axes = fig.add_subplot(111)
    # for i in range(len(running_avg)):
    #     running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    axes.plot(x, scores)
    axes.set_title(title)
    fig.savefig(figure_file)
    
def evaluate_q_nn(episodes,stochastic_train = False,noisy_train = False,stochastic_test = False,noisy_test = False):
    rewards_per_problem = []
    problems = [0,1,2,3,4,5,6,7,8,9]
    for problem in problems:
        env = virl.Epidemic(stochastic = stochastic_train, noisy = noisy_train,problem_id = problem)
        episodes = episodes
        agent_1 = Agent(gamma =0.99,epsilon = 0.7,eps_end = 0.01,eps_dec = 0.9993, batch_size = 64,n_actions = 4,input_dims = [4],
                      lr = 0.001)
        scores_train_1,eps_hist_1 = evaluate_model(env,episodes,agent_1)
        agent_2 = Agent(gamma =0.99,epsilon = 0.7,eps_end = 0.01,eps_dec = 0.9993, batch_size = 64,n_actions = 4,input_dims = [4],
                      lr = 0.001)
        scores_train_2,eps_hist_2 = evaluate_model(env,episodes,agent_2)
        agent_3 = Agent(gamma =0.99,epsilon = 0.7,eps_end = 0.01,eps_dec = 0.9993, batch_size = 64,n_actions = 4,input_dims = [4],
                      lr = 0.001)
        scores_train_3,eps_hist_3 = evaluate_model(env,episodes,agent_3)
        scores_final = np.array((np.array(scores_train_1)+np.array(scores_train_2)+np.array(scores_train_3)))/3
        eps_hist_final = np.array((np.array(eps_hist_1) + np.array(eps_hist_2) + np.array(eps_hist_3)))/3
        x = [i+1 for i in range(episodes)]
        if(noisy_train == True):
            plot_learning_curve(scores_final,x,"Q_nn_plots_noisy/nn_train_q_" + str(problem) + ".png","Train on problem " + str(problem) + "_noisy")
        else:
            plot_learning_curve(scores_final,x,"Q_nn_plots/nn_train_q_" + str(problem) + ".png","Train on problem " + str(problem))
        best_agent_test = agent_1
        best_agent = np.argmax([np.sum(scores_train_1),np.sum(scores_train_2),np.sum(scores_train_3)])
        best_agent_train = best_agent + 1
        best_agent_test = None
        if(best_agent_train == 1):
            best_agent_test = agent_1
        elif(best_agent_train == 2):
            best_agent_test = agent_2
        else:
            best_agent_test = agent_3
        plot_learning_curve(scores_train_1,x,"Q_nn_plots/nn_train_q_" + str(problem) + ".png","Train on problem " + str(problem))
        test_epochs = 1
        rewards_test = []
        env = virl.Epidemic(stochastic = stochastic_test, noisy = noisy_test,problem_id=problem)
        actions_taken = []
        rewards_plotting = []
        states = []
        for i in range(test_epochs):
            state = env.reset()
            done = False
            reward_ep = 0
            states.append(state)
            while not done:
            #    state /=6e8 
               action =  best_agent_test.choose_action_test(state/6e8)
               next_state,reward,done,_ = env.step(action)
               states.append(next_state)
               rewards_plotting.append(reward)
               reward_ep+=reward
               state = next_state
            rewards_test.append(reward_ep)
        fig,axes = plt.subplots(1,2,figsize=(20,8))
        labels = ['s[0]: susceptible','s[1]: infectious','s[2]: quarantined','s[3]: recovereds']
        states = np.array(states)
        for i in range(4):
            axes[0].plot(states[:,i],label = labels[i])
        figurefile = None
        if(noisy_test):
            figurefile = "Q_nn_plots/nn_test_q_noisy" + str(problem) + ".png"
        elif(stochastic_test):
            figurefile = "Q_nn_plots/nn_test_q_stochastic" + str(problem) + ".png"
        else:
            figurefile = "Q_nn_plots/nn_test_q_" + str(problem) + ".png"
        axes[0].set_xlabel('weeks since start of epidemic')
        axes[0].set_ylabel('State s(t)')
        axes[0].legend()
        axes[1].plot(rewards_plotting);
        axes[1].set_title('Reward')
        axes[1].set_xlabel('weeks since start of epidemic')
        axes[1].set_ylabel('reward r(t)')
        fig.savefig(figurefile)
        print('total reward', np.sum(rewards_plotting))
        # x = [i+1 for i in range(test_epochs)]
        # if(noisy_test == True):
        #     plot_learning_curve(rewards_test,x,"Q_nn_plots_noisy/nn_test_q_" + str(problem) + ".png","Test on problem " + str(problem) + "_noisy")
        # elif(stochastic_test == True):
        #     plot_learning_curve(rewards_test,x,"Q_nn_plots_stochastic/nn_test_q_" + str(problem) + ".png","Test on problem " + str(problem) + "_stochastic")
        # else:
        #     plot_learning_curve(rewards_test,x,"Q_nn_plots/nn_test_q_" + str(problem) + ".png","Test on problem ")
        rewards_per_problem.append(np.sum(rewards_plotting))
    return rewards_per_problem