import gym
import matplotlib.pyplot as plt
import numpy as np
import virl  
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
plt.ioff()
import os
if not os.path.exists('Policy_Search'):
    os.mkdir("Policy_Search")


class PolicyNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)


        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)


        return x


class PolicyGradientAgent():
    def __init__(self, lr, input_dims, gamma=0.99, n_actions=4):
        self.gamma = gamma
        self.lr = lr
        self.reward_memory = []
        self.action_memory = []


        self.policy = PolicyNetwork(self.lr, input_dims, n_actions)


    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.policy.device)
        probabilities = F.softmax(self.policy.forward(state))
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)


        return action.item()


    def store_rewards(self, reward):
        self.reward_memory.append(reward)


    def learn(self):
        self.policy.optimizer.zero_grad()


        # G_t = R_t+1 + gamma * R_t+2 + gamma**2 * R_t+3
        # G_t = sum from k=0 to k=T {​​​​​​​gamma**k * R_t+k+1}​​​​​​​
        G = np.zeros_like(self.reward_memory, dtype=np.float64)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        G = T.tensor(G, dtype=T.float).to(self.policy.device)
        
        loss = 0
        for g, logprob in zip(G, self.action_memory):
            loss += -g * logprob
        loss.backward()
        self.policy.optimizer.step()


        self.action_memory = []
        self.reward_memory = []
        
def plot_learning_curve(scores, x, figure_file):
    plt.figure()
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-25):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

def evaluate_policy_search(episodes,stochastic_train = False, noisy_train = False, stochastic_test = False, noisy_test = False):
    problems = [0,4]
    rewards_per_problem = []
    for problem in problems:
        env = virl.Epidemic(stochastic = stochastic_train, noisy = noisy_train,problem_id = problem)
        episodes = episodes
        agent = PolicyGradientAgent(gamma=0.6, lr=0.00001, input_dims=[4],
                                    n_actions=4)
    
    
        figure_file = "Policy_Search/policy_" + str(problem) + '_train.png'
        scores = []
        for i in tqdm(range(episodes)):
            done = False
            observation = env.reset()
            score = 0
            while not done:
                observation/=6e8
                action = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                score += reward
                agent.store_rewards(reward)
                observation = observation_
            agent.learn()
            scores.append(score)
    
    
            avg_score = np.mean(scores[-100:])
            # print('episode ', i, 'score %.2f' % score,
            #         'average score %.2f' % avg_score)
    
    
        x = [i+1 for i in range(len(scores))]
        plot_learning_curve(scores, x, figure_file)
        env_test = virl.Epidemic(problem_id = problem)
        test_epochs = 1
        scores_test = []
        states = []
        rewards_plotting = []
        # for i in range(test_epochs):
        done = False
        observation = env.reset()
        score = 0
        states.append(observation)
        while not done:
            # observation/=6e8
            action = agent.choose_action(observation/6e8)
            observation_, reward, done, info = env.step(action)
            states.append(observation_)
            rewards_plotting.append(reward)
            score += reward
            agent.store_rewards(reward)
            observation = observation_
        scores_test.append(score)
        figure_file = "Policy_Search/policy_" + str(problem) + '_test.png'
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
        fig.savefig("Policy_Search/policy_" + str(problem) + ".png")
        print('total reward', np.sum(rewards_plotting))
        rewards_per_problem.append(np.sum(rewards_plotting))
    return rewards_per_problem