import os
import matplotlib.pyplot as plt
import numpy as np
import virl
if not os.path.exists('Random'):
    os.mkdirs("Random")

def plot_learning_curve(scores, x, figure_file):
    plt.figure()
    running_avg = np.zeros(len(scores))
    # for i in range(len(running_avg)):
    #     running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, scores)
    plt.title("Random")
    plt.savefig(figure_file)

def evaluate_random(stochastic = False, noisy = False):
    problems = [0,4]
    for problem in problems:
        env = virl.Epidemic(stochastic = stochastic, noisy = noisy, problem_id=problem)
        
        states = []
        rewards = []
        info = []
        done_info = []
        done = False
        
        s = env.reset()
        states.append(s)
        print(states)
        # for i in range(episodes):
        #     reward_ep = 0
        while not done:
            s,r,done,i = env.step(action= np.random.choice(env.action_space.n))
            states.append(s)
            done_info.append(done)
            info.append(i)
            rewards.append(r)
        # x = [i+1 for i in range(episodes)]
        # if(stochastic):
        #     plot_learning_curve(rewards, x, "Random_stochastic/random_" + str(problem) + "_stochastic.png")
        # elif(noisy):
        #     plot_learning_curve(rewards, x, "Random_noisy/random_" + str(problem) + "_noisy.png")
        # else:            
        #     plot_learning_curve(rewards, x, "Random/random_" + str(problem) + ".png")
        fig,axes = plt.subplots(1,2,figsize=(20,8))
        labels = ['s[0]: susceptible','s[1]: infectious','s[2]: quarantined','s[3]: recovereds']
        states = np.array(states)
        for i in range(4):
            axes[0].plot(states[:,i],label = labels[i])
        
        axes[0].set_xlabel('weeks since start of epidemic')
        axes[0].set_ylabel('State s(t)')
        axes[0].legend()
        axes[1].plot(rewards);
        axes[1].set_title('Reward')
        axes[1].set_xlabel('weeks since start of epidemic')
        axes[1].set_ylabel('reward r(t)')
        fig.savefig("Random/random_" + str(problem) + ".png")
        print('total reward', np.sum(rewards))