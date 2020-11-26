import os
import matplotlib.pyplot as plt
import numpy as np
import virl

env = virl.Epidemic(stochastic = False, noisy = False)

states = []
rewards = []
info = []
done_info = []
done = False

s = env.reset()
states.append(s)
print(states)

while not done:
    s,r,done,i = env.step(action= np.random.choice(env.action_space.n))
    states.append(s)
    rewards.append(r)
    done_info.append(done)
    info.append(i)

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

print('total reward', np.sum(rewards))