import os
import matplotlib.pyplot as plt
import numpy as np
import virl

def policy(states,actions):
    print("")

env = virl.Epidemic(stochastic = False, noisy = False)

print("Observations/States: " + str(env.observation_space))
print("Actions: " + str(env.action_space))
print("Rewards: " + str(env.reward_range))

states = []
rewards = []
done = False

state = env.reset()
print("\nInitial state: " + "\nSusceptible: " + str(state[0]) + "\nInfectious: " + str(state[1])
      + "\nQuarantined: " + str(state[2]) + "\nRecovered: " + str(state[3]))
states.append(state)

actions = np.linspace(0,3,4)
print(actions)

while not done:
    for action in actions:
        s,r,done,i = env.step(action = int(action))
        states.append(s)
        rewards.append(r)
        
print("\nFinal state: " + "\nSusceptible: " + str(s[0]) + "\nInfectious: " + str(s[1])
      + "\nQuarantined: " + str(s[2]) + "\nRecovered: " + str(s[3]))

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