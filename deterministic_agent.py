import os
import matplotlib.pyplot as plt
import numpy as np
import virl

def policy_1(states,actions,week):
    print("\nStep state (" + str(week) + "): " + "\nSusceptible: " + str(state[0]) + "\nInfectious: " + str(state[1])
      + "\nQuarantined: " + str(state[2]) + "\nRecovered: " + str(state[3]))
    susceptible = states[0]
    infected = states[1]
    quarantined = states[2]
    recovereds = states[3]
    none = actions[0]
    full_lockdown = actions[1]
    track_and_trace = actions[2]
    social_distancing = actions[3]
    max_state = max(states) # Max is always going to be susceptible (Do not use)
    min_state = min(states)
    if(infected > recovereds):
        print("Action chosen: track and trace")
        return track_and_trace
    elif(min_state == quarantined):
        print("Action chosen: full lockdown")
        return full_lockdown
    elif(recovereds > infected):
        print("Action chosen: social distancing")
        return social_distancing
    
    
    

def policy_2(states,actions,week):
    susceptible = states[0]
    infected = states[1]
    quarantined = states[2]
    max_state = max(states)
    min_state = min(states)
    if(max_state == infected):
        return 1
    elif(min_state == quarantined):
        return 2
    else:
        return 3

def policy_3(states,actions,week):
    print("\nStep state (" + str(week) + "): " + "\nSusceptible: " + str(state[0]) + "\nInfectious: " + str(state[1])
      + "\nQuarantined: " + str(state[2]) + "\nRecovered: " + str(state[3]))
    susceptible = states[0]
    infected = states[1]
    quarantined = states[2]
    recovereds = states[3]
    none = actions[0]
    full_lockdown = actions[1]
    track_and_trace = actions[2]
    social_distancing = actions[3]
    max_state = max(states) # Max is always going to be susceptible (Do not use)
    min_state = min(states)
    if(infected < quarantined):
        print("Action chosen: track and trace")
        return track_and_trace
    elif(quarantined < infected):
        print("Action chosen: full lockdown")
        return full_lockdown
    else:
        print("Action chosen: social distancing")
        return social_distancing
    
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

actions = [0,1,2,3]
print(actions)
week = 0
while not done:
    action = policy_3(state,actions,week)
    state,r,done,i = env.step(action = (action))
    states.append(state)
    rewards.append(r)
    week+=1
#Polic 1: reward = -0.87
#Policy 2: reward = -1.55
#Policy 3: reward = -0.83 (!) Problem is that when we put trackandtrace, we dont have social dis, (vice versa)
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
axes[1].plot(rewards)
axes[1].set_title('Reward')
axes[1].set_xlabel('weeks since start of epidemic')
axes[1].set_ylabel('reward r(t)')

print('Total reward', np.sum(rewards))