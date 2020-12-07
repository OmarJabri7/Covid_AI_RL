import os
import matplotlib.pyplot as plt
import numpy as np
import virl            

def policy(states,actions,week,state):
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
    
def evaluate_deterministic_model(stochastic,noisy):
    
    problems = [0,1,2,3,4,5,6,7,8,9]
    for problem in problems:
        env = virl.Epidemic(stochastic = stochastic, noisy = noisy, problem_id = problem)
        
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
            action = policy(state,actions,week,state)
            state,r,done,i = env.step(action = (action))
            states.append(state)
            rewards.append(r)
            week+=1
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
        figurefile = None
        if(noisy):
            figurefile = "deterministic_noisy_" + str(problem) + ".png"
        elif(stochastic):
            figurefile = "deterministic_stochastic_" + str(problem) + ".png"
        else:
            figurefile = "deterministic_" + str(problem) + ".png"
        fig.savefig("Deterministic/"  + figurefile)
        print('Total reward', np.sum(rewards))