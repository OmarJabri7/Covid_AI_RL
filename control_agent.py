import gym

env = gym.make('Acrobot-v1')
env.reset()
for i in range(500):
    env.render()
    states, rewards, done, info = env.step(env.action_space.sample())
print(info)