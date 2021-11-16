import gym
import we_envs
import numpy as np

env_name = 'Adroit-relocate-v0'
env_name = 'Adroit-door-v0'
env_name = 'Adroit-hammer-v0'
env_name = 'Adroit-pen-v0'
env = gym.make(env_name)

# env = env.unwrapped # don't need
# sim = env.sim
# for i in range(1000):
#     sim.step()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_steps = env._max_episode_steps



num_episodes=10

act_dim = env.env.action_space.shape[0]
for ep in range(num_episodes):
    o = env.reset()
    d = False
    t = 0
    # print(env.spec.timestep_limit)
    while True: #t < env.spec.timestep_limit and d is False:
        env.env.mj_render()
        a = np.random.uniform(low=-.1, high=0.1, size=act_dim)
        o, r, d, _ = env.step(a)
        t = t+1