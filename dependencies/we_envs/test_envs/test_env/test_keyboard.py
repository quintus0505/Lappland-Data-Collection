import gym
import we_envs
import numpy as np

env_name = 'Adroit-relocate-v0'


env = gym.make(env_name)

env = env.unwrapped


# while True:
#     action = np.zeros(19, dtype=np.float32)
#     # action[0:6] = [0.01, 0.01, -0.01, 0.0, 0.0, 0.0]
#     action[0:6] = [0, 0, 0, 0.2, 0.2, 0.2]
#     # env.sim.data.ctrl[12] = 5
#     # env.sim.data.ctrl[14] = 5
#     env.sim.step()
#     obs, reward, done, info = env.step(action)
#     # touchforce = env.getTouchforce()
#     # print('touch_force:', touchforce)
#     # fforce, tforce =  env.getFTforce()
#     # print('ft_force:', fforce[:3])
#     # keyboard_raw_data = env.get_raw_keyboardData()
#     # space_len = keyboard_raw_data['esc']
#     # if space_len>0.0085:
#     # print(space_len)
#     env.display_keyboardData(10)
#     env.render()

import time
for i in range(100):
    env.reset()
    # time.sleep(1)
    # print(env.goal)
    for j in range(50):
        # action = np.zeros(19, dtype=np.float32)
        # # action[0:6] = [0.01, 0.01, -0.01, 0.0, 0.0, 0.0]
        # action[0:6] = [0.00, 0, -0.001, 0.0, 0.0, 0.0]
        # # env.sim.data.ctrl[12] = 5
        # # env.sim.data.ctrl[14] = 5
        # env.sim.step()
        # obs, reward, done, info = env.step(action)
        # # touchforce = env.getTouchforce()
        # # print('touch_force:', touchforce)
        # fforce, tforce =  env.getFTforce()
        # # print('ft_force:', fforce[:3])
        # # keyboard_raw_data = env.get_raw_keyboardData()
        # # space_len = keyboard_raw_data['esc']
        # # if space_len>0.0085:
        # # print(space_len)
        # if info['dangerous']:
        #     print("dangerous!!")
        # # env.display_keyboardData(10)
        env.render()