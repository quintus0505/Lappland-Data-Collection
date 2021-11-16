import gym
import tianshou as ts
import numpy as np
print(gym.__version__)
env = gym.make('State-Ant-v2')
env = env.unwrapped
env.reset()

actuation = -30*3.1415926/180
count = 1000

for i in range(count):


    # print(np.array(full_state['joint_angle']*180/3.1415926).round())

    # desired_state = env.controllable_state_space.sample()
    desired_state = np.zeros(8, dtype=np.float)
    desired_state[1] = 1.22
    desired_state[3] = -1.22
    desired_state[5] = -1.22
    desired_state[7] = 1.22
    desired_state[0]=actuation+i/count*60*3.1415926/180
    desired_state[2]=actuation+i/count*60*3.1415926/180
    desired_state[4]=actuation+i/count*60*3.1415926/180
    desired_state[6]=actuation+i/count*60*3.1415926/180

    observation, reward, done, info = env.state_step(desired_state)
    full_state = env.get_full_state()
    actual_angle = full_state['joint_angle']
    print(np.array((actual_angle-desired_state)*180/3.1415926).round())

    # desire_state = np.zeros(8, dtype=np.float)
    # desire_state[1] = 1.22
    # desire_state[3] = -1.22
    # desire_state[5] = -1.22
    # desire_state[7] = 1.22
    # desire_state[0]=actuation+i/100*60*3.1415926/180
    # observation, reward, done, info = env.state_step(desire_state)
    # print(desire_state[0])
    # print(f"disired angle: {desire_state[1]}, actual angle: {joint_angle[1]}")
    env.render()

